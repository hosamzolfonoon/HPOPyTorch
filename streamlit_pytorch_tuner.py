# streamlit_pytorch_tuner.py
# PyTorch version of your Streamlit TensorFlow/Keras dense-network tuner.
# Run with: streamlit run streamlit_pytorch_tuner.py

import io
import os
import random
import re
import string
import tempfile
import warnings
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from imblearn.over_sampling import RandomOverSampler, SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    IMBLEARN_AVAILABLE = True
except Exception:
    IMBLEARN_AVAILABLE = False

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")


# =========================================================
# General helpers
# =========================================================

def randstr(length: int = 8) -> str:
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))


def set_random_seed(seed: int = 42) -> int:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_tuning_inputs():
    # softmax is kept for compatibility with your original UI, but it is not recommended for hidden layers.
    activations = ["relu", "leaky_relu", "sigmoid", "tanh", "softmax"]
    batch_size_list = [16, 32, 64, 128, 256, 512]
    dense_input_list = [16, 32, 64, 128, 256, 512]
    seed_options = [0, 1, 42, 1337, 2024]
    test_sizes = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    return activations, batch_size_list, dense_input_list, seed_options, test_sizes


# =========================================================
# PyTorch model pieces
# =========================================================

class SoftmaxHidden(nn.Module):
    def forward(self, x):
        return torch.softmax(x, dim=1)


def activation_layer(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01)
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "tanh":
        return nn.Tanh()
    if name == "softmax":
        return SoftmaxHidden()
    raise ValueError(f"Unsupported activation: {name}")


class DenseClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        dense_input: int = 128,
        hidden_layer_number: int = 3,
        activation_input_hidden: str = "tanh",
        dropout_rate: Optional[float] = None,
        batch_norm: bool = False,
        staircase: bool = False,
    ):
        super().__init__()
        layers: List[nn.Module] = []

        current_dim = input_dim
        units = dense_input

        # First hidden block
        layers.append(nn.Linear(current_dim, units))
        if batch_norm:
            layers.append(nn.BatchNorm1d(units))
        layers.append(activation_layer(activation_input_hidden))
        if dropout_rate is not None and dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))

        # Additional hidden blocks
        if staircase:
            units = max(1, units // 2)

        for _ in range(hidden_layer_number):
            layers.append(nn.Linear(current_dim if len(layers) == 0 else layers_last_out_dim(layers), units))
            if batch_norm:
                layers.append(nn.BatchNorm1d(units))
            layers.append(activation_layer(activation_input_hidden))
            if dropout_rate is not None and dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            if staircase:
                units = max(1, units // 2)

        final_in = layers_last_out_dim(layers)
        layers.append(nn.Linear(final_in, num_classes))
        # Important: no Softmax here. CrossEntropyLoss expects raw logits.

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def layers_last_out_dim(layers: List[nn.Module]) -> int:
    for layer in reversed(layers):
        if isinstance(layer, nn.Linear):
            return layer.out_features
    raise ValueError("No Linear layer found.")


def build_model(input_dim: int, num_classes: int, config: Dict) -> nn.Module:
    return DenseClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        dense_input=config["dense_input"],
        hidden_layer_number=config["hidden_layer_number"],
        activation_input_hidden=config["activation_input_hidden"],
        dropout_rate=config["dropout_rate"],
        batch_norm=config["batch_normalization_check"],
        staircase=config["staircase"],
    )


# =========================================================
# Optimizer, regularization, callbacks
# =========================================================

def configure_regularizer() -> Dict:
    st.subheader("Kernel Regularizer")
    regularizer_type = st.selectbox("Select kernel regularization type:", ("None", "L1", "L2", "L1_L2"))
    l1_value = 0.0
    l2_value = 0.0

    if regularizer_type == "L1":
        l1_value = st.number_input("L1 value", min_value=0.0, value=0.01)
    elif regularizer_type == "L2":
        l2_value = st.number_input("L2 value", min_value=0.0, value=0.01)
    elif regularizer_type == "L1_L2":
        l1_value = st.number_input("L1 value", min_value=0.0, value=0.01, key="l1_value_l1l2")
        l2_value = st.number_input("L2 value", min_value=0.0, value=0.01, key="l2_value_l1l2")

    return {"type": regularizer_type, "l1": float(l1_value), "l2": float(l2_value)}


def configure_dropout_layer() -> Optional[float]:
    st.header("Configure Dropout Layer")
    use_dropout = st.checkbox("Use Dropout Layer", value=True)
    if use_dropout:
        return float(st.slider("Dropout Rate", min_value=0.0, max_value=1.0, value=0.6, step=0.05))
    return None


def configure_early_stopping() -> Dict:
    st.header("Configure EarlyStopping")
    use_es = st.checkbox("Use EarlyStopping", value=True)
    if not use_es:
        return {"use": False}
    monitor = st.selectbox("Metric to monitor", ["val_loss", "val_accuracy", "loss", "accuracy"], index=0)
    patience = int(st.number_input("Patience", value=6, min_value=0))
    min_delta = float(st.number_input("Minimum improvement", value=0.0))
    mode = st.selectbox("Mode", ["auto", "min", "max"], index=0)
    restore_best = st.checkbox("Restore best weights", value=False)
    return {
        "use": True,
        "monitor": monitor,
        "patience": patience,
        "min_delta": min_delta,
        "mode": mode,
        "restore_best": restore_best,
    }


def configure_reduce_lr_on_plateau() -> Dict:
    st.header("Configure ReduceLROnPlateau")
    use_scheduler = st.checkbox("Use ReduceLROnPlateau", value=True)
    if not use_scheduler:
        return {"use": False}
    monitor = st.selectbox("Metric to monitor", ["val_loss", "val_accuracy", "loss", "accuracy"], index=0, key="lr_monitor")
    factor = float(st.number_input("Factor", min_value=0.0, max_value=1.0, value=0.5))
    patience = int(st.number_input("Patience", min_value=0, value=2, key="lr_patience"))
    min_delta = float(st.number_input("Threshold / min_delta", value=1e-4, format="%.0e"))
    cooldown = int(st.number_input("Cooldown", min_value=0, value=0))
    min_lr = float(st.number_input("Minimum learning rate", value=1e-4, format="%.0e"))
    mode = st.selectbox("Mode", ["auto", "min", "max"], index=0, key="lr_mode")
    return {
        "use": True,
        "monitor": monitor,
        "factor": factor,
        "patience": patience,
        "threshold": min_delta,
        "cooldown": cooldown,
        "min_lr": min_lr,
        "mode": mode,
    }


def optimizer_selection() -> Dict:
    st.header("Select Optimizer and Set Hyperparameters")
    optimizer_name = st.radio(
        "Choose an optimizer:",
        ("SGD", "Adam", "RMSprop", "Adagrad", "Adadelta", "Adamax", "Nadam"),
        index=1,
    )
    st.subheader(f"{optimizer_name} Hyperparameters")
    params = {"name": optimizer_name}

    if optimizer_name == "SGD":
        params["lr"] = float(st.number_input("Learning rate", value=0.01, key="SGD_lr"))
        params["momentum"] = float(st.number_input("Momentum", value=0.0, key="SGD_momentum"))
        params["nesterov"] = bool(st.checkbox("Use Nesterov", value=False, key="SGD_nesterov"))
    elif optimizer_name == "Adam":
        params["lr"] = float(st.number_input("Learning rate", value=0.001, key="Adam_lr"))
        params["betas"] = (
            float(st.number_input("Beta 1", value=0.9, key="Adam_beta1")),
            float(st.number_input("Beta 2", value=0.999, key="Adam_beta2")),
        )
        params["eps"] = float(st.number_input("Epsilon", value=1e-7, format="%.0e", key="Adam_eps"))
    elif optimizer_name == "RMSprop":
        params["lr"] = float(st.number_input("Learning rate", value=0.001, key="RMS_lr"))
        params["alpha"] = float(st.number_input("Rho / alpha", value=0.9, key="RMS_alpha"))
        params["momentum"] = float(st.number_input("Momentum", value=0.0, key="RMS_momentum"))
        params["eps"] = float(st.number_input("Epsilon", value=1e-7, format="%.0e", key="RMS_eps"))
        params["centered"] = bool(st.checkbox("Centered", value=False, key="RMS_centered"))
    elif optimizer_name == "Adagrad":
        params["lr"] = float(st.number_input("Learning rate", value=0.01, key="Adagrad_lr"))
        params["lr_decay"] = float(st.number_input("LR decay", value=0.0, key="Adagrad_decay"))
        params["eps"] = float(st.number_input("Epsilon", value=1e-7, format="%.0e", key="Adagrad_eps"))
    elif optimizer_name == "Adadelta":
        params["lr"] = float(st.number_input("Learning rate", value=1.0, key="Adadelta_lr"))
        params["rho"] = float(st.number_input("Rho", value=0.95, key="Adadelta_rho"))
        params["eps"] = float(st.number_input("Epsilon", value=1e-7, format="%.0e", key="Adadelta_eps"))
    elif optimizer_name == "Adamax":
        params["lr"] = float(st.number_input("Learning rate", value=0.002, key="Adamax_lr"))
        params["betas"] = (
            float(st.number_input("Beta 1", value=0.9, key="Adamax_beta1")),
            float(st.number_input("Beta 2", value=0.999, key="Adamax_beta2")),
        )
        params["eps"] = float(st.number_input("Epsilon", value=1e-7, format="%.0e", key="Adamax_eps"))
    elif optimizer_name == "Nadam":
        params["lr"] = float(st.number_input("Learning rate", value=0.002, key="Nadam_lr"))
        params["betas"] = (
            float(st.number_input("Beta 1", value=0.9, key="Nadam_beta1")),
            float(st.number_input("Beta 2", value=0.999, key="Nadam_beta2")),
        )
        params["eps"] = float(st.number_input("Epsilon", value=1e-7, format="%.0e", key="Nadam_eps"))
    return params


def create_optimizer(model: nn.Module, optimizer_config: Dict, regularizer: Dict):
    # L2 in PyTorch is usually optimizer weight_decay.
    weight_decay = regularizer.get("l2", 0.0) if regularizer.get("type") in ["L2", "L1_L2"] else 0.0
    name = optimizer_config["name"]

    if name == "SGD":
        return torch.optim.SGD(
            model.parameters(),
            lr=optimizer_config["lr"],
            momentum=optimizer_config.get("momentum", 0.0),
            nesterov=optimizer_config.get("nesterov", False),
            weight_decay=weight_decay,
        )
    if name == "Adam":
        return torch.optim.Adam(model.parameters(), lr=optimizer_config["lr"], betas=optimizer_config["betas"], eps=optimizer_config["eps"], weight_decay=weight_decay)
    if name == "RMSprop":
        return torch.optim.RMSprop(model.parameters(), lr=optimizer_config["lr"], alpha=optimizer_config["alpha"], momentum=optimizer_config["momentum"], eps=optimizer_config["eps"], centered=optimizer_config["centered"], weight_decay=weight_decay)
    if name == "Adagrad":
        return torch.optim.Adagrad(model.parameters(), lr=optimizer_config["lr"], lr_decay=optimizer_config["lr_decay"], eps=optimizer_config["eps"], weight_decay=weight_decay)
    if name == "Adadelta":
        return torch.optim.Adadelta(model.parameters(), lr=optimizer_config["lr"], rho=optimizer_config["rho"], eps=optimizer_config["eps"], weight_decay=weight_decay)
    if name == "Adamax":
        return torch.optim.Adamax(model.parameters(), lr=optimizer_config["lr"], betas=optimizer_config["betas"], eps=optimizer_config["eps"], weight_decay=weight_decay)
    if name == "Nadam":
        return torch.optim.NAdam(model.parameters(), lr=optimizer_config["lr"], betas=optimizer_config["betas"], eps=optimizer_config["eps"], weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def l1_penalty(model: nn.Module) -> torch.Tensor:
    penalty = torch.tensor(0.0, device=next(model.parameters()).device)
    for p in model.parameters():
        penalty = penalty + p.abs().sum()
    return penalty


class EarlyStoppingTorch:
    def __init__(self, config: Dict):
        self.enabled = config.get("use", False)
        self.monitor = config.get("monitor", "val_loss")
        self.patience = int(config.get("patience", 6))
        self.min_delta = float(config.get("min_delta", 0.0))
        self.restore_best = bool(config.get("restore_best", False))
        mode = config.get("mode", "auto")
        if mode == "auto":
            mode = "max" if "accuracy" in self.monitor else "min"
        self.mode = mode
        self.best_score = None
        self.wait = 0
        self.best_state = None

    def improved(self, current: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "min":
            return current < self.best_score - self.min_delta
        return current > self.best_score + self.min_delta

    def step(self, model: nn.Module, metrics: Dict[str, float]) -> bool:
        if not self.enabled:
            return False
        current = metrics[self.monitor]
        if self.improved(current):
            self.best_score = current
            self.wait = 0
            if self.restore_best:
                self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.wait += 1
        if self.wait > self.patience:
            if self.restore_best and self.best_state is not None:
                model.load_state_dict(self.best_state)
            return True
        return False


def create_scheduler(optimizer, scheduler_config: Dict):
    if not scheduler_config.get("use", False):
        return None, None
    monitor = scheduler_config.get("monitor", "val_loss")
    mode = scheduler_config.get("mode", "auto")
    if mode == "auto":
        mode = "max" if "accuracy" in monitor else "min"
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=mode,
        factor=scheduler_config["factor"],
        patience=scheduler_config["patience"],
        threshold=scheduler_config["threshold"],
        cooldown=scheduler_config["cooldown"],
        min_lr=scheduler_config["min_lr"],
    )
    return scheduler, monitor


# =========================================================
# Data and preprocessing
# =========================================================

def df_split_data(df: pd.DataFrame, test_size_input: float, random_state_: int):
    label_col = df.columns[-1]
    feature_df = df.drop(columns=[label_col]).copy()
    label_series = df[label_col].copy()

    # Force numeric features. Non-numeric values become NaN and should be handled before training.
    feature_df = feature_df.apply(pd.to_numeric, errors="coerce")
    valid_mask = ~feature_df.isna().any(axis=1)
    feature_df = feature_df.loc[valid_mask]
    label_series = label_series.loc[valid_mask]

    encoder = LabelEncoder()
    y = encoder.fit_transform(label_series.astype(str))
    X = feature_df.values.astype(np.float32)

    stratify = y if len(np.unique(y)) > 1 and min(Counter(y).values()) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size_input,
        random_state=random_state_,
        shuffle=True,
        stratify=stratify,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    return X_train, X_test, y_train.astype(np.int64), y_test.astype(np.int64), encoder, scaler


def make_loaders(X_train, X_test, y_train, y_test, batch_size: int):
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def drop_nans_function(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("📌 Drop Missing Values")
    drop_nans = st.checkbox("Drop rows with missing values", value=False, key="drop_nans")
    return df.dropna() if drop_nans else df


def select_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    all_columns = df.columns.tolist()
    label_column = all_columns[-1]
    st.subheader("📌 Column Selection")
    st.markdown(f"🟨 **Label column (target):** `{label_column}`")
    feature_columns = [col for col in all_columns if col != label_column]
    default_feature = feature_columns
    select_all_features = st.checkbox("Select all features", value=True, key="select_all_features")
    if select_all_features:
        selected_columns = st.multiselect("Features:", options=feature_columns, default=feature_columns)
    else:
        selected_columns = st.multiselect("Features:", options=feature_columns, default=default_feature[:1])
    if not selected_columns:
        st.warning("Please select at least one feature column.")
        return df[[feature_columns[0], label_column]]
    return df[selected_columns + [label_column]]


def plot_class_distribution(y, title="Class Distribution"):
    counts = Counter(y)
    labels = list(counts.keys())
    values = list(counts.values())
    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    return fig


def balance_data(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("📌 Data Balancing")
    balance_check = st.checkbox("Enable Data Balancing", value=False, key="balance_data_check")
    if not balance_check:
        return df

    if not IMBLEARN_AVAILABLE:
        st.error("imbalanced-learn is not installed. Install it with: pip install imbalanced-learn")
        return df

    strategy = st.radio("Select balancing strategy", ["undersample", "oversample", "smote"], index=2)
    feature_cols = df.columns[:-1].tolist()
    label_col = df.columns[-1]
    X = df[feature_cols]
    y = df[label_col]

    if strategy == "undersample":
        sampler = RandomUnderSampler(random_state=42)
    elif strategy == "oversample":
        sampler = RandomOverSampler(random_state=42)
    else:
        sampler = SMOTE(random_state=42)

    try:
        X_res, y_res = sampler.fit_resample(X, y)
    except Exception as e:
        st.error(f"Balancing failed: {e}")
        return df

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("🔍 **Original Class Distribution**")
        st.pyplot(plot_class_distribution(y, "Before Balancing"))
    with col2:
        st.markdown(f"✅ **Resampled Distribution using `{strategy}`**")
        st.pyplot(plot_class_distribution(y_res, "After Balancing"))

    df_resampled = pd.DataFrame(X_res, columns=feature_cols)
    df_resampled[label_col] = y_res
    return df_resampled


def preprocessing_options(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("⚙️ Preprocessing Options")
    df = drop_nans_function(df)
    df = select_feature_columns(df)
    df = balance_data(df)
    st.subheader("Preprocessed Data Preview")
    st.dataframe(df.head())
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Preprocessed CSV", csv, "preprocessed_data.csv", "text/csv")
    return df


# =========================================================
# Training and evaluation
# =========================================================

def train_one_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict,
    device: torch.device,
) -> Dict[str, List[float]]:
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, config["optimizer"], config["regularizer"])
    scheduler, scheduler_monitor = create_scheduler(optimizer, config["lr_scheduler"])
    early_stopper = EarlyStoppingTorch(config["early_stopping"])

    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": [], "lr": []}
    l1_lambda = config["regularizer"].get("l1", 0.0) if config["regularizer"].get("type") in ["L1", "L1_L2"] else 0.0

    progress = st.progress(0)
    status = st.empty()

    for epoch in range(config["epoch_fix"]):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            if l1_lambda > 0:
                loss = loss + l1_lambda * l1_penalty(model)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * xb.size(0)
            train_correct += (logits.argmax(dim=1) == yb).sum().item()
            train_total += xb.size(0)

        train_loss = train_loss_sum / max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)

        metrics = {
            "loss": train_loss,
            "accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        }
        for k, v in metrics.items():
            history[k].append(float(v))
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        if scheduler is not None:
            scheduler.step(metrics[scheduler_monitor])

        progress.progress((epoch + 1) / config["epoch_fix"])
        status.write(
            f"Epoch {epoch + 1}/{config['epoch_fix']} | "
            f"loss={train_loss:.4f}, acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if early_stopper.step(model, metrics):
            st.info(f"Early stopping at epoch {epoch + 1}.")
            break

    return history


def evaluate_model(model: nn.Module, data_loader: DataLoader, criterion, device: torch.device) -> Tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in data_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss_sum += loss.item() * xb.size(0)
            correct += (logits.argmax(dim=1) == yb).sum().item()
            total += xb.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


# =========================================================
# Tuning
# =========================================================

def run_training_for_tuning(df: pd.DataFrame, config: Dict, override: Dict) -> Dict[str, List[float]]:
    cfg = {**config, **override}
    set_random_seed(cfg["random_seed"])
    X_train, X_test, y_train, y_test, _, _ = df_split_data(df, cfg["test_size"], cfg["random_seed"])
    train_loader, val_loader = make_loaders(X_train, X_test, y_train, y_test, cfg["batch_size"])
    model = build_model(input_dim=X_train.shape[1], num_classes=len(np.unique(y_train)), config=cfg)

    # Silent tuning: no Streamlit progress inside repeated loops.
    return train_one_model_silent(model, train_loader, val_loader, cfg, get_device())


def train_one_model_silent(model, train_loader, val_loader, config, device) -> Dict[str, List[float]]:
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, config["optimizer"], config["regularizer"])
    scheduler, scheduler_monitor = create_scheduler(optimizer, config["lr_scheduler"])
    early_stopper = EarlyStoppingTorch(config["early_stopping"])
    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    l1_lambda = config["regularizer"].get("l1", 0.0) if config["regularizer"].get("type") in ["L1", "L1_L2"] else 0.0

    for _ in range(config["epoch_fix"]):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            if l1_lambda > 0:
                loss = loss + l1_lambda * l1_penalty(model)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * xb.size(0)
            train_correct += (logits.argmax(dim=1) == yb).sum().item()
            train_total += xb.size(0)

        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        metrics = {
            "loss": train_loss_sum / max(train_total, 1),
            "accuracy": train_correct / max(train_total, 1),
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        }
        for k, v in metrics.items():
            history[k].append(float(v))
        if scheduler is not None:
            scheduler.step(metrics[scheduler_monitor])
        if early_stopper.step(model, metrics):
            break
    return history


def plot_history_dict(histories: Dict, metric: str, title: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, hist in histories.items():
        y = hist[metric]
        x = range(1, len(y) + 1)
        ax.plot(x, y, label=str(label))
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="best")
    return fig


def run_tuning(df: pd.DataFrame, config: Dict):
    st.subheader("🧪 PyTorch Hyperparameter Tuning")
    activations, batch_sizes, _, _, test_sizes = get_model_tuning_inputs()

    with st.spinner("Running activation tuning..."):
        activation_histories = {}
        for act in activations:
            activation_histories[act] = run_training_for_tuning(df, config, {"activation_input_hidden": act})

    with st.spinner("Running batch-size tuning..."):
        batch_histories = {}
        for bs in batch_sizes:
            batch_histories[bs] = run_training_for_tuning(df, config, {"batch_size": bs})

    with st.spinner("Running validation-split tuning..."):
        test_size_histories = {}
        for ts in test_sizes:
            test_size_histories[ts] = run_training_for_tuning(df, config, {"test_size": ts})

    figs = []
    figs.append(plot_history_dict({"train": {"loss": activation_histories[config["activation_input_hidden"]]["loss"]}, "val": {"loss": activation_histories[config["activation_input_hidden"]]["val_loss"]}}, "loss", "Overfitting Check"))
    figs.append(plot_history_dict(activation_histories, "val_loss", "Activation Tuning - Validation Loss"))
    figs.append(plot_history_dict(activation_histories, "val_accuracy", "Activation Tuning - Validation Accuracy"))
    figs.append(plot_history_dict(batch_histories, "val_accuracy", "Batch Size Tuning - Validation Accuracy"))
    figs.append(plot_history_dict(test_size_histories, "val_loss", "Validation Split Tuning - Validation Loss"))
    figs.append(plot_history_dict(test_size_histories, "val_accuracy", "Validation Split Tuning - Validation Accuracy"))

    cols = st.columns(2)
    for i, fig in enumerate(figs):
        with cols[i % 2]:
            st.pyplot(fig)

    # Save all figures into one PNG-like report is not simple; provide individual downloads.
    for i, fig in enumerate(figs, start=1):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        st.download_button(f"📥 Download tuning plot {i}", data=buf.getvalue(), file_name=f"tuning_plot_{i}.png", mime="image/png")


# =========================================================
# Saving and summary
# =========================================================

def generate_model_filename(config: Dict):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    parts = [
        f"ep{config['epoch_fix']}",
        f"bs{config['batch_size']}",
        f"ts{int(config['test_size'] * 100)}",
        f"hid{config['hidden_layer_number']}",
        f"units{config['dense_input']}",
        f"act_{config['activation_input_hidden']}",
        f"bn{'on' if config['batch_normalization_check'] else 'off'}",
        f"stair{'on' if config['staircase'] else 'off'}",
        f"seed{config['random_seed']}",
        f"drop{'on' if config['dropout_rate'] else 'off'}",
        f"reg{config['regularizer']['type']}",
        f"opt{config['optimizer']['name']}",
    ]
    stem = re.sub(r"[^\w\-_\.]", "", "_".join(parts))
    return (
        f"{stem}_{timestamp}.pt",
        f"{stem}_summary_{timestamp}.txt",
        f"{stem}_training_summary_{timestamp}.txt",
    )


def model_summary_text(model: nn.Module, input_dim: int) -> str:
    lines = [str(model), ""]
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lines.append(f"Input dimension: {input_dim}")
    lines.append(f"Total parameters: {total_params:,}")
    lines.append(f"Trainable parameters: {trainable_params:,}")
    return "\n".join(lines)


def format_training_summary(history: Optional[Dict[str, List[float]]]) -> str:
    if history is None:
        return "No training history available."
    epochs = len(history["loss"])
    lines = [f"📊 Training Summary (Epochs: {epochs})"]
    lines.append(f"- Final Training Loss: {history['loss'][-1]:.4f}")
    lines.append(f"- Final Validation Loss: {history['val_loss'][-1]:.4f}")
    lines.append(f"- Final Training Accuracy: {history['accuracy'][-1] * 100:.2f}%")
    lines.append(f"- Final Validation Accuracy: {history['val_accuracy'][-1] * 100:.2f}%")
    if history["val_loss"][-1] > history["loss"][-1] * 1.1:
        lines.append("⚠️ Possible overfitting: validation loss is notably higher than training loss.")
    return "\n".join(lines)


def plot_training_history(history: Dict[str, List[float]]):
    fig_loss, ax_loss = plt.subplots(figsize=(8, 5))
    ax_loss.plot(range(1, len(history["loss"]) + 1), history["loss"], label="train")
    ax_loss.plot(range(1, len(history["val_loss"]) + 1), history["val_loss"], label="validation")
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("CrossEntropyLoss")
    ax_loss.grid(True, linestyle="--", alpha=0.6)
    ax_loss.legend()

    fig_acc, ax_acc = plt.subplots(figsize=(8, 5))
    ax_acc.plot(range(1, len(history["accuracy"]) + 1), history["accuracy"], label="train")
    ax_acc.plot(range(1, len(history["val_accuracy"]) + 1), history["val_accuracy"], label="validation")
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.grid(True, linestyle="--", alpha=0.6)
    ax_acc.legend()
    return fig_loss, fig_acc


# =========================================================
# Streamlit configuration UI
# =========================================================

def get_model_training_inputs() -> Dict:
    st.header("Model Training Configuration")
    activations, batch_size_list, dense_input_list, seed_options, test_sizes = get_model_tuning_inputs()

    random_seed = st.radio("Select a random seed:", options=seed_options, index=seed_options.index(42))
    random_seed = set_random_seed(random_seed)
    test_size = st.radio("Validation split", options=test_sizes, index=test_sizes.index(0.25))
    batch_size = st.radio("Batch size", options=batch_size_list, index=batch_size_list.index(32))
    epoch_fix = int(st.number_input("Number of epochs", min_value=1, max_value=500, value=30, step=5))

    st.subheader("Model Architecture")
    dense_input = st.selectbox("Input hidden layer size", options=dense_input_list, index=dense_input_list.index(128))
    hidden_layer_number = int(st.slider("Number of additional hidden layers", min_value=0, max_value=10, value=3))
    staircase = st.checkbox("Apply staircase reduction in layer sizes", value=False)
    activation_input_hidden = st.selectbox("Hidden activation", options=activations, index=activations.index("tanh"))

    # Output activation is intentionally not used for training because CrossEntropyLoss expects logits.
    st.info("PyTorch CrossEntropyLoss expects raw logits, so the output layer has no Softmax during training. Softmax is applied only when you need probabilities for inference.")
    batch_normalization_check = st.checkbox("Enable Batch Normalization", value=False)

    regularizer = configure_regularizer()
    dropout_rate = configure_dropout_layer()
    early_stopping = configure_early_stopping()
    lr_scheduler = configure_reduce_lr_on_plateau()
    optimizer = optimizer_selection()

    return {
        "epoch_fix": epoch_fix,
        "test_size": test_size,
        "batch_size": batch_size,
        "dense_input": dense_input,
        "hidden_layer_number": hidden_layer_number,
        "activation_input_hidden": activation_input_hidden,
        "random_seed": random_seed,
        "batch_normalization_check": batch_normalization_check,
        "staircase": staircase,
        "regularizer": regularizer,
        "dropout_rate": dropout_rate,
        "early_stopping": early_stopping,
        "lr_scheduler": lr_scheduler,
        "optimizer": optimizer,
    }


def operation_mode_function():
    st.title("Auto-Tuning and Training - PyTorch")
    operation_mode = st.radio("Select Operation Mode:", ("Tune Only", "Train Only", "Tune and Train"), index=1)
    st.write(f"Selected Mode: **{operation_mode}**")
    return operation_mode


def run_final_training(df: pd.DataFrame, config: Dict):
    set_random_seed(config["random_seed"])
    X_train, X_test, y_train, y_test, encoder, scaler = df_split_data(df, config["test_size"], config["random_seed"])
    train_loader, val_loader = make_loaders(X_train, X_test, y_train, y_test, config["batch_size"])
    model = build_model(input_dim=X_train.shape[1], num_classes=len(encoder.classes_), config=config)
    history = train_one_model(model, train_loader, val_loader, config, get_device())
    return model, history, encoder, scaler, X_train.shape[1]


def process(df: pd.DataFrame):
    config = get_model_training_inputs()
    operation_mode = operation_mode_function()

    if st.button("🚀 Start"):
        if operation_mode in ["Tune Only", "Tune and Train"]:
            run_tuning(df, config)

        if operation_mode in ["Train Only", "Tune and Train"]:
            st.subheader("🏋️ Final Training")
            model, history, encoder, scaler, input_dim = run_final_training(df, config)

            filename_model, filename_summary, filename_training = generate_model_filename(config)
            summary = model_summary_text(model, input_dim)
            training_summary = format_training_summary(history)

            st.subheader("📄 Model Summary")
            st.text(summary)
            st.subheader("📈 Training Summary")
            st.text(training_summary)

            fig_loss, fig_acc = plot_training_history(history)
            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(fig_loss)
            with col2:
                st.pyplot(fig_acc)

            checkpoint = {
                "model_state_dict": model.cpu().state_dict(),
                "config": config,
                "classes": encoder.classes_.tolist(),
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
                "input_dim": input_dim,
            }
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
                torch.save(checkpoint, tmp.name)
                with open(tmp.name, "rb") as f:
                    st.download_button("📥 Download PyTorch Model Checkpoint", f, file_name=filename_model, mime="application/octet-stream")

            st.download_button("📥 Download Model Summary", summary, file_name=filename_summary, mime="text/plain")
            st.download_button("📥 Download Training Summary", training_summary, file_name=filename_training, mime="text/plain")


# =========================================================
# App pages
# =========================================================

def introduction_page():
    st.markdown("""
# Welcome to the PyTorch Model Configuration Portal

This app lets you upload a CSV file, preprocess it, configure a dense PyTorch neural network, tune hyperparameters, train the final model, and download the trained `.pt` checkpoint.

**Expected dataset format:**
- CSV file.
- Last column is the class label.
- All previous selected columns are numeric features.

**Main difference from TensorFlow/Keras:**
- Labels are integer encoded, not one-hot encoded.
- The final model returns raw logits.
- `nn.CrossEntropyLoss()` internally handles the softmax calculation.
""")

    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        max_size_mb = st.sidebar.number_input("Max file size MB", min_value=0.1, max_value=100.0, value=5.0)
        if file_size_mb > max_size_mb:
            st.error(f"🚫 File too large. Limit is {max_size_mb:.2f} MB. Your file is {file_size_mb:.2f} MB.")
        else:
            st.success("✅ File uploaded successfully!")
            st.session_state.df = pd.read_csv(uploaded_file)
            st.subheader("📄 Preview of Uploaded Data")
            st.write(f"**File size:** {file_size_mb:.2f} MB")
            st.dataframe(st.session_state.df.head())


def main_app():
    st.set_page_config(page_title="PyTorch CSV Trainer", layout="wide")
    st.sidebar.markdown("### 📂 Navigate Pages")
    page = st.sidebar.radio(
        "Page",
        ["📄 Introduction & Data Preview", "⚙️ Preprocessing & Analysis", "🧪 Configure & Train Model"],
    )

    st.sidebar.markdown("---")
    st.sidebar.write(f"Torch version: `{torch.__version__}`")
    st.sidebar.write(f"Device: `{get_device()}`")

    if page == "📄 Introduction & Data Preview":
        introduction_page()
    elif page == "⚙️ Preprocessing & Analysis":
        if "df" not in st.session_state:
            st.info("Please upload a CSV file first.")
        else:
            st.session_state.df_preprocessed = preprocessing_options(st.session_state.df)
    elif page == "🧪 Configure & Train Model":
        if "df" not in st.session_state:
            st.info("Please upload a CSV file first.")
        else:
            df_to_use = st.session_state.get("df_preprocessed", st.session_state.df)
            process(df_to_use)


if __name__ == "__main__":
    main_app()
