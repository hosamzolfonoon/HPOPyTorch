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
from torch.utils.data import DataLoader, TensorDataset, Dataset

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


def safe_download_button(label, data, file_name, mime, key=None):
    """Download without rerunning the whole Streamlit app.

    Streamlit reruns the app by default when st.download_button is clicked.
    on_click="ignore" prevents that in recent Streamlit versions.
    """
    return st.download_button(
        label=label,
        data=data,
        file_name=file_name,
        mime=mime,
        key=key,
        on_click="ignore",
    )


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
    """
    Original simple split:
        train = 1 - test_size_input
        validation = test_size_input

    This is useful for quick tuning, but it does NOT create a separate test set.
    """
    label_col = df.columns[-1]
    feature_df = df.drop(columns=[label_col]).copy()
    label_series = df[label_col].copy()

    # Force numeric features. Non-numeric values become NaN and are removed here.
    feature_df = feature_df.apply(pd.to_numeric, errors="coerce")
    valid_mask = ~feature_df.isna().any(axis=1)
    feature_df = feature_df.loc[valid_mask]
    label_series = label_series.loc[valid_mask]

    encoder = LabelEncoder()
    y = encoder.fit_transform(label_series.astype(str))
    X = feature_df.values.astype(np.float32)

    stratify = y if len(np.unique(y)) > 1 and min(Counter(y).values()) >= 2 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size_input,
        random_state=random_state_,
        shuffle=True,
        stratify=stratify,
    )

    # IMPORTANT: fit scaler on training only, then transform validation.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)

    return X_train, X_val, y_train.astype(np.int64), y_val.astype(np.int64), encoder, scaler


def df_split_data_three_way(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_state_: int,
):
    """
    Three-way split:
        train / validation / test

    Example:
        train_ratio=0.70, val_ratio=0.15, test_ratio=0.15

    Validation is used during training and early stopping.
    Test is used only once after final training.
    """
    ratio_sum = train_ratio + val_ratio + test_ratio
    if not np.isclose(ratio_sum, 1.0):
        raise ValueError(
            f"Split ratios must sum to 1.0, but got {ratio_sum:.3f}"
        )

    label_col = df.columns[-1]
    feature_df = df.drop(columns=[label_col]).copy()
    label_series = df[label_col].copy()

    # Force numeric features. Non-numeric values become NaN and are removed here.
    feature_df = feature_df.apply(pd.to_numeric, errors="coerce")
    valid_mask = ~feature_df.isna().any(axis=1)
    feature_df = feature_df.loc[valid_mask]
    label_series = label_series.loc[valid_mask]

    encoder = LabelEncoder()
    y = encoder.fit_transform(label_series.astype(str))
    X = feature_df.values.astype(np.float32)

    # First split: train vs temporary validation+test.
    temp_ratio = val_ratio + test_ratio
    stratify_main = y if len(np.unique(y)) > 1 and min(Counter(y).values()) >= 3 else None

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=temp_ratio,
        random_state=random_state_,
        shuffle=True,
        stratify=stratify_main,
    )

    # Second split: temporary into validation and test.
    # Example 70/15/15: temp is 30%; test_size=0.5 gives 15% val and 15% test.
    test_fraction_inside_temp = test_ratio / temp_ratio
    stratify_temp = y_temp if len(np.unique(y_temp)) > 1 and min(Counter(y_temp).values()) >= 2 else None

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=test_fraction_inside_temp,
        random_state=random_state_,
        shuffle=True,
        stratify=stratify_temp,
    )

    # IMPORTANT: fit scaler on training only, then transform validation/test.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    return (
        X_train,
        X_val,
        X_test,
        y_train.astype(np.int64),
        y_val.astype(np.int64),
        y_test.astype(np.int64),
        encoder,
        scaler,
    )


def make_loaders(X_train, X_val, y_train, y_val, batch_size: int):
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def make_loaders_three_way(X_train, X_val, X_test, y_train, y_val, y_test, batch_size: int):
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


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
    safe_download_button("📥 Download Preprocessed CSV", csv, "preprocessed_data.csv", "text/csv", key="download_preprocessed_csv")
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

    # During tuning we use train + validation only.
    # If the user selected a three-way split, the test set is created but ignored here.
    # The test set is reserved for final evaluation only.
    if cfg.get("split_strategy") == "Train/Validation/Test":
        (
            X_train,
            X_val,
            _X_test,
            y_train,
            y_val,
            _y_test,
            encoder,
            _scaler,
        ) = df_split_data_three_way(
            df,
            cfg["train_ratio"],
            cfg["val_ratio"],
            cfg["test_ratio"],
            cfg["random_seed"],
        )
    else:
        X_train, X_val, y_train, y_val, encoder, _scaler = df_split_data(
            df,
            cfg["test_size"],
            cfg["random_seed"],
        )

    train_loader, val_loader = make_loaders(X_train, X_val, y_train, y_val, cfg["batch_size"])
    model = build_model(input_dim=X_train.shape[1], num_classes=len(encoder.classes_), config=cfg)

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

    if config.get("split_strategy") == "Simple Train/Validation":
        with st.spinner("Running validation-split tuning..."):
            test_size_histories = {}
            for ts in test_sizes:
                test_size_histories[ts] = run_training_for_tuning(df, config, {"test_size": ts})
    else:
        test_size_histories = None
        st.info("Validation-split tuning is skipped because Train/Validation/Test mode uses fixed ratios such as 70/15/15.")

    figs = []
    figs.append(plot_history_dict({"train": {"loss": activation_histories[config["activation_input_hidden"]]["loss"]}, "val": {"loss": activation_histories[config["activation_input_hidden"]]["val_loss"]}}, "loss", "Overfitting Check"))
    figs.append(plot_history_dict(activation_histories, "val_loss", "Activation Tuning - Validation Loss"))
    figs.append(plot_history_dict(activation_histories, "val_accuracy", "Activation Tuning - Validation Accuracy"))
    figs.append(plot_history_dict(batch_histories, "val_accuracy", "Batch Size Tuning - Validation Accuracy"))
    if test_size_histories is not None:
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
        safe_download_button(f"📥 Download tuning plot {i}", data=buf.getvalue(), file_name=f"tuning_plot_{i}.png", mime="image/png", key=f"download_tuning_plot_{i}")


# =========================================================
# Saving and summary
# =========================================================

def generate_model_filename(config: Dict):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    parts = [
        f"ep{config['epoch_fix']}",
        f"bs{config['batch_size']}",
        f"split{config.get('split_label', 'simple')}",
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

    st.subheader("Data Split")
    split_strategy = st.radio(
        "Select data split strategy",
        ["Train/Validation/Test", "Simple Train/Validation"],
        index=0,
        help=(
            "Use Train/Validation/Test for final experiments. "
            "Use Simple Train/Validation only for quick tuning."
        ),
    )

    # Defaults. These keys always exist so the rest of the code is safe.
    test_size = 0.15
    train_ratio = 0.70
    val_ratio = 0.15
    test_ratio = 0.15
    split_label = "70_15_15"

    if split_strategy == "Train/Validation/Test":
        split_preset = st.selectbox(
            "Train / Validation / Test ratio",
            [
                "70 / 15 / 15",
                "80 / 10 / 10",
                "60 / 20 / 20",
                "Custom",
            ],
            index=0,
        )

        if split_preset == "70 / 15 / 15":
            train_ratio, val_ratio, test_ratio = 0.70, 0.15, 0.15
        elif split_preset == "80 / 10 / 10":
            train_ratio, val_ratio, test_ratio = 0.80, 0.10, 0.10
        elif split_preset == "60 / 20 / 20":
            train_ratio, val_ratio, test_ratio = 0.60, 0.20, 0.20
        else:
            train_percent = st.slider("Training percentage", min_value=50, max_value=90, value=70, step=5)
            val_percent = st.slider("Validation percentage", min_value=5, max_value=40, value=15, step=5)
            test_percent = 100 - train_percent - val_percent

            if test_percent <= 0:
                st.error("Invalid split: test percentage must be greater than 0. Reduce train or validation percentage.")
                st.stop()

            train_ratio = train_percent / 100
            val_ratio = val_percent / 100
            test_ratio = test_percent / 100

        split_label = f"{int(train_ratio * 100)}_{int(val_ratio * 100)}_{int(test_ratio * 100)}"

        st.success(
            f"Selected split: **{int(train_ratio * 100)}% train / "
            f"{int(val_ratio * 100)}% validation / "
            f"{int(test_ratio * 100)}% test**"
        )
        st.caption(
            "Validation is used during training and early stopping. "
            "Test is used only once after final training."
        )

    else:
        test_size = st.radio(
            "Validation split",
            options=test_sizes,
            index=test_sizes.index(0.25),
        )
        split_label = f"simple_val{int(test_size * 100)}"
        st.info(
            f"Selected simple split: **{int((1 - test_size) * 100)}% train / "
            f"{int(test_size * 100)}% validation**. No separate test set is created."
        )

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
        "split_strategy": split_strategy,
        "split_label": split_label,
        "test_size": test_size,          # Used only for Simple Train/Validation mode
        "train_ratio": train_ratio,      # Used only for Train/Validation/Test mode
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
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
    device = get_device()

    if config.get("split_strategy") == "Train/Validation/Test":
        (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            encoder,
            scaler,
        ) = df_split_data_three_way(
            df,
            config["train_ratio"],
            config["val_ratio"],
            config["test_ratio"],
            config["random_seed"],
        )

        train_loader, val_loader, test_loader = make_loaders_three_way(
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            config["batch_size"],
        )

        model = build_model(input_dim=X_train.shape[1], num_classes=len(encoder.classes_), config=config)
        history = train_one_model(model, train_loader, val_loader, config, device)

        # Final honest test evaluation: test set is not used during training.
        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)

        test_metrics = {
            "has_test": True,
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
            "n_train": int(len(y_train)),
            "n_val": int(len(y_val)),
            "n_test": int(len(y_test)),
        }

        return model, history, encoder, scaler, X_train.shape[1], test_metrics

    # Simple Train/Validation mode
    X_train, X_val, y_train, y_val, encoder, scaler = df_split_data(
        df,
        config["test_size"],
        config["random_seed"],
    )
    train_loader, val_loader = make_loaders(X_train, X_val, y_train, y_val, config["batch_size"])
    model = build_model(input_dim=X_train.shape[1], num_classes=len(encoder.classes_), config=config)
    history = train_one_model(model, train_loader, val_loader, config, device)

    test_metrics = {
        "has_test": False,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": 0,
    }

    return model, history, encoder, scaler, X_train.shape[1], test_metrics


def process(df: pd.DataFrame):
    config = get_model_training_inputs()
    operation_mode = operation_mode_function()

    if st.button("🚀 Start"):
        if operation_mode in ["Tune Only", "Tune and Train"]:
            run_tuning(df, config)

        if operation_mode in ["Train Only", "Tune and Train"]:
            st.subheader("🏋️ Final Training")
            model, history, encoder, scaler, input_dim, test_metrics = run_final_training(df, config)

            filename_model, filename_summary, filename_training = generate_model_filename(config)
            summary = model_summary_text(model, input_dim)
            training_summary = format_training_summary(history)

            training_summary += "\n\n📦 Data Split"
            training_summary += f"\n- Train samples: {test_metrics['n_train']}"
            training_summary += f"\n- Validation samples: {test_metrics['n_val']}"
            training_summary += f"\n- Test samples: {test_metrics['n_test']}"

            if test_metrics.get("has_test"):
                training_summary += "\n\n🧪 Final Test Results"
                training_summary += f"\n- Final Test Loss: {test_metrics['test_loss']:.4f}"
                training_summary += f"\n- Final Test Accuracy: {test_metrics['test_accuracy'] * 100:.2f}%"

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
                "test_metrics": test_metrics,
            }
            models_dir = "trained_models"
            os.makedirs(models_dir, exist_ok=True)
            model_path = os.path.join(models_dir, filename_model)
            torch.save(checkpoint, model_path)
            st.success(f"✅ Saved model permanently to: {model_path}")
            with open(model_path, "rb") as f:
                safe_download_button(
                    "📥 Download PyTorch Model Checkpoint",
                    f.read(),
                    file_name=filename_model,
                    mime="application/octet-stream",
                    key="download_tabular_checkpoint",
                )

            safe_download_button("📥 Download Model Summary", summary, file_name=filename_summary, mime="text/plain", key="download_model_summary")
            safe_download_button("📥 Download Training Summary", training_summary, file_name=filename_training, mime="text/plain", key="download_training_summary")




# =========================================================
# CNN BELIEF UPDATER PAGE
# =========================================================
# This section is intentionally independent from the CSV dense-classifier
# workflow above. The existing app trains a tabular classifier from CSV.
# The CNN belief updater is different: it learns a map-to-map function
#     [B_prev, Z_t, C_t, A_t] -> M or B_t
# where:
#   B_prev = previous belief map
#   Z_t    = current noisy observation map
#   C_t    = footprint/visibility mask
#   A_t    = altitude channel
#   M      = binary ground-truth terrain map
# Therefore, this page expects .npz files with:
#   X shape = (N, 4, H, W)
#   Y shape = (N, 1, H, W)


class BeliefArrayDataset(Dataset):
    """
    PyTorch Dataset for pre-generated CNN belief-updater samples.

    Expected arrays:
        X: float32, shape (N, 4, H, W)
        Y: float32, shape (N, 1, H, W)

    Each sample means:
        input  X[n] = [B_prev, Z_t, C_t, A_t]
        target Y[n] = binary terrain M, or desired updated belief target.
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.float32)

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.Y[idx])


class CNNBeliefUpdater(nn.Module):
    """
    Lightweight fully-convolutional CNN belief updater.

    Input:
        x shape = (batch, 4, H, W)

    Output:
        logits shape = (batch, 1, H, W)

    Important:
        The model returns raw logits. During training we use
        BCEWithLogitsLoss, which internally applies sigmoid safely.
        During visualization/evaluation we apply torch.sigmoid(logits)
        to obtain belief probabilities in [0, 1].
    """

    def __init__(self, in_channels: int = 4, base_channels: int = 32, use_batch_norm: bool = True):
        super().__init__()

        def block(cin, cout):
            layers = [nn.Conv2d(cin, cout, kernel_size=3, padding=1)]
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(cout))
            layers.append(nn.ReLU(inplace=True))
            return layers

        self.net = nn.Sequential(
            *block(in_channels, base_channels),
            *block(base_channels, base_channels),
            *block(base_channels, base_channels * 2),
            *block(base_channels * 2, base_channels * 2),
            *block(base_channels * 2, base_channels),
            nn.Conv2d(base_channels, 1, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)


def load_belief_npz_from_upload(uploaded_file, name_for_messages: str):
    """
    Load a Streamlit-uploaded .npz file containing X and Y arrays.
    The .npz should have exactly the arrays generated earlier:
        X = (N, 4, H, W)
        Y = (N, 1, H, W)
    """
    if uploaded_file is None:
        return None, None

    try:
        data = np.load(io.BytesIO(uploaded_file.getvalue()))
    except Exception as e:
        st.error(f"Could not read {name_for_messages} .npz file: {e}")
        return None, None

    if "X" not in data or "Y" not in data:
        st.error(f"{name_for_messages} must contain arrays named `X` and `Y`.")
        return None, None

    X = data["X"]
    Y = data["Y"]

    if X.ndim != 4 or Y.ndim != 4:
        st.error(
            f"{name_for_messages} has invalid dimensions. Expected X=(N,4,H,W), Y=(N,1,H,W). "
            f"Got X={X.shape}, Y={Y.shape}."
        )
        return None, None

    if X.shape[0] != Y.shape[0]:
        st.error(f"{name_for_messages}: X and Y have different sample counts: {X.shape[0]} vs {Y.shape[0]}.")
        return None, None

    if X.shape[1] != 4:
        st.error(f"{name_for_messages}: X must have 4 channels [B_prev, Z_t, C_t, A_t]. Got {X.shape[1]} channels.")
        return None, None

    if Y.shape[1] != 1:
        st.error(f"{name_for_messages}: Y must have 1 output channel. Got {Y.shape[1]} channels.")
        return None, None

    if X.shape[2:] != Y.shape[2:]:
        st.error(f"{name_for_messages}: X and Y patch sizes differ. Got X={X.shape}, Y={Y.shape}.")
        return None, None

    return X.astype(np.float32), Y.astype(np.float32)


def summarize_belief_array(name: str, X: np.ndarray, Y: np.ndarray):
    """Show dataset shape and value range in Streamlit."""
    st.markdown(f"**{name} dataset**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Samples", f"{X.shape[0]:,}")
    c2.metric("Input shape", str(tuple(X.shape[1:])))
    c3.metric("Target shape", str(tuple(Y.shape[1:])))
    c4.metric("Memory", f"{(X.nbytes + Y.nbytes) / (1024**2):.1f} MB")
    st.caption(
        f"X range: [{float(np.nanmin(X)):.3f}, {float(np.nanmax(X)):.3f}] | "
        f"Y range: [{float(np.nanmin(Y)):.3f}, {float(np.nanmax(Y)):.3f}]"
    )


def make_belief_loaders_from_arrays(
    X_train, Y_train, X_val, Y_val, X_test=None, Y_test=None,
    batch_size: int = 4, num_workers: int = 0
):
    """Create DataLoaders for CNN belief updater arrays."""
    train_loader = DataLoader(
        BeliefArrayDataset(X_train, Y_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        BeliefArrayDataset(X_val, Y_val),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = None
    if X_test is not None and Y_test is not None:
        test_loader = DataLoader(
            BeliefArrayDataset(X_test, Y_test),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    return train_loader, val_loader, test_loader


def belief_metrics_from_logits(logits: torch.Tensor, y: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute map-level metrics for one batch.

    MSE uses probabilities.
    IoU/F1 use binary maps after thresholding probabilities at 0.5 by default.
    """
    prob = torch.sigmoid(logits)
    mse = torch.mean((prob - y) ** 2).item()

    pred_bin = (prob >= threshold).float()
    y_bin = (y >= 0.5).float()

    tp = torch.sum(pred_bin * y_bin).item()
    fp = torch.sum(pred_bin * (1.0 - y_bin)).item()
    fn = torch.sum((1.0 - pred_bin) * y_bin).item()

    iou = tp / max(tp + fp + fn, 1e-8)
    precision = tp / max(tp + fp, 1e-8)
    recall = tp / max(tp + fn, 1e-8)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)

    # Mean entropy of predicted belief. Lower entropy means more confident map.
    p = torch.clamp(prob, 1e-6, 1.0 - 1e-6)
    entropy = torch.mean(-(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))).item()

    return {
        "mse": float(mse),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "entropy": float(entropy),
    }


def evaluate_cnn_belief_model(model, data_loader, criterion, device, threshold: float = 0.5) -> Dict[str, float]:
    """Evaluate CNN updater on validation/test data."""
    model.eval()
    total_samples = 0
    accum = {"loss": 0.0, "mse": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "entropy": 0.0}

    with torch.no_grad():
        for xb, yb in data_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            metrics = belief_metrics_from_logits(logits, yb, threshold=threshold)

            n = xb.size(0)
            total_samples += n
            accum["loss"] += loss.item() * n
            for k in metrics:
                accum[k] += metrics[k] * n

    return {k: v / max(total_samples, 1) for k, v in accum.items()}


def train_cnn_belief_model(
    model,
    train_loader,
    val_loader,
    epochs: int,
    lr: float,
    weight_decay: float,
    device,
    threshold: float = 0.5,
    patience: int = 8,
):
    """
    Train the CNN belief updater.

    We use BCEWithLogitsLoss because each output pixel/cell is a binary
    probability target. This is different from the CSV classifier above,
    which uses CrossEntropyLoss for one class label per row.
    """
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    history = {"loss": [], "val_loss": [], "mse": [], "val_mse": [], "val_iou": [], "val_f1": [], "val_entropy": [], "lr": []}
    best_val_loss = None
    best_state = None
    wait = 0

    progress = st.progress(0)
    status = st.empty()

    for epoch in range(epochs):
        model.train()
        total_samples = 0
        loss_sum = 0.0
        mse_sum = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                metrics = belief_metrics_from_logits(logits, yb, threshold=threshold)

            n = xb.size(0)
            total_samples += n
            loss_sum += loss.item() * n
            mse_sum += metrics["mse"] * n

        train_loss = loss_sum / max(total_samples, 1)
        train_mse = mse_sum / max(total_samples, 1)
        val_metrics = evaluate_cnn_belief_model(model, val_loader, criterion, device, threshold=threshold)
        scheduler.step(val_metrics["loss"])

        history["loss"].append(float(train_loss))
        history["mse"].append(float(train_mse))
        history["val_loss"].append(float(val_metrics["loss"]))
        history["val_mse"].append(float(val_metrics["mse"]))
        history["val_iou"].append(float(val_metrics["iou"]))
        history["val_f1"].append(float(val_metrics["f1"]))
        history["val_entropy"].append(float(val_metrics["entropy"]))
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        progress.progress((epoch + 1) / epochs)
        status.write(
            f"Epoch {epoch + 1}/{epochs} | "
            f"loss={train_loss:.4f}, mse={train_mse:.4f} | "
            f"val_loss={val_metrics['loss']:.4f}, val_mse={val_metrics['mse']:.4f}, "
            f"val_iou={val_metrics['iou']:.4f}, val_f1={val_metrics['f1']:.4f}"
        )

        if best_val_loss is None or val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if wait > patience:
            st.info(f"Early stopping at epoch {epoch + 1}. Restored best validation-loss weights.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def plot_cnn_belief_history(history: Dict[str, List[float]]):
    """Create training curves for CNN updater."""
    fig_loss, ax_loss = plt.subplots(figsize=(8, 5))
    ax_loss.plot(range(1, len(history["loss"]) + 1), history["loss"], label="train")
    ax_loss.plot(range(1, len(history["val_loss"]) + 1), history["val_loss"], label="validation")
    ax_loss.set_title("CNN Belief Updater Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("BCEWithLogitsLoss")
    ax_loss.grid(True, linestyle="--", alpha=0.6)
    ax_loss.legend()

    fig_metrics, ax_metrics = plt.subplots(figsize=(8, 5))
    ax_metrics.plot(range(1, len(history["val_mse"]) + 1), history["val_mse"], label="val_mse")
    ax_metrics.plot(range(1, len(history["val_iou"]) + 1), history["val_iou"], label="val_iou")
    ax_metrics.plot(range(1, len(history["val_f1"]) + 1), history["val_f1"], label="val_f1")
    ax_metrics.set_title("CNN Belief Updater Validation Metrics")
    ax_metrics.set_xlabel("Epoch")
    ax_metrics.grid(True, linestyle="--", alpha=0.6)
    ax_metrics.legend()
    return fig_loss, fig_metrics


def plot_belief_sample(X: np.ndarray, Y: np.ndarray, model=None, device=None, sample_index: int = 0):
    """Visualize one CNN belief-updater sample and optional prediction."""
    x = X[sample_index]
    y = Y[sample_index, 0]

    pred = None
    if model is not None:
        model.eval()
        if device is None:
            device = get_device()
        with torch.no_grad():
            xb = torch.from_numpy(x[None].astype(np.float32)).to(device)
            logits = model.to(device)(xb)
            pred = torch.sigmoid(logits).detach().cpu().numpy()[0, 0]

    ncols = 6 if pred is not None else 5
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
    titles = ["B_prev", "Z_t", "C_t mask", "A_t altitude", "Target M"]
    imgs = [x[0], x[1], x[2], x[3], y]

    if pred is not None:
        titles.append("CNN prediction")
        imgs.append(pred)

    for ax, title, img in zip(axes, titles, imgs):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")

    return fig


def validate_belief_split_shapes(X_train, Y_train, X_val, Y_val, X_test=None, Y_test=None) -> bool:
    """
    Validate that train/validation/test belief-updater arrays are compatible.

    This avoids a common mistake: training on 256x256 patches and validating on
    a different patch size, or using different channel counts. The CNN expects
    all splits to have the same sample shape:
        X: (N, 4, H, W)
        Y: (N, 1, H, W)
    """
    if X_train is None or Y_train is None or X_val is None or Y_val is None:
        return False

    if X_train.shape[1:] != X_val.shape[1:] or Y_train.shape[1:] != Y_val.shape[1:]:
        st.error(
            "Train and validation shapes must match. "
            f"Train X={X_train.shape}, Y={Y_train.shape}; "
            f"Validation X={X_val.shape}, Y={Y_val.shape}."
        )
        return False

    if X_test is not None and Y_test is not None:
        if X_train.shape[1:] != X_test.shape[1:] or Y_train.shape[1:] != Y_test.shape[1:]:
            st.error(
                "Train and test shapes must match. "
                f"Train X={X_train.shape}, Y={Y_train.shape}; "
                f"Test X={X_test.shape}, Y={Y_test.shape}."
            )
            return False

    return True


def split_belief_arrays_three_way(
    X: np.ndarray,
    Y: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_state: int,
):
    """
    Split one generated belief-updater dataset into train/validation/test.

    This mirrors the CSV page idea, but it does not use LabelEncoder or
    StandardScaler because the CNN input channels are already numeric maps.

    Validation is used during training and early stopping.
    Test is reserved for final evaluation after training.
    """
    ratio_sum = train_ratio + val_ratio + test_ratio
    if not np.isclose(ratio_sum, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum:.3f}")

    rng = np.random.default_rng(random_state)
    indices = rng.permutation(X.shape[0])

    n_total = X.shape[0]
    n_train = int(round(n_total * train_ratio))
    n_val = int(round(n_total * val_ratio))

    # Ensure all samples are used and test receives the remaining samples.
    n_train = min(max(n_train, 1), n_total - 2)
    n_val = min(max(n_val, 1), n_total - n_train - 1)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return X[train_idx], Y[train_idx], X[val_idx], Y[val_idx], X[test_idx], Y[test_idx]


def split_belief_arrays_train_val(
    X: np.ndarray,
    Y: np.ndarray,
    val_ratio: float,
    random_state: int,
):
    """
    Split one generated belief-updater dataset into train/validation only.

    This is useful for quick debugging, but it does not provide an honest final
    test set. For final experiments, prefer the Train/Validation/Test split.
    """
    if val_ratio <= 0 or val_ratio >= 1:
        raise ValueError("Validation ratio must be between 0 and 1.")

    rng = np.random.default_rng(random_state)
    indices = rng.permutation(X.shape[0])

    n_total = X.shape[0]
    n_val = int(round(n_total * val_ratio))
    n_val = min(max(n_val, 1), n_total - 1)

    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    return X[train_idx], Y[train_idx], X[val_idx], Y[val_idx]


def cnn_belief_split_controls(prefix: str = "belief") -> Dict:
    """
    Streamlit split controls for the CNN belief-updater page.

    The layout intentionally follows the CSV Configure & Train Model page:
    - Train/Validation/Test for final experiments.
    - Simple Train/Validation for quick tuning/debugging.
    """
    st.subheader("Data Split")
    split_strategy = st.radio(
        "Select data split strategy",
        ["Train/Validation/Test", "Simple Train/Validation"],
        index=0,
        key=f"{prefix}_split_strategy",
        help=(
            "Use Train/Validation/Test for final CNN belief-updater experiments. "
            "Use Simple Train/Validation only for quick debugging."
        ),
    )

    train_ratio = 0.70
    val_ratio = 0.15
    test_ratio = 0.15
    val_only_ratio = 0.15
    split_label = "70_15_15"

    if split_strategy == "Train/Validation/Test":
        split_preset = st.selectbox(
            "Train / Validation / Test ratio",
            ["70 / 15 / 15", "80 / 10 / 10", "60 / 20 / 20", "Custom"],
            index=0,
            key=f"{prefix}_split_preset",
        )

        if split_preset == "70 / 15 / 15":
            train_ratio, val_ratio, test_ratio = 0.70, 0.15, 0.15
        elif split_preset == "80 / 10 / 10":
            train_ratio, val_ratio, test_ratio = 0.80, 0.10, 0.10
        elif split_preset == "60 / 20 / 20":
            train_ratio, val_ratio, test_ratio = 0.60, 0.20, 0.20
        else:
            train_percent = st.slider(
                "Training percentage", min_value=50, max_value=90, value=70, step=5,
                key=f"{prefix}_train_percent",
            )
            val_percent = st.slider(
                "Validation percentage", min_value=5, max_value=40, value=15, step=5,
                key=f"{prefix}_val_percent",
            )
            test_percent = 100 - train_percent - val_percent

            if test_percent <= 0:
                st.error("Invalid split: test percentage must be greater than 0. Reduce train or validation percentage.")
                st.stop()

            train_ratio = train_percent / 100.0
            val_ratio = val_percent / 100.0
            test_ratio = test_percent / 100.0

        split_label = f"{int(train_ratio * 100)}_{int(val_ratio * 100)}_{int(test_ratio * 100)}"
        st.success(
            f"Selected split: **{int(train_ratio * 100)}% train / "
            f"{int(val_ratio * 100)}% validation / "
            f"{int(test_ratio * 100)}% test**"
        )
        st.caption("Validation is used during training and early stopping. Test is used only once after final training.")

    else:
        val_only_ratio = st.radio(
            "Validation split",
            options=[0.10, 0.15, 0.20, 0.25, 0.30, 0.35],
            index=1,
            key=f"{prefix}_val_only_ratio",
        )
        split_label = f"simple_val{int(val_only_ratio * 100)}"
        st.info(
            f"Selected simple split: **{int((1 - val_only_ratio) * 100)}% train / "
            f"{int(val_only_ratio * 100)}% validation**. No separate test set is created."
        )

    return {
        "split_strategy": split_strategy,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "val_only_ratio": val_only_ratio,
        "split_label": split_label,
    }


def cnn_belief_updater_page():
    st.title("🛰️ CNN Belief Updater")

    st.markdown("""
This page is for the **CNN-based belief updater**, not for the CSV dense classifier.

The expected learning problem is:

\[
[B_{t-1}, Z_t, C_t, A_t] \rightarrow M
\]

where:
- **B_prev** is the previous belief map,
- **Z_t** is the noisy UAV observation map,
- **C_t** is the footprint/visibility mask,
- **A_t** is the altitude channel,
- **M** is the target binary terrain map or target updated belief.

Upload `.npz` files generated from your UAV terrain workflow. Each file must contain:

```text
X shape = (N, 4, H, W)
Y shape = (N, 1, H, W)
```

The model uses `Conv2d` layers and `BCEWithLogitsLoss`, because this is **map-to-map binary probability prediction**.
""")

    with st.expander("📘 Why this is separate from the CSV trainer", expanded=False):
        st.markdown("""
The original pages train a dense classifier from CSV rows:

```text
numeric features -> class label
```

The belief updater trains a CNN from spatial maps:

```text
4-channel map -> 1-channel belief/terrain map
```

Therefore, the old `DenseClassifier`, `StandardScaler`, `LabelEncoder`, and `CrossEntropyLoss` are not used here.
""")

    st.subheader("1) Upload generated belief-updater datasets")
    st.markdown("""
You can now manage the CNN dataset in **two ways**, similar in spirit to the split control in
**🧪 Configure & Train Model**:

1. **Upload one full generated `.npz` dataset and split it here** using Train/Validation/Test ratios.
2. **Upload already-separated Train, Validation, and Test `.npz` files**.

For final experiments, prefer a separate test set. The validation set is used during training/early stopping;
the test set is evaluated only after training.
""")

    dataset_mode = st.radio(
        "Dataset input mode",
        [
            "Upload one generated dataset and split here",
            "Upload separate Train / Validation / Test files",
        ],
        index=0,
        key="belief_dataset_mode",
    )

    split_config = None
    X_train = Y_train = X_val = Y_val = X_test = Y_test = None

    if dataset_mode == "Upload one generated dataset and split here":
        st.info(
            "Use this when you generated one large file such as `cnn_belief_all.npz`. "
            "The app will split samples into train/validation/test using the controls below."
        )
        full_file = st.file_uploader("Full generated belief-updater dataset .npz", type=["npz"], key="belief_full_npz")
        X_full, Y_full = load_belief_npz_from_upload(full_file, "Full generated dataset")

        if X_full is not None and Y_full is not None:
            summarize_belief_array("Full generated", X_full, Y_full)

            # Same idea as the CSV model page: user chooses Train/Validation/Test or Simple Train/Validation.
            split_config = cnn_belief_split_controls(prefix="belief_full")
            split_seed = int(st.selectbox("Split random seed", [0, 1, 42, 1337, 2024], index=2, key="belief_split_seed"))

            try:
                if split_config["split_strategy"] == "Train/Validation/Test":
                    X_train, Y_train, X_val, Y_val, X_test, Y_test = split_belief_arrays_three_way(
                        X_full,
                        Y_full,
                        train_ratio=split_config["train_ratio"],
                        val_ratio=split_config["val_ratio"],
                        test_ratio=split_config["test_ratio"],
                        random_state=split_seed,
                    )
                else:
                    X_train, Y_train, X_val, Y_val = split_belief_arrays_train_val(
                        X_full,
                        Y_full,
                        val_ratio=split_config["val_only_ratio"],
                        random_state=split_seed,
                    )
                    X_test, Y_test = None, None
            except Exception as e:
                st.error(f"Could not split full generated dataset: {e}")
                X_train = Y_train = X_val = Y_val = X_test = Y_test = None

    else:
        st.info(
            "Use this when your generator already saved separate files such as "
            "`cnn_belief_train.npz`, `cnn_belief_val.npz`, and `cnn_belief_test.npz`."
        )
        col_train, col_val, col_test = st.columns(3)
        with col_train:
            train_file = st.file_uploader("Train .npz", type=["npz"], key="belief_train_npz")
        with col_val:
            val_file = st.file_uploader("Validation .npz", type=["npz"], key="belief_val_npz")
        with col_test:
            test_file = st.file_uploader("Test .npz", type=["npz"], key="belief_test_npz")

        X_train, Y_train = load_belief_npz_from_upload(train_file, "Train")
        X_val, Y_val = load_belief_npz_from_upload(val_file, "Validation")
        X_test, Y_test = load_belief_npz_from_upload(test_file, "Test") if test_file is not None else (None, None)

    if X_train is not None and Y_train is not None:
        summarize_belief_array("Train", X_train, Y_train)
    if X_val is not None and Y_val is not None:
        summarize_belief_array("Validation", X_val, Y_val)
    if X_test is not None and Y_test is not None:
        summarize_belief_array("Test", X_test, Y_test)
    elif X_train is not None and X_val is not None:
        st.caption("No test set is currently available. This is acceptable for quick debugging, but not ideal for final reporting.")

    if X_train is not None and Y_train is not None:
        st.subheader("2) Visual check of one training sample")
        max_idx = max(int(X_train.shape[0]) - 1, 0)
        sample_idx = st.slider("Sample index", min_value=0, max_value=max_idx, value=0, step=1)
        st.pyplot(plot_belief_sample(X_train, Y_train, sample_index=sample_idx))

    st.subheader("3) CNN training configuration")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        base_channels = st.selectbox("Base CNN channels", [16, 32, 64], index=1)
    with c2:
        batch_size = st.selectbox("Batch size", [2, 4, 8, 16, 32], index=1)
    with c3:
        epochs = int(st.number_input("Epochs", min_value=1, max_value=300, value=30, step=5))
    with c4:
        threshold = float(st.slider("Metric threshold", 0.1, 0.9, 0.5, 0.05))

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        lr = float(st.number_input("Learning rate", min_value=1e-6, max_value=1e-1, value=1e-3, format="%.0e"))
    with c6:
        weight_decay = float(st.number_input("Weight decay", min_value=0.0, max_value=1e-2, value=0.0, format="%.0e"))
    with c7:
        patience = int(st.number_input("Early stopping patience", min_value=0, max_value=50, value=8))
    with c8:
        seed = int(st.selectbox("Random seed", [0, 1, 42, 1337, 2024], index=2))

    use_batch_norm = st.checkbox("Use BatchNorm2d", value=True)
    num_workers = int(st.number_input("DataLoader workers", min_value=0, max_value=8, value=0))

    st.info(
        "Recommended first run for 256×256 patches: base_channels=32, batch_size=4 or 8, "
        "epochs=30, learning_rate=1e-3. If CUDA memory is low, reduce batch size."
    )

    if st.button("🚀 Train CNN Belief Updater"):
        if X_train is None or Y_train is None or X_val is None or Y_val is None:
            st.error("Please provide a valid Train and Validation dataset, either by splitting one full .npz file or by uploading separate .npz files.")
            return

        if not validate_belief_split_shapes(X_train, Y_train, X_val, Y_val, X_test, Y_test):
            return

        set_random_seed(seed)
        device = get_device()
        st.write(f"Using device: `{device}`")

        train_loader, val_loader, test_loader = make_belief_loaders_from_arrays(
            X_train, Y_train,
            X_val, Y_val,
            X_test, Y_test,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        model = CNNBeliefUpdater(
            in_channels=4,
            base_channels=int(base_channels),
            use_batch_norm=use_batch_norm,
        )

        st.subheader("Model summary")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        st.text(str(model))
        st.write(f"Total parameters: **{total_params:,}** | Trainable parameters: **{trainable_params:,}**")

        model, history = train_cnn_belief_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
            threshold=threshold,
            patience=patience,
        )

        st.subheader("4) Training curves")
        fig_loss, fig_metrics = plot_cnn_belief_history(history)
        col_a, col_b = st.columns(2)
        with col_a:
            st.pyplot(fig_loss)
        with col_b:
            st.pyplot(fig_metrics)

        criterion = nn.BCEWithLogitsLoss()
        val_metrics = evaluate_cnn_belief_model(model, val_loader, criterion, device, threshold=threshold)
        st.subheader("5) Final validation metrics")
        st.json({k: round(float(v), 6) for k, v in val_metrics.items()})

        test_metrics = None
        if test_loader is not None:
            test_metrics = evaluate_cnn_belief_model(model, test_loader, criterion, device, threshold=threshold)
            st.subheader("6) Final test metrics")
            st.json({k: round(float(v), 6) for k, v in test_metrics.items()})

        st.subheader("7) Prediction visual check")
        st.pyplot(plot_belief_sample(X_val, Y_val, model=model, device=device, sample_index=0))

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        checkpoint = {
            "model_state_dict": model.cpu().state_dict(),
            "model_type": "CNNBeliefUpdater",
            "in_channels": 4,
            "base_channels": int(base_channels),
            "use_batch_norm": bool(use_batch_norm),
            "history": history,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "input_shape": tuple(X_train.shape[1:]),
            "target_shape": tuple(Y_train.shape[1:]),
            "dataset_mode": dataset_mode,
            "split_config": split_config,
            "n_train": int(X_train.shape[0]),
            "n_val": int(X_val.shape[0]),
            "n_test": int(0 if X_test is None else X_test.shape[0]),
            "explanation": "CNN belief updater: [B_prev, Z_t, C_t, A_t] -> target belief/binary terrain map.",
        }

        models_dir = "trained_cnn_belief_models"
        os.makedirs(models_dir, exist_ok=True)
        checkpoint_filename = f"cnn_belief_updater_{timestamp}.pt"
        checkpoint_path = os.path.join(models_dir, checkpoint_filename)
        torch.save(checkpoint, checkpoint_path)
        st.session_state["last_cnn_checkpoint_path"] = checkpoint_path
        st.session_state["last_cnn_checkpoint_filename"] = checkpoint_filename
        st.success(f"✅ CNN belief updater saved permanently to: {checkpoint_path}")
        st.code(checkpoint_path)

        with open(checkpoint_path, "rb") as f:
            checkpoint_bytes = f.read()
        st.session_state["last_cnn_checkpoint_bytes"] = checkpoint_bytes
        safe_download_button(
            "📥 Download CNN Belief Updater Checkpoint",
            checkpoint_bytes,
            file_name=checkpoint_filename,
            mime="application/octet-stream",
            key="download_cnn_belief_checkpoint",
        )

        summary_lines = [
            "CNN Belief Updater Training Summary",
            f"Timestamp: {timestamp}",
            f"Input: [B_prev, Z_t, C_t, A_t] with shape {tuple(X_train.shape[1:])}",
            f"Target shape: {tuple(Y_train.shape[1:])}",
            f"Dataset mode: {dataset_mode}",
            f"Split strategy: {split_config['split_strategy'] if split_config else 'Separate uploaded files'}",
            f"Split label: {split_config['split_label'] if split_config else 'manual_files'}",
            f"Train samples: {X_train.shape[0]}",
            f"Validation samples: {X_val.shape[0]}",
            f"Test samples: {0 if X_test is None else X_test.shape[0]}",
            f"Base channels: {base_channels}",
            f"Batch size: {batch_size}",
            f"Epochs requested: {epochs}",
            f"Epochs completed: {len(history['loss'])}",
            f"Learning rate: {lr}",
            f"Weight decay: {weight_decay}",
            "",
            "Validation metrics:",
        ]
        summary_lines.extend([f"- {k}: {v:.6f}" for k, v in val_metrics.items()])
        if test_metrics is not None:
            summary_lines.append("")
            summary_lines.append("Test metrics:")
            summary_lines.extend([f"- {k}: {v:.6f}" for k, v in test_metrics.items()])
        summary_text = "\n".join(summary_lines)

        summary_filename = f"cnn_belief_updater_summary_{timestamp}.txt"
        st.session_state["last_cnn_summary_text"] = summary_text
        st.session_state["last_cnn_summary_filename"] = summary_filename
        safe_download_button(
            "📥 Download CNN Training Summary",
            summary_text,
            file_name=summary_filename,
            mime="text/plain",
            key="download_cnn_training_summary",
        )

    # Keep download buttons available after any harmless rerun.
    # This prevents losing access to the trained model if Streamlit reruns.
    if "last_cnn_checkpoint_bytes" in st.session_state:
        st.subheader("Last trained CNN model")
        st.caption("Available from the current Streamlit session.")
        st.code(st.session_state.get("last_cnn_checkpoint_path", ""))
        safe_download_button(
            "📥 Download Last CNN Belief Updater Checkpoint",
            st.session_state["last_cnn_checkpoint_bytes"],
            file_name=st.session_state.get("last_cnn_checkpoint_filename", "cnn_belief_updater.pt"),
            mime="application/octet-stream",
            key="download_last_cnn_checkpoint",
        )
        if "last_cnn_summary_text" in st.session_state:
            safe_download_button(
                "📥 Download Last CNN Training Summary",
                st.session_state["last_cnn_summary_text"],
                file_name=st.session_state.get("last_cnn_summary_filename", "cnn_belief_updater_summary.txt"),
                mime="text/plain",
                key="download_last_cnn_summary",
            )


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



# =========================================================
# Sequential tabular dense-classifier workflow
# =========================================================
# The original app had three sidebar radio pages for the tabular task:
#   1) Introduction & Data Preview
#   2) Preprocessing & Analysis
#   3) Configure & Train Model
# This guided workflow merges those related pages into one sequential flow.
# It is only for the tabular learning problem:
#       numeric features -> class label
# The CNN belief-updater page remains separate because it uses map-to-map data.


def _init_tabular_workflow_state():
    """Initialize the current step index for the tabular workflow."""
    if "tabular_workflow_step" not in st.session_state:
        st.session_state.tabular_workflow_step = 0


def _go_to_tabular_step(step_index: int):
    """Move to a specific tabular workflow step while keeping bounds safe."""
    st.session_state.tabular_workflow_step = int(np.clip(step_index, 0, 2))


def _render_tabular_step_header(steps: List[str]):
    """Render a small progress header for the sequential tabular workflow."""
    current_step = st.session_state.tabular_workflow_step
    st.progress((current_step + 1) / len(steps))

    cols = st.columns(len(steps))
    for i, label in enumerate(steps):
        prefix = "✅" if i < current_step else ("▶️" if i == current_step else "▫️")
        with cols[i]:
            st.markdown(f"**{prefix} Step {i + 1}**")
            st.caption(label)


def tabular_step_upload_preview():
    """Step 1: Upload a CSV file and preview the tabular dataset."""
    st.header("Step 1 — Upload CSV and preview data")
    st.markdown("""
This workflow is for a **tabular dense classifier**:

```text
numeric features -> class label
```

Expected format:
- upload a `.csv` file;
- the **last column** is treated as the target class label;
- all previous selected columns should be numeric features or convertible to numeric values.
""")

    uploaded_file = st.file_uploader(
        "Upload CSV file for numeric-features-to-class-label training",
        type=["csv"],
        key="tabular_csv_upload",
    )
    max_size_mb = st.number_input(
        "Maximum allowed CSV size (MB)",
        min_value=0.1,
        max_value=500.0,
        value=50.0,
        step=5.0,
        key="tabular_max_csv_size_mb",
    )

    if uploaded_file is None:
        st.info("Upload a CSV file to continue to preprocessing.")
        return

    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > max_size_mb:
        st.error(f"🚫 File too large. Limit is {max_size_mb:.2f} MB. Your file is {file_size_mb:.2f} MB.")
        return

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read CSV file: {e}")
        return

    st.session_state.df = df
    # Reset the preprocessed dataframe when a new file is uploaded/read.
    if "df_preprocessed" in st.session_state:
        del st.session_state.df_preprocessed

    st.success("✅ CSV loaded successfully.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("File size", f"{file_size_mb:.2f} MB")

    if df.shape[1] >= 2:
        st.markdown(f"**Detected target column:** `{df.columns[-1]}`")
    else:
        st.warning("The CSV should contain at least one feature column and one label column.")

    st.subheader("Data preview")
    st.dataframe(df.head())

    st.subheader("Column overview")
    overview = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(t) for t in df.dtypes],
        "missing_values": df.isna().sum().values,
        "role": ["feature"] * max(df.shape[1] - 1, 0) + (["label"] if df.shape[1] > 0 else []),
    })
    st.dataframe(overview, use_container_width=True)


def tabular_step_preprocessing():
    """Step 2: Run the existing preprocessing UI for feature selection and balancing."""
    st.header("Step 2 — Preprocess and select features")
    st.markdown("""
Use this step to prepare the tabular dataset before training.
The app will keep the **last column** as the class label and let you select feature columns.

The training function later converts selected features to numeric values, removes invalid rows,
encodes the label column, and fits `StandardScaler` on the training split only.
""")

    if "df" not in st.session_state:
        st.info("Please complete Step 1 first by uploading a CSV file.")
        return

    st.session_state.df_preprocessed = preprocessing_options(st.session_state.df)


def tabular_step_configure_train():
    """Step 3: Configure, tune, train, evaluate, and download the dense classifier."""
    st.header("Step 3 — Configure, tune, train, and evaluate")
    st.markdown("""
This step trains the existing PyTorch **dense classifier** for tabular data.
Use **Train/Validation/Test** for final experiments. The test set is used only after training.
""")

    if "df" not in st.session_state:
        st.info("Please complete Step 1 first by uploading a CSV file.")
        return

    df_to_use = st.session_state.get("df_preprocessed", st.session_state.df)
    process(df_to_use)


def _render_tabular_navigation_buttons():
    """Render Back/Next controls for the sequential tabular workflow."""
    current_step = st.session_state.tabular_workflow_step
    left, mid, right = st.columns([1, 2, 1])

    with left:
        if st.button("⬅️ Back", disabled=current_step == 0, key="tabular_back_button"):
            _go_to_tabular_step(current_step - 1)
            st.rerun()

    with mid:
        st.caption("Sequential workflow for: numeric features → class label")

    with right:
        next_disabled = current_step == 2
        if current_step == 0 and "df" not in st.session_state:
            next_disabled = True
        if current_step == 1 and "df" not in st.session_state:
            next_disabled = True

        if st.button("Next ➡️", disabled=next_disabled, key="tabular_next_button"):
            _go_to_tabular_step(current_step + 1)
            st.rerun()


def tabular_dense_classifier_workflow_page():
    """Single guided page that replaces the old three radio pages for tabular classification."""
    _init_tabular_workflow_state()

    st.title("📊 Tabular Dense Classifier Workflow")
    st.markdown("""
This guided sequence manages the complete tabular workflow:

```text
numeric features -> class label
```

The CNN belief updater is intentionally kept as a separate page because it uses `.npz` map tensors,
not CSV rows.
""")

    steps = [
        "Upload & preview CSV",
        "Preprocess & select features",
        "Configure & train model",
    ]
    _render_tabular_step_header(steps)
    st.divider()

    current_step = st.session_state.tabular_workflow_step
    if current_step == 0:
        tabular_step_upload_preview()
    elif current_step == 1:
        tabular_step_preprocessing()
    else:
        tabular_step_configure_train()

    st.divider()
    _render_tabular_navigation_buttons()


def main_app():
    st.set_page_config(page_title="PyTorch Trainer", layout="wide")
    st.sidebar.markdown("### 📂 Main Workflow")
    workflow = st.sidebar.radio(
        "Select workflow",
        [
            "📊 Tabular Dense Classifier",
            "🛰️ CNN Belief Updater",
        ],
        help=(
            "The tabular workflow is now a sequential step-by-step flow. "
            "The CNN belief updater remains separate because it uses map tensors."
        ),
    )

    st.sidebar.markdown("---")
    st.sidebar.write(f"Torch version: `{torch.__version__}`")
    st.sidebar.write(f"Device: `{get_device()}`")

    if workflow == "📊 Tabular Dense Classifier":
        tabular_dense_classifier_workflow_page()
    elif workflow == "🛰️ CNN Belief Updater":
        cnn_belief_updater_page()


if __name__ == "__main__":
    main_app()
