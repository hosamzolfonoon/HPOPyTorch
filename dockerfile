# --- PyTorch GPU base image ---
FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime
# Environment variable for Node.js major version
ENV NODE_MAJOR=22
WORKDIR /pytorchhpost

RUN apt-get update && \
    apt-get install -y \
        nginx \
        zip \
        nano \
        bash \
        curl \
        gnupg \
        ca-certificates \
        default-mysql-client && \
    mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | \
        gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_${NODE_MAJOR}.x nodistro main" \
        > /etc/apt/sources.list.d/nodesource.list && \
    apt-get update && \
    apt-get install -y nodejs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# Copy application files
COPY pytorchhpost.py requirements.txt start.sh default ./
RUN mkdir -p .streamlit
COPY config.toml secrets.toml ./.streamlit/

# Install PM2 + Python deps
RUN node -v && npm -v
RUN npm install -g npm@11.1.0 pm2@latest
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt

# Permissions + start
RUN chmod +x start.sh
CMD ["bash", "start.sh"]