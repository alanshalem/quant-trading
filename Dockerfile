FROM python:3.12-slim

LABEL maintainer="quant-research"
LABEL description="Quant Trading Research - ML-based trading strategies"

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# System deps needed by torch + general scientific Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first (better layer caching)
COPY pyproject.toml ./
COPY src/ ./src/

# PyTorch CPU-only wheel (separate index).
# Override at build time with: --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "torch>=2.4.0" --index-url ${TORCH_INDEX}

# Everything else (deps + notebook + dev + docs extras) comes from pyproject.toml
RUN pip install --no-cache-dir -e ".[notebook,dev,docs]"

# Data directories
RUN mkdir -p data/cache data/cache/ccxt data/models

# Copy the rest of the project
COPY . .

# JupyterLab port
EXPOSE 8888
# MkDocs port
EXPOSE 8000

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
