############################################################
# Multi-stage Dockerfile
#
# Stages:
#   base     — common layers: python + apt deps + project skeleton
#   docs     — slim image for mkdocs; NO torch (docs don't need it)
#   jupyter  — full image for notebooks; installs torch (CPU by default)
#
# docker-compose maps:
#   docs     service → target=docs
#   jupyter  service → target=jupyter
#
# Build just one manually:
#   docker build --target=docs -t quant-trading-docs .
#   docker build --target=jupyter -t quant-trading-jupyter .
############################################################

# ---------- base ----------
FROM python:3.12-slim AS base

LABEL maintainer="quant-research"

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# System deps (build-essential needed for any C-extension builds; curl + git
# needed by setuptools for VCS-like metadata).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first so the deps layer caches across rebuilds.
COPY pyproject.toml ./
COPY src/ ./src/

RUN pip install --no-cache-dir --upgrade pip

# Data directories shared by both runtime images.
RUN mkdir -p data/cache data/cache/ccxt data/models


# ---------- docs (no torch; slim) ----------
FROM base AS docs

LABEL target="docs"

# Install ONLY the docs extra — skips torch, pandas, sklearn, etc.
# Editable install still registers the quant_research package so
# mkdocstrings can resolve ``:::`` references.
RUN pip install --no-cache-dir -e ".[docs]"

COPY mkdocs.yml ./
COPY docs/ ./docs/
COPY README.md ./

EXPOSE 8000
CMD ["mkdocs", "serve", "--dev-addr", "0.0.0.0:8000"]


# ---------- jupyter (full; with torch) ----------
FROM base AS jupyter

LABEL target="jupyter"

# PyTorch from a separate index (CPU wheel by default — override for GPU).
#   docker build --target=jupyter --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121 .
# Installing torch here FIRST means the subsequent `.[ml]` extra finds it
# satisfied and won't fall back to PyPI's default (CUDA-bundled) wheel.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir "torch>=2.4.0" --index-url ${TORCH_INDEX}

# Everything (notebook + dev + docs + ml).
RUN pip install --no-cache-dir -e ".[notebook,dev,docs,ml]"

# Copy the rest of the project.
COPY . .

EXPOSE 8888
EXPOSE 8000
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
