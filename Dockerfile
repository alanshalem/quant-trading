FROM python:3.12-slim

LABEL maintainer="quant-research"
LABEL description="Quant Trading Research - ML-based trading strategies"

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app:/app/src

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first (better layer caching)
COPY pyproject.toml ./
COPY src/ ./src/

# Install PyTorch CPU-only (separate index)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    "torch>=2.4.0" --index-url https://download.pytorch.org/whl/cpu

# Install all other dependencies
RUN pip install --no-cache-dir \
    "polars>=1.0.0" \
    "numpy>=2.0.0" \
    "altair>=5.4.0" \
    "vegafusion[embed]>=2.0.0" \
    "pandas>=2.2.0" \
    "matplotlib>=3.9.0" \
    "seaborn>=0.13.0" \
    "scikit-learn>=1.5.0" \
    "requests>=2.32.0" \
    "tqdm>=4.66.0" \
    "ipykernel>=7.0.0" \
    "jupyterlab>=4.2.0" \
    "nbformat>=5.10.0" && \
    pip install --no-cache-dir -e ".[dev,docs]"

# Create data directories
RUN mkdir -p data/cache data/models

# Copy the rest of the project
COPY . .

# Expose Jupyter port
EXPOSE 8888
# Expose MkDocs port
EXPOSE 8000

# Default: start JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
