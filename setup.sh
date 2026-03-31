#!/bin/bash
set -e

echo "============================================"
echo "  Quant Trading - Environment Setup"
echo "============================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "[ERROR] Python not found. Install Python 3.10+"
    exit 1
fi

PYTHON=$(command -v python3 || command -v python)
PY_VER=$($PYTHON --version 2>&1)
echo "[OK] $PY_VER"

# Create venv
if [ ! -d ".venv" ]; then
    echo "[..] Creating virtual environment..."
    $PYTHON -m venv .venv
    echo "[OK] Virtual environment created"
else
    echo "[OK] Virtual environment exists"
fi

# Activate
source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate

# Upgrade pip
echo "[..] Upgrading pip..."
pip install --upgrade pip --quiet

# Install PyTorch CPU
echo "[..] Installing PyTorch (CPU)..."
pip install "torch>=2.4.0" --index-url https://download.pytorch.org/whl/cpu --quiet

# Install project with all extras
echo "[..] Installing dependencies..."
pip install -e ".[notebook,dev,docs]" --quiet
pip install pandas seaborn scikit-learn "vegafusion[embed]" "vl-convert-python>=1.8.0" --quiet

# Create data directories
mkdir -p data/cache data/models

echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "Open this folder in VS Code."
echo "The kernel '.venv' will be detected automatically."
echo "Just click 'Run All' in any notebook."
