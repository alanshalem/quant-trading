#!/bin/bash
set -e

echo "============================================"
echo "  Quant Trading - Environment Setup"
echo "============================================"
echo ""

# --- Detect Python 3.10+ ---
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo "[ERROR] Python not found. Install Python 3.10+"
    exit 1
fi

PY_VER=$("$PYTHON" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    echo "[ERROR] Python 3.10+ required. Found: $PY_VER"
    exit 1
fi
echo "[OK] Python $PY_VER"

# --- Create venv ---
if [ ! -d ".venv" ]; then
    echo "[..] Creating virtual environment..."
    "$PYTHON" -m venv .venv
    echo "[OK] Virtual environment created"
else
    echo "[OK] Virtual environment exists"
fi

# --- Activate ---
source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate

# --- Upgrade pip ---
echo "[..] Upgrading pip..."
pip install --upgrade pip --quiet

# --- PyTorch (CPU wheel, separate index) ---
# For GPU: set QUANT_TORCH_INDEX=https://download.pytorch.org/whl/cu121 (or latest CUDA)
TORCH_INDEX="${QUANT_TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"
echo "[..] Installing PyTorch from $TORCH_INDEX"
pip install "torch>=2.4.0" --index-url "$TORCH_INDEX" --quiet

# --- Project + extras (everything else lives in pyproject.toml) ---
# [ml] pulls torch from PyPI's default index, but torch is already installed
# above from the CPU (or QUANT_TORCH_INDEX) wheel, so pip skips it.
echo "[..] Installing project and extras..."
pip install -e ".[notebook,dev,docs,ml]" --quiet

# --- Data directories ---
mkdir -p data/cache data/cache/ccxt data/models

echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "Activate later with:"
echo "    source .venv/bin/activate"
echo ""
echo "Open this folder in VS Code and click 'Run All' in any notebook;"
echo "the '.venv' kernel is detected automatically."
