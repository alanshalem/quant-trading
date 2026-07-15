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
    echo "[ERROR] Python not found. Install Python 3.10-3.13"
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

# PyTorch (the [ml] extra) currently ships no wheels for Python 3.14+.
TORCH_UNSUPPORTED=0
if [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -ge 14 ]; then
    TORCH_UNSUPPORTED=1
fi

# Resolve the venv interpreter (Linux/macOS vs Git-Bash on Windows).
if [ -f ".venv/bin/python" ]; then
    VENV_PY=".venv/bin/python"
else
    VENV_PY=".venv/Scripts/python.exe"
fi

# --- Create or repair venv ---
# A venv copied/moved from another machine keeps an absolute path to its
# original base interpreter and its python becomes a dead shim. Detect that and
# rebuild rather than trusting mere directory existence.
if [ -d ".venv" ] && "$VENV_PY" -c "import sys" >/dev/null 2>&1; then
    echo "[OK] Virtual environment exists"
else
    if [ -d ".venv" ]; then
        echo "[..] Existing .venv is broken - rebuilding..."
        rm -rf .venv
    else
        echo "[..] Creating virtual environment..."
    fi
    "$PYTHON" -m venv .venv
    echo "[OK] Virtual environment ready"
fi
# Re-resolve after (re)creation.
if [ -f ".venv/bin/python" ]; then VENV_PY=".venv/bin/python"; else VENV_PY=".venv/Scripts/python.exe"; fi

# --- Upgrade pip (use the venv interpreter explicitly; do not rely on PATH) ---
echo "[..] Upgrading pip..."
"$VENV_PY" -m pip install --upgrade pip --quiet

# --- PyTorch (CPU wheel, separate index). GPU: set QUANT_TORCH_INDEX=.../whl/cu121
if [ "$TORCH_UNSUPPORTED" -eq 1 ]; then
    echo "[WARN] Python $PY_VER has no PyTorch wheels. Skipping the [ml] extra."
    echo "       Classical (02) and event-driven (04) notebooks work without it."
    echo "       For the ML notebooks (03) use Python 3.10-3.13 or Docker."
    echo "[..] Installing project and extras (no ml)..."
    "$VENV_PY" -m pip install -e ".[notebook,dev,docs]" --quiet
else
    TORCH_INDEX="${QUANT_TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"
    echo "[..] Installing PyTorch from $TORCH_INDEX"
    if "$VENV_PY" -m pip install "torch>=2.4.0" --index-url "$TORCH_INDEX" --quiet; then
        echo "[..] Installing project and extras..."
        "$VENV_PY" -m pip install -e ".[notebook,dev,docs,ml]" --quiet
    else
        echo "[WARN] PyTorch install failed - continuing without the [ml] extra."
        echo "[..] Installing project and extras (no ml)..."
        "$VENV_PY" -m pip install -e ".[notebook,dev,docs]" --quiet
    fi
fi

# --- Data directories ---
mkdir -p data/cache data/cache/ccxt data/models

echo ""
echo "============================================"
echo "  Setup complete"
echo "============================================"
echo ""
echo "Activate later with:"
echo "    source .venv/bin/activate"
echo ""
echo "Open this folder in VS Code and click 'Run All' in any notebook;"
echo "the '.venv' kernel is detected automatically."
