@echo off
setlocal enabledelayedexpansion

echo ============================================
echo   Quant Trading - Environment Setup
echo ============================================
echo.

REM --- Detect Python ---
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10+ from python.org
    exit /b 1
)

REM --- Version check (3.10+) ---
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"
if errorlevel 1 (
    for /f "tokens=2" %%i in ('python --version 2^>^&1') do set pyver=%%i
    echo [ERROR] Python 3.10+ required. Found: !pyver!
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set pyver=%%i
echo [OK] Python !pyver!

REM --- Create venv ---
if not exist ".venv" (
    echo [..] Creating virtual environment...
    python -m venv .venv
    echo [OK] Virtual environment created
) else (
    echo [OK] Virtual environment exists
)

REM --- Activate ---
call .venv\Scripts\activate.bat

REM --- Upgrade pip ---
echo [..] Upgrading pip...
python -m pip install --upgrade pip --quiet

REM --- PyTorch (CPU wheel). For GPU: set QUANT_TORCH_INDEX=https://download.pytorch.org/whl/cu121
if not defined QUANT_TORCH_INDEX set QUANT_TORCH_INDEX=https://download.pytorch.org/whl/cpu
echo [..] Installing PyTorch from !QUANT_TORCH_INDEX!
pip install "torch>=2.4.0" --index-url !QUANT_TORCH_INDEX! --quiet

REM --- Project + extras (torch already installed above via CPU index) ---
echo [..] Installing project and extras...
pip install -e ".[notebook,dev,docs,ml]" --quiet

REM --- Data directories ---
if not exist "data\cache" mkdir data\cache
if not exist "data\cache\ccxt" mkdir data\cache\ccxt
if not exist "data\models" mkdir data\models

echo.
echo ============================================
echo   Setup complete!
echo ============================================
echo.
echo Activate later with:
echo     .venv\Scripts\activate.bat
echo.
echo Open this folder in VS Code and click "Run All" in any notebook;
echo the ".venv" kernel is detected automatically.
echo.
