@echo off
setlocal enabledelayedexpansion

echo ============================================
echo   Quant Trading - Environment Setup
echo ============================================
echo.

REM --- Detect Python ---
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10-3.13 from python.org
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

REM PyTorch (the [ml] extra) currently ships no wheels for Python 3.14+.
set TORCH_UNSUPPORTED=0
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 14) else 1)" && set TORCH_UNSUPPORTED=1

set VENV_PY=.venv\Scripts\python.exe

REM --- Create or repair venv ---
REM A venv copied/moved from another machine keeps an absolute path to its
REM original base interpreter and its python.exe becomes a dead shim. Detect
REM that and rebuild rather than trusting mere directory existence.
set VENV_OK=0
if exist "!VENV_PY!" (
    "!VENV_PY!" -c "import sys" >nul 2>&1 && set VENV_OK=1
)
if "!VENV_OK!"=="1" (
    echo [OK] Virtual environment exists
) else (
    if exist ".venv" (
        echo [..] Existing .venv is broken - rebuilding...
        rmdir /s /q .venv
    ) else (
        echo [..] Creating virtual environment...
    )
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        exit /b 1
    )
    echo [OK] Virtual environment ready
)

REM --- Upgrade pip (use the venv interpreter explicitly; do not rely on PATH) ---
echo [..] Upgrading pip...
"!VENV_PY!" -m pip install --upgrade pip --quiet

REM --- PyTorch (CPU wheel). For GPU: set QUANT_TORCH_INDEX=https://download.pytorch.org/whl/cu121
if "!TORCH_UNSUPPORTED!"=="1" (
    echo [WARN] Python !pyver! has no PyTorch wheels. Skipping the [ml] extra.
    echo        Classical ^(02^) and event-driven ^(04^) notebooks work without it.
    echo        For the ML notebooks ^(03^) use Python 3.10-3.13 or Docker.
    echo [..] Installing project and extras ^(no ml^)...
    "!VENV_PY!" -m pip install -e ".[notebook,dev,docs]" --quiet
) else (
    if not defined QUANT_TORCH_INDEX set QUANT_TORCH_INDEX=https://download.pytorch.org/whl/cpu
    echo [..] Installing PyTorch from !QUANT_TORCH_INDEX!
    "!VENV_PY!" -m pip install "torch>=2.4.0" --index-url !QUANT_TORCH_INDEX! --quiet
    if errorlevel 1 (
        echo [WARN] PyTorch install failed - continuing without the [ml] extra.
        echo [..] Installing project and extras ^(no ml^)...
        "!VENV_PY!" -m pip install -e ".[notebook,dev,docs]" --quiet
    ) else (
        echo [..] Installing project and extras...
        "!VENV_PY!" -m pip install -e ".[notebook,dev,docs,ml]" --quiet
    )
)

REM --- Data directories ---
if not exist "data\cache" mkdir data\cache
if not exist "data\cache\ccxt" mkdir data\cache\ccxt
if not exist "data\models" mkdir data\models

echo.
echo ============================================
echo   Setup complete
echo ============================================
echo.
echo Activate later with:
echo     .venv\Scripts\activate.bat
echo.
echo Open this folder in VS Code and click "Run All" in any notebook;
echo the ".venv" kernel is detected automatically.
echo.
