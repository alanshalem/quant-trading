@echo off
setlocal enabledelayedexpansion

echo ============================================
echo   Quant Trading - Environment Setup
echo ============================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10+ from python.org
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set pyver=%%i
echo [OK] Python !pyver!

REM Create venv
if not exist ".venv" (
    echo [..] Creating virtual environment...
    python -m venv .venv
    echo [OK] Virtual environment created
) else (
    echo [OK] Virtual environment exists
)

REM Activate
call .venv\Scripts\activate.bat

REM Upgrade pip
echo [..] Upgrading pip...
python -m pip install --upgrade pip --quiet

REM Install PyTorch CPU first (separate index)
echo [..] Installing PyTorch (CPU)...
pip install "torch>=2.4.0" --index-url https://download.pytorch.org/whl/cpu --quiet

REM Install all other dependencies
echo [..] Installing dependencies...
pip install -e ".[notebook,dev,docs]" --quiet

REM Install extra deps not in pyproject.toml
pip install pandas seaborn scikit-learn vegafusion[embed] --quiet

REM Create data directories
if not exist "data\cache" mkdir data\cache
if not exist "data\models" mkdir data\models

echo.
echo ============================================
echo   Setup complete!
echo ============================================
echo.
echo Open this folder in VS Code.
echo The kernel ".venv" will be detected automatically.
echo Just click "Run All" in any notebook.
echo.
