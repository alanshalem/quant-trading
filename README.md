# Quant Trading Research

ML-based quantitative trading strategies using PyTorch, Polars, and real market data from Binance.

## Quick Start

### Option A: Local Setup (recommended)

Requires Python 3.10+ installed.

**Windows:**
```bash
git clone <repo-url>
cd quant-trading
setup.bat
```

**Mac/Linux:**
```bash
git clone <repo-url>
cd quant-trading
bash setup.sh
```

Then open the folder in VS Code. The `.venv` kernel is detected automatically — just open any notebook and click **Run All**.

### Option B: Docker

Requires Docker installed. No Python needed on host.

```bash
docker compose up --build
```

Open http://localhost:8888 for JupyterLab.

For API docs:
```bash
docker compose --profile docs up
```

Open http://localhost:8000 for MkDocs.

---

## Project Structure

```
quant-trading/
├── src/
│   ├── quant_research/            # Core library (pip installable)
│   │   ├── backtest/              # Trade simulation, equity curves, Sharpe ratio
│   │   ├── engineering/           # Data loading, OHLC aggregation, feature engineering
│   │   ├── models/                # PyTorch architectures, training loops, validation
│   │   ├── utils/                 # Reproducibility, tensor helpers, plotting
│   │   └── config.py              # Global constants (seed, trading days, paths)
│   │
│   └── connectors/                # Exchange data connectors
│       ├── binance.py             # Binance Futures historical trades
│       ├── bybit.py               # Bybit connector
│       ├── coinbase.py            # Coinbase connector
│       ├── kraken.py              # Kraken connector
│       └── okx.py                 # OKX connector
│
├── accelerator/                   # Learning materials
│   ├── 01_notebooks/              # 8 modules: Python fundamentals → strategy logic
│   └── 02_strategy/               # 3-part strategy: model → development → implementation
│
├── data/
│   ├── cache/                     # Downloaded trade data (parquet, gitignored)
│   └── models/                    # Saved model weights (gitignored)
│
├── docs/                          # MkDocs documentation source
├── tests/                         # pytest test suite
│
├── pyproject.toml                 # Project metadata and dependencies
├── Dockerfile                     # Docker image definition
├── docker-compose.yml             # Docker services (Jupyter + MkDocs)
├── setup.bat                      # Windows setup script
├── setup.sh                       # Mac/Linux setup script
└── mkdocs.yml                     # Documentation config
```

---

## Trading Pipeline

```
Market Data → Feature Engineering → Model → Signal → Strategy → Execution
```

The system follows a three-step pipeline:

```python
y_hat = model(x)          # 1. Predict future log returns
orders = strategy(y_hat)   # 2. Convert predictions to trade signals
execute(orders)            # 3. Execute trades
```

### Core Library (`quant_research`)

| Module | Purpose |
|--------|---------|
| `engineering` | Load parquet data, create OHLC bars, add log returns and lag features |
| `models` | 5 PyTorch architectures (Linear, NonLinear, Deep, LSTM, Attention), training with LBFGS/Adam |
| `backtest` | Trade simulation, compounding, leverage, transaction fees, performance metrics |
| `utils` | Reproducibility (`set_seed`), Polars→PyTorch conversion, Altair/Matplotlib charts |

### Connectors

Download historical trade data from exchanges. Data is cached as parquet files in `data/cache/`.

```python
from src.connectors.binance import BinanceConnector

connector = BinanceConnector()
connector.download_date_range("BTCUSDT", start_date, end_date)
```

Supported: Binance, Bybit, Coinbase, Kraken, OKX.

---

## Accelerator (Learning Path)

### Module 1: Fundamentals (`01_notebooks/`)

| # | Module | Topics |
|---|--------|--------|
| 01 | Variables | Types, casting, f-strings |
| 02 | Arrays | Lists, NumPy, O(1) vs O(n), log returns |
| 03 | Vectorization | Vector/DataFrame classes, SIMD, Sharpe ratio |
| 04 | Time Series | Stationarity, autocorrelation, AR(1), mean reversion vs momentum |
| 05 | Statistical Edge | Matrix algebra, linear models, directional accuracy |
| 06 | Classification | Logistic regression, confusion matrix, ROC AUC |
| 07 | Cross-Validation | Rolling window, expanding window, walk-forward |
| 08 | Strategy Logic | Entry/exit signals, position sizing, leverage, transaction costs |

All notebooks available in English and Spanish (`_es` suffix).

### Module 2: Strategy (`02_strategy/`)

| Part | Notebook | Focus |
|------|----------|-------|
| 1 | `01-ml_model_pytorch` | Build AR model, train with PyTorch, evaluate performance |
| 2 | `02-strategy_development` | Entry/exit signals, trade sizing, compounding, leverage, liquidation |
| 3 | `03-implementation` | Streaming inference, live trading loop, order management |

Available in English and Spanish.

---

## Model Architectures

```python
from quant_research.models.architectures import (
    LinearModel,       # AR(n) linear regression
    NonLinearModel,    # Single hidden layer + ReLU
    DeepModel,         # Multi-layer with BatchNorm + Dropout
    LSTMModel,         # LSTM for sequence modeling
    AttentionModel,    # Self-attention mechanism
)
```

---

## Configuration

Global settings in `src/quant_research/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SEED` | 42 | Reproducibility seed |
| `TRADING_DAYS_PER_YEAR` | 365 | For Sharpe annualization (crypto) |
| `TRADING_HOURS_PER_DAY` | 24 | For Sharpe annualization (crypto) |
| `DEFAULT_EPOCHS` | 6000 | Training epochs |
| `DEFAULT_LEARNING_RATE` | 0.0002 | Adam learning rate |
| `DEFAULT_TEST_SIZE` | 0.25 | Train/test split ratio |

---

## API Documentation

Generated from docstrings with MkDocs + mkdocstrings.

**Local:**
```bash
# With venv
.venv/Scripts/python.exe -m mkdocs serve    # Windows
.venv/bin/python -m mkdocs serve            # Mac/Linux

# With Docker
docker compose --profile docs up
```

Open http://localhost:8000.

---

## Development

### Run Tests

```bash
.venv/Scripts/python.exe -m pytest          # Windows
.venv/bin/python -m pytest                  # Mac/Linux
```

### Lint

```bash
.venv/Scripts/python.exe -m ruff check src/ # Windows
.venv/bin/python -m ruff check src/         # Mac/Linux
```

### Type Check

```bash
.venv/Scripts/python.exe -m mypy src/       # Windows
.venv/bin/python -m mypy src/               # Mac/Linux
```

---

## Tech Stack

| Category | Tools |
|----------|-------|
| ML Framework | PyTorch |
| Data | Polars, Pandas, NumPy |
| Visualization | Altair, Matplotlib, Seaborn |
| Documentation | MkDocs Material + mkdocstrings |
| Testing | pytest |
| Linting | Ruff, mypy |
| Containerization | Docker |
| Data Sources | Binance, Bybit, Coinbase, Kraken, OKX |

---

## License

MIT
