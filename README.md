# Quant Trading Research

[![CI](https://github.com/alanshalem/quant-trading/actions/workflows/ci.yml/badge.svg)](https://github.com/alanshalem/quant-trading/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Polars](https://img.shields.io/badge/polars-1.0%2B-orange)
![PyTorch](https://img.shields.io/badge/pytorch-2.4%2B-ee4c2c)

Two complementary backtesting paradigms on Polars-native OHLCV data:

- **Vectorized** — PyTorch model predictions → trade log-returns → equity curves via Polars expressions. Fast to iterate; used for ML research.
- **Event-driven** — Bar-by-bar state machine with `Position`, SL/TP/liquidation, fees and pyramid-in support. Used for discretionary rules and realistic execution.

Includes ready-made `EnvelopeStrategy` / `SimpleSMAStrategy` (event-driven) and 5 PyTorch architectures (vectorized), plus data connectors for Binance, Bybit, Coinbase, Kraken, OKX (raw trades) and any exchange CCXT supports (OHLCV candles).

See [`docs/architecture.md`](docs/architecture.md) for a full module map and paradigm comparison.

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

Open <http://localhost:8888> for JupyterLab.

For API docs:

```bash
docker compose --profile docs up
```

Open <http://localhost:8000> for MkDocs.

---

## Sanity check

After `bash setup.sh`, verify the install:

```bash
.venv/bin/python -c "
from quant_research.connectors import CCXTLoader
from quant_research.backtest import Position, BacktestAnalysis
from quant_research.strategies import EnvelopeStrategy
print('imports ok')
"
```

Run the test suite:

```bash
.venv/bin/python -m pytest -q
```

Expect `28 passed`.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Running cells with 'Python 3.12.x' requires the ipykernel package` | VS Code picked the system interpreter, not `.venv` | `Ctrl+Shift+P` → **Python: Select Interpreter** → `.venv/bin/python`. Then open notebook → kernel picker top-right → `.venv`. |
| `ModuleNotFoundError: No module named 'quant_research'` | Editable install didn't register (pre-reorg cache) | `rm -rf src/quant_research.egg-info && .venv/bin/pip install -e .` |
| `error: externally-managed-environment` on `pip install` | Running pip against system Python (PEP 668) | Always activate the venv first: `source .venv/bin/activate`. |
| `CCXTLoader` raises `FileNotFoundError: No cached data` | You called `.load()` before `.download()` | Run the `data_engine.ipynb` cell or call `.download(symbol, timeframe, start_date=...)` once. |
| Torch installs take forever / download > 800 MB | Default is CPU wheel; GPU wheels are bigger | Accept CPU (default). For GPU: `QUANT_TORCH_INDEX=https://download.pytorch.org/whl/cu121 bash setup.sh`. |

---

## Project Structure

```text
quant-trading/
├── src/
│   └── quant_research/                 # Core library (pip installable)
│       ├── backtest/
│       │   ├── vectorized/             # ML-pipeline PnL (engine, performance)
│       │   └── event_driven/           # Bar-loop state machine (Position, BacktestAnalysis)
│       ├── connectors/                 # Exchange data connectors
│       │   ├── binance.py              # Binance Futures historical trades
│       │   ├── bybit.py                # Bybit connector
│       │   ├── coinbase.py, kraken.py, okx.py
│       │   └── ccxt_loader.py          # CCXT unified OHLCV loader
│       ├── engineering/                # Data loading, OHLC aggregation, feature engineering
│       ├── models/                     # PyTorch architectures, training loops, validation
│       ├── strategies/                 # Event-driven strategies (Envelope, SimpleSMA) + indicators
│       ├── utils/                      # Reproducibility, tensor helpers, plotting
│       └── config.py                   # Global constants (seed, trading days, paths)
│
├── accelerator/                        # Learning materials
│   ├── 01_fundamentals/                # 8 modules: Python fundamentals → strategy logic
│   ├── 02_ml_strategy/                 # 3-part ML strategy: model → development → implementation
│   └── 03_event_driven_strategies/     # Envelope + SMA event-driven backtests
│
├── data/                          # Gitignored
│   ├── cache/
│   │   ├── *-trades-*.parquet     # Tick trades from Binance/Bybit/Kraken/OKX/Coinbase connectors
│   │   └── ccxt/<exchange>/<tf>/  # OHLCV candles via CCXTLoader
│   └── models/                    # Saved PyTorch weights
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

```text
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
| `connectors` | Exchange data: Binance/Bybit/Coinbase/Kraken/OKX tick trades + `CCXTLoader` OHLCV candles |
| `engineering` | Load parquet data, create OHLC bars, add log returns and lag features |
| `models` | 5 PyTorch architectures (Linear, NonLinear, Deep, LSTM, Attention), training with LBFGS/Adam |
| `backtest.vectorized` | ML-pipeline PnL: predictions → trade log-returns → fees → equity curves (Polars-only) |
| `backtest.event_driven` | Bar-loop state machine: `Position`, SL/TP/liquidation, `BacktestAnalysis` metrics + plots |
| `strategies` | `EnvelopeStrategy`, `SimpleSMAStrategy`, Polars-native indicators (SMA/EMA/WMA/Donchian) |
| `utils` | Reproducibility (`set_seed`), Polars→PyTorch conversion, Altair/Matplotlib charts |

### Connectors

Download historical trade data from exchanges. Data is cached as parquet files in `data/cache/`.

```python
from quant_research.connectors.binance import BinanceConnector

connector = BinanceConnector()
connector.download_date_range("BTCUSDT", start_date, end_date)
```

Supported: Binance, Bybit, Coinbase, Kraken, OKX.

**CCXT OHLCV candles (any exchange)** — for event-driven strategies that
consume pre-aggregated bars instead of tick trades:

```python
from quant_research.connectors import CCXTLoader

loader = CCXTLoader(exchange="binanceusdm")
loader.download("BTC/USDT:USDT", timeframe="1h", start_date="2023-01-01")
df = loader.load("BTC/USDT:USDT", timeframe="1h")  # polars DataFrame
```

---

## Accelerator (Learning Path)

### Module 1: Fundamentals (`01_fundamentals/`)

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

All notebooks available in English and Spanish (`_es` suffix). Each `.ipynb`
is paired with a `.py` ([jupytext](https://github.com/mwouts/jupytext) sync)
for diff-friendly version control — edit either side, Jupyter keeps them in
sync on save.

### Module 2: ML Strategy (`02_ml_strategy/`)

| Part | Notebook | Focus |
|------|----------|-------|
| 1 | `01-ml_model_pytorch` | Build AR model, train with PyTorch, evaluate performance |
| 2 | `02-strategy_development` | Entry/exit signals, trade sizing, compounding, leverage, liquidation |
| 3 | `03-implementation` | Streaming inference, live trading loop, order management |

Available in English and Spanish.

### Module 3: Event-Driven Strategies (`03_event_driven_strategies/`)

| Notebook | Focus |
|----------|-------|
| `data_engine` | Download OHLCV candles via `CCXTLoader` (cached as parquet) |
| `run_envelope` | Multi-band `EnvelopeStrategy` backtest (mean reversion, scale-in) |
| `run_sma` | `SimpleSMAStrategy` backtest (triple-SMA trend following) |

Uses the event-driven `quant_research.strategies` classes and
`BacktestAnalysis` for metrics + plots (Altair + Matplotlib).

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
| `TRADING_DAYS_PER_YEAR` | 365 | Sharpe annualization — crypto trades 24/7/365 (use 252 for equities) |
| `TRADING_HOURS_PER_DAY` | 24 | Sharpe annualization — crypto (use 6.5 for US equities RTH) |
| `DEFAULT_EPOCHS` | 6000 | Training epochs |
| `DEFAULT_LEARNING_RATE` | 0.0002 | Adam learning rate |
| `DEFAULT_TEST_SIZE` | 0.25 | Train/test split ratio |

Override at the call site instead of editing `config.py`:

```python
from quant_research.backtest import sharpe_annualization_factor
equities_sharpe_factor = sharpe_annualization_factor("1h", 252, 6.5)
```

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

Open <http://localhost:8000>.

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
