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

PyTorch loads lazily — `from quant_research.strategies import EnvelopeStrategy` stays torch-free (≈ 200–500 ms saved on startup).

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

Requires Docker installed. No Python needed on host. Two independent
images — pick one, not both:

```bash
# JupyterLab (full image, includes PyTorch ≈ 500 MB → first build 3–10 min)
docker compose --profile jupyter up --build
```

Open <http://localhost:8888>.

```bash
# MkDocs only (slim image, NO PyTorch → first build ~40 s, image 429 MB vs jupyter multi-GB)
docker compose --profile docs up --build
```

Open <http://localhost:8000>.

Stop / clean up:

```bash
docker compose --profile docs down           # or --profile jupyter
```

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
# or
make test
```

Expect `54 passed`.

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
│   └── quant_research/                 # Core library (pip installable, PEP 561 typed)
│       ├── backtest/
│       │   ├── vectorized/             # ML-pipeline PnL (engine, performance) — lazy-loaded
│       │   └── event_driven/           # Bar-loop state machine (Position, BacktestAnalysis)
│       ├── connectors/                 # Exchange data connectors
│       │   ├── binance.py              # Binance Futures historical trades
│       │   ├── bybit.py, coinbase.py, kraken.py, okx.py
│       │   └── ccxt_loader.py          # CCXT unified OHLCV loader
│       ├── engineering/                # Data loading, OHLC aggregation, feature engineering
│       ├── models/                     # PyTorch architectures, training loops, validation — lazy-loaded
│       ├── strategies/                 # Event-driven strategies + indicators (SMA, EMA, WMA, Donchian, RSI, MACD, ATR, Bollinger)
│       ├── utils/                      # plotting (eager) + common reproducibility (lazy)
│       ├── _logging.py                 # Library logger (QUANT_LOG_LEVEL env hook)
│       ├── config.py                   # Global constants (seed, trading days, paths)
│       └── py.typed                    # PEP 561 marker
│
├── accelerator/                        # Learning materials
│   ├── 01_fundamentals/                # 8 Python + stats warmup notebooks
│   ├── 02_ml_strategy/                 # 3-part ML strategy (vectorized paradigm)
│   └── 03_event_driven_strategies/     # Envelope + SMA backtests (event-driven paradigm)
│
├── examples/                           # Runnable .py scripts (no Jupyter required)
│   └── minimal_envelope.py
│
├── data/                               # Gitignored
│   ├── cache/
│   │   ├── *-trades-*.parquet          # Tick trades from Binance/Bybit/Kraken/OKX/Coinbase
│   │   └── ccxt/<exchange>/<tf>/       # OHLCV candles via CCXTLoader + _meta.json
│   └── models/                         # Saved PyTorch weights
│
├── docs/                               # MkDocs: index, getting-started, architecture, api/
├── tests/                              # pytest suite (54 tests)
├── scripts/                            # strip_torch_from_lock.py (lockfile post-processor)
│
├── .github/
│   ├── workflows/{ci,docs}.yml         # CI (ruff+mypy+pytest, py 3.10/3.11/3.12) + mkdocs gh-pages
│   └── dependabot.yml                  # Weekly pip + Actions updates
├── .pre-commit-config.yaml             # ruff + nbstripout + eof-fixer
├── .markdownlint.json                  # Doc lint overrides
│
├── pyproject.toml                      # Project metadata + deps + ruff/mypy/pytest config
├── requirements-lock.txt               # Pinned lockfile (torch stripped — install separately)
├── Makefile                            # make test | lint | fmt | typecheck | docs | lockfile
├── Dockerfile                          # Multi-stage: base → docs (slim, no torch) | jupyter (full)
├── docker-compose.yml                  # Two profiles: --profile docs | --profile jupyter
├── setup.sh | setup.bat                # Local-venv bootstrap (Mac/Linux | Windows)
├── CONTRIBUTING.md                     # Dev workflow + style
└── mkdocs.yml                          # Documentation config
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
| `strategies` | `EnvelopeStrategy`, `SimpleSMAStrategy`, Polars-native indicators (SMA, EMA, WMA, Donchian, RSI, MACD, ATR, Bollinger) |
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

Each `.ipynb` is paired with a `.py` ([jupytext](https://github.com/mwouts/jupytext) sync)
for diff-friendly version control — edit either side, Jupyter keeps them in
sync on save.

### Module 2: ML Strategy (`02_ml_strategy/`)

| Part | Notebook | Focus |
|------|----------|-------|
| 1 | `01-ml_model_pytorch` | Build AR model, train with PyTorch, evaluate performance |
| 2 | `02-strategy_development` | Entry/exit signals, trade sizing, compounding, leverage, liquidation |
| 3 | `03-implementation` | Streaming inference, live trading loop, order management |

### Module 3: Event-Driven Strategies (`03_event_driven_strategies/`)

| Notebook | Focus |
|----------|-------|
| `data_engine` | Download OHLCV candles via `CCXTLoader` (cached as parquet) |
| `run_envelope` | Multi-band `EnvelopeStrategy` backtest (mean reversion, scale-in) |
| `run_sma` | `SimpleSMAStrategy` backtest (triple-SMA trend following) |

Uses the event-driven `quant_research.strategies` classes and
`BacktestAnalysis` for metrics + plots (Altair + Matplotlib).

### Examples (`examples/`)

Plain-Python entry points for copy-paste quickstarts:

```bash
# Download + run an envelope backtest end to end, no Jupyter needed
python examples/minimal_envelope.py --download --symbol BTC/USDT:USDT
```

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

Common commands via [Makefile](Makefile):

| Command           | What it does |
|-------------------|--------------|
| `make test`       | Run pytest (54 tests) |
| `make lint`       | Ruff check (E, F, W, I, B, UP, SIM rules) |
| `make fmt`        | Ruff auto-fix + format |
| `make typecheck`  | Mypy on `src/` (tiered strict) |
| `make docs`       | Build mkdocs site |
| `make docs-serve` | Live-reload docs at <http://127.0.0.1:8000> |
| `make lockfile`   | Regenerate `requirements-lock.txt` from pyproject |
| `make precommit`  | Run all pre-commit hooks on every file |
| `make clean`      | Nuke caches + egg-info + site/ |

Pre-commit hooks (installed via `pre-commit install`) run ruff auto-fix,
[nbstripout](https://github.com/kynan/nbstripout) (wipe notebook cell
outputs for reviewable diffs), and standard hygiene (trailing whitespace,
EOL, large-file guard).

CI runs the same lint + typecheck + test matrix on every push / PR across
Python 3.10 / 3.11 / 3.12 (see [.github/workflows/ci.yml](.github/workflows/ci.yml)).
Docs auto-deploy to GitHub Pages on push to `main`
([.github/workflows/docs.yml](.github/workflows/docs.yml)).
[Dependabot](.github/dependabot.yml) opens weekly update PRs for pip +
GitHub Actions.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full dev flow.

---

## Tech Stack

| Category | Tools |
|----------|-------|
| ML Framework | PyTorch (lazy-loaded) |
| Data | Polars (primary), Pandas, NumPy |
| Exchange API | CCXT (unified OHLCV), raw public-data dumps |
| Visualization | Altair, Matplotlib |
| Documentation | MkDocs Material + mkdocstrings |
| Testing | pytest (54 tests) + pytest-cov |
| Lint / Format | Ruff (`E F W I B UP SIM`) + Mypy (tiered strict) |
| Pre-commit | ruff, nbstripout, eof-fixer, yaml/toml checks |
| CI / CD | GitHub Actions — CI matrix (3.10/3.11/3.12) + docs deploy + Dependabot |
| Containerization | Docker + docker-compose (Jupyter + MkDocs) |
| Data Sources | Binance, Bybit, Coinbase, Kraken, OKX (raw) + any CCXT exchange (OHLCV) |

---

## License

MIT
