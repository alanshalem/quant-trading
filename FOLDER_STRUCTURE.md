# Folder Structure

This document describes the purpose of each directory in the repository.

```
build-a-quant-trading-strategy/
│
├── src/                           # Source code (pip installable package)
│   ├── quant_research/            # Main quantitative research library
│   │   ├── backtest/              # Backtesting engine and performance metrics
│   │   │   ├── engine.py          # Trade simulation, equity curves, compounding
│   │   │   └── performance.py     # Sharpe ratio, drawdown, win rate calculations
│   │   ├── engineering/           # Feature engineering and data processing
│   │   │   ├── loaders.py         # Data loading from parquet cache
│   │   │   ├── processors.py      # OHLC aggregation, log returns, lags
│   │   │   └── analysis.py        # Correlation analysis, autocorrelation
│   │   ├── models/                # Machine learning models and training
│   │   │   ├── architectures.py   # PyTorch model classes (Linear, NonLinear, LSTM, etc.)
│   │   │   ├── trainer.py         # Model training loops and benchmarking
│   │   │   ├── validation.py      # Time series cross-validation splits
│   │   │   ├── inference.py       # Model prediction functions
│   │   │   └── inspection.py      # Model parameter inspection utilities
│   │   ├── utils/                 # Common utilities
│   │   │   ├── common.py          # Seed setting, helper functions
│   │   │   └── plotting.py        # Altair chart generation
│   │   └── config.py              # Library configuration
│   │
│   ├── connectors/                # Exchange data connectors
│   │   ├── base.py                # Abstract base class for connectors
│   │   ├── binance.py             # Binance USDT-M Futures connector
│   │   ├── bybit.py               # Bybit Linear Perpetuals connector
│   │   ├── okx.py                 # OKX Linear Swaps connector
│   │   ├── kraken.py              # Kraken Futures connector
│   │   └── coinbase.py            # Coinbase Advanced Trade connector
│   │
│   └── utils/                     # Development utilities
│       └── convert_to_notebooks.py  # Convert .py scripts to Jupyter notebooks
│
├── data/                          # All data files (gitignored except structure)
│   ├── cache/                     # Downloaded market data cache (parquet files)
│   │   └── *.parquet              # Cached trade data by date
│   └── models/                    # Trained model weights
│       └── model_weights.pth      # Saved PyTorch model state
│
├── accelerator/                   # Quant Trading Accelerator course materials
│   └── notebooks/                 # Training modules (01-08)
│       ├── 01_variables.ipynb     # Python basics: variables and types
│       ├── 02_arrays.ipynb        # NumPy arrays and operations
│       ├── 03_vectorization.ipynb # Vectorized operations for speed
│       ├── 04_time_series.ipynb   # Time series with Polars
│       ├── 05_statistical_edge.ipynb  # Finding trading edge
│       ├── 06_classification.ipynb    # Classification models
│       ├── 07_cross_validation.ipynb  # Time series CV
│       └── 08_strategy_logic.ipynb    # Trading strategy implementation
│
├── tests/                         # Test suite
│   ├── test_models.py             # Tests for model architectures
│   ├── test_connectors.py         # Tests for exchange connectors
│   └── test_backtest.py           # Tests for backtesting functions
│
├── video1.py                      # Video 1: Build ML Model in PyTorch
├── video2.py                      # Video 2: Trading Strategy Development
├── video3.py                      # Video 3: Live Trading Implementation
├── video1.ipynb                   # Jupyter notebook version of video1
├── video2.ipynb                   # Jupyter notebook version of video2
├── video3.ipynb                   # Jupyter notebook version of video3
│
├── pyproject.toml                 # Python package configuration
├── requirements.txt               # Direct pip dependencies
└── FOLDER_STRUCTURE.md            # This file
```

## Directory Purposes

### `src/quant_research/`
The main library containing all quantitative trading research functionality. This is the core package that gets imported in scripts.

**Usage:**
```python
from src.quant_research import (
    LinearModel,
    add_log_return_features,
    eval_model_performance,
)
```

### `src/connectors/`
Exchange-specific data connectors that download and cache market data. All connectors inherit from `BaseConnector` and provide a unified interface.

**Usage:**
```python
from src.connectors.binance import BinanceConnector, MAKER_FEE, TAKER_FEE

connector = BinanceConnector()
df = connector.download_ohlc_timeseries('BTCUSDT', no_days=30, time_interval='12h')
```

### `data/`
Contains all data files. This directory structure should be preserved but contents are gitignored.

- `cache/`: Market data cached as parquet files (downloaded by connectors)
- `models/`: Trained model checkpoints (.pth files)

### `accelerator/notebooks/`
Educational Jupyter notebooks for the Quant Trading Accelerator course. Each module builds on previous concepts.

### `tests/`
pytest test suite. Run with:
```bash
pytest tests/ -v
```

## Installation

```bash
# Install as editable package
pip install -e .

# Or with dev dependencies
pip install -e ".[dev]"
```
