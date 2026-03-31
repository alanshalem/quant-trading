# Quant Research

**A comprehensive Python library for quantitative trading research and ML-based trading strategies.**

Quant Research provides an end-to-end pipeline for building, training, and evaluating machine learning models for quantitative trading. It covers everything from data loading and feature engineering to model training, backtesting, and performance analysis.

## Key Features

- **Data Engineering** - Load, process, and analyze OHLC timeseries data with Polars for high performance
- **ML Models** - Train PyTorch models including Linear, NonLinear, Deep, LSTM, and Attention architectures
- **Backtesting** - Simulate trades, compute equity curves, and evaluate strategy performance with transaction costs
- **Visualization** - Interactive charts with Altair for timeseries, distributions, and performance analysis
- **Reproducibility** - Built-in seed management and deterministic training utilities

## Quick Install

```bash
pip install quant-research
```

Or install from source for development:

```bash
git clone
cd build-a-quant-trading-strategy
pip install -e ".[dev]"
```

## Quick Example

```python
from quant_research import (
    set_seed,
    load_ohlc_timeseries,
    add_lags,
    add_log_return_features,
    train_reg_model,
    eval_model_performance,
    LinearModel,
)

# Reproducibility
set_seed(42)

# Load and engineer features
df = load_ohlc_timeseries("data/trades", interval="1h")
df = add_log_return_features(df)
df = add_lags(df, n_lags=5)

# Train a model
model = LinearModel(n_features=5)
result = train_reg_model(model, df, target_col="log_return")

# Evaluate trading performance
perf = eval_model_performance(result)
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- Polars >= 0.20
- Altair >= 5.0

## License

MIT License.
