# Getting Started

This guide walks you through the core workflow of the Quant Research library: from loading raw trade data to evaluating a trained model's trading performance.

## Pipeline Overview

The library follows a three-stage pipeline:

```mermaid
graph LR
    A[Engineering] -->|Features & Labels| B[Models]
    B -->|Predictions| C[Backtest]

    A:::eng
    B:::mod
    C:::bt

    classDef eng fill:#4051b5,color:#fff,stroke:#4051b5
    classDef mod fill:#7c4dff,color:#fff,stroke:#7c4dff
    classDef bt fill:#00bfa5,color:#fff,stroke:#00bfa5
```

1. **Engineering** - Load raw trade data, resample into OHLC bars, and compute features (log returns, lags, correlations).
2. **Models** - Split data with time-aware methods, train PyTorch regression models, and validate out-of-sample.
3. **Backtest** - Convert model predictions into trading signals, simulate trades with transaction costs, and compute performance metrics (Sharpe ratio, equity curves).

## Installation

```bash
pip install quant-research
```

For development with testing and linting tools:

```bash
pip install -e ".[dev]"
```

## Step 1: Load and Prepare Data

Use the engineering module to load OHLC timeseries and create features:

```python
from quant_research.engineering import (
    load_ohlc_timeseries,
    add_lags,
    add_log_return_features,
)

# Load 1-hour OHLC bars from trade files
df = load_ohlc_timeseries("data/trades", interval="1h")

# Add log return features and lagged columns
df = add_log_return_features(df)
df = add_lags(df, n_lags=5)
```

## Step 2: Train a Model

Choose a model architecture and train it:

```python
from quant_research.utils import set_seed
from quant_research.models import (
    LinearModel,
    timeseries_train_test_split,
    train_reg_model,
)

set_seed(42)

# Time-aware train/test split
train_df, test_df = timeseries_train_test_split(df, test_size=0.25)

# Train a linear regression model
model = LinearModel(n_features=5)
result = train_reg_model(model, train_df, target_col="log_return")
```

### Available Architectures

| Model | Description |
|-------|-------------|
| `LinearModel` | Simple linear regression |
| `NonLinearModel` | Single hidden layer with ReLU |
| `DeepModel` | Multi-layer deep network |
| `LSTMModel` | LSTM-based sequence model |
| `AttentionModel` | Self-attention mechanism |

## Step 3: Backtest the Strategy

Evaluate how the model's predictions translate into trading performance:

```python
from quant_research.backtest import (
    eval_model_performance,
    add_tx_fees,
    add_equity_curve,
)

# Evaluate full model performance (Sharpe, returns, etc.)
perf = eval_model_performance(result)

# Or build a detailed trade log with transaction costs
trades = add_tx_fees(result, fee_bps=5)
trades = add_equity_curve(trades)
```

## Step 4: Visualize Results

Use the built-in plotting utilities:

```python
from quant_research.utils import plot_dyn_timeseries, plot_distribution

# Interactive timeseries chart
plot_dyn_timeseries(trades, y_col="equity", title="Equity Curve")

# Return distribution
plot_distribution(trades, col="trade_return", title="Trade Returns")
```

## Next Steps

- Browse the [API Reference](api/backtest.md) for detailed documentation of every function and class.
- Check the Jupyter notebooks in the repository for complete worked examples.
