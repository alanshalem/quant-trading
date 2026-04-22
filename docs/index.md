# Quant Research

**Python library for quantitative trading research: feature engineering, ML models, and two complementary backtesting paradigms (vectorized PnL and event-driven bar-loop) — all built on Polars.**

## What's inside

- **Data engineering** — Load tick data, aggregate to OHLC bars, build log-return and lag features.
- **ML models** — PyTorch architectures (`LinearModel`, `NonLinearModel`, `DeepModel`, `LSTMModel`, `AttentionModel`) with LBFGS / Adam training loops.
- **Vectorized backtest** — ML-pipeline PnL. Model predictions → trade log-returns → fees → equity curves, all via Polars expressions.
- **Event-driven backtest** — Bar-by-bar state machine (`Position`, SL/TP/liquidation) with rich metrics and plots via `BacktestAnalysis`.
- **Strategies** — Ready-made `EnvelopeStrategy` (multi-band mean reversion) and `SimpleSMAStrategy` (triple-SMA trend following).
- **Connectors** — Binance, Bybit, Coinbase, Kraken, OKX (raw trades) and `CCXTLoader` (unified OHLCV via [ccxt](https://github.com/ccxt/ccxt)).

See [Architecture](architecture.md) for the full module map and when to use which paradigm.

## Install

This package is not on PyPI. Install from source:

```bash
git clone <repo-url> quant-trading
cd quant-trading
bash setup.sh          # or: setup.bat on Windows
```

The script creates `.venv/`, installs PyTorch (CPU wheel by default), then runs `pip install -e ".[notebook,dev,docs]"`.

## Quick example — event-driven envelope strategy

```python
from quant_research.connectors import CCXTLoader
from quant_research.strategies import EnvelopeStrategy
from quant_research.backtest import BacktestAnalysis

# 1. Fetch OHLCV candles (first run downloads + caches to parquet)
loader = CCXTLoader(exchange="binanceusdm")
loader.download("BTC/USDT:USDT", timeframe="1h", start_date="2023-01-01")
df = loader.load("BTC/USDT:USDT", timeframe="1h")  # polars DataFrame

# 2. Configure + run
strategy = EnvelopeStrategy(
    params={
        "average_type": "SMA",
        "average_period": 6,
        "envelopes": [0.07, 0.11, 0.14],
        "stop_loss_pct": 0.3,
        "position_size_percentage": 100,
    },
    ohlcv=df,
)
strategy.run_backtest(initial_balance=1000, leverage=1,
                      open_fee_rate=0.0002, close_fee_rate=0.0006)

# 3. Metrics + plots
results = BacktestAnalysis(strategy)
results.print_metrics()
results.plot_equity()
```

## Quick example — vectorized ML-pipeline PnL

```python
import polars as pl
import torch
from quant_research.utils import set_seed
from quant_research.engineering import load_ohlc_timeseries, add_log_return_features
from quant_research.models import LinearModel, train_reg_model
from quant_research.backtest import (
    learn_model_trades, add_compounding_trades, sharpe_annualization_factor,
)

set_seed(42)

# 1. Load OHLC bars previously downloaded via a connector
df = load_ohlc_timeseries("BTCUSDT", "1h")

# 2. Add close_log_return + 5 lagged returns
df = add_log_return_features(df, col="close", forecast_horizon=1, max_no_lags=5)
features = [f"close_log_return_lag_{i}" for i in range(1, 6)]
target = "close_log_return"

# 3. Train + get trade-level results in one call
model = LinearModel(len(features))
trades = learn_model_trades(df, features=features, target=target, model=model)

# 4. Apply fees + compounding
trades = trades.join(df.select(["open", "close_log_return"]), on="close_log_return", how="left")
trades = add_compounding_trades(trades, capital=10_000, leverage=1.0,
                                maker_fee=0.0001, taker_fee=0.0003)
print(trades.select("equity_curve_taker").tail(1))
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.4
- Polars >= 1.0
- Altair >= 5.4

See [pyproject.toml](https://github.com/) for the full dependency list.

## License

MIT.
