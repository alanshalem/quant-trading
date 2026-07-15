# Getting Started

This page walks through both backtesting paradigms end to end. Pick whichever matches your use case — they share data loaders and indicators but diverge at the trade-generation step.

See [Architecture](architecture.md) for a high-level map.

## Install

```bash
git clone <repo-url> quant-trading
cd quant-trading
bash setup.sh          # Mac/Linux
setup.bat              # Windows
```

Activate the venv afterward (`source .venv/bin/activate`) or let VS Code pick up the `.venv` kernel automatically when opening notebooks.

---

## Path A — Event-driven strategy (envelope / SMA)

Uses pre-aggregated OHLCV candles and a bar-by-bar state machine.

### 1. Download candles

```python
from quant_research.connectors import CCXTLoader

loader = CCXTLoader(exchange="binanceusdm")
loader.download("BTC/USDT:USDT", timeframe="1h", start_date="2023-01-01")
df = loader.load("BTC/USDT:USDT", timeframe="1h")
```

`CCXTLoader` caches parquet files under `data/cache/ccxt/<exchange>/<timeframe>/`. Re-calling `.load()` reads from cache — no network hit.

### 2. Configure the strategy

```python
from quant_research.strategies import EnvelopeStrategy

strategy = EnvelopeStrategy(
    params={
        "average_type": "SMA",         # 'SMA' | 'EMA' | 'WMA' | 'DCM'
        "average_period": 6,
        "envelopes": [0.07, 0.11, 0.14],
        "stop_loss_pct": 0.3,
        "position_size_percentage": 100,
    },
    ohlcv=df,
)
```

### 3. Run + analyze

```python
from quant_research.backtest import BacktestAnalysis

strategy.run_backtest(initial_balance=1000, leverage=1,
                      open_fee_rate=0.0002, close_fee_rate=0.0006)

results = BacktestAnalysis(strategy)
results.print_metrics()      # Sharpe/Sortino/Calmar, profit factor, drawdowns
results.plot_equity()        # altair
results.plot_drawdown()
results.plot_monthly_performance(year="all")
```

### 4. Extend with your own strategy

Subclass `quant_research.strategies.base.BaseStrategy` and implement
`populate_indicators`, `populate_long_signals`, `populate_short_signals`,
`evaluate_orders(time, row)`. See
[`strategies/envelope.py`](api/strategies.md) for a full example.

---

## Path B — Vectorized ML-pipeline PnL

Trains a model on log returns and converts predictions into trade-level results using Polars expressions (no per-bar state).

### 1. Load historical tick data

```python
from quant_research.connectors import BinanceConnector

conn = BinanceConnector()
conn.download_date_range("BTCUSDT", start_date="2023-01-01", end_date="2023-12-31")
```

Ticks land in `data/cache/BTCUSDT-trades-YYYY-MM-DD.parquet`.

### 2. Aggregate + add features

```python
from quant_research.engineering import load_ohlc_timeseries, add_log_return_features

df = load_ohlc_timeseries("BTCUSDT", "1h")
df = add_log_return_features(df, col="close", forecast_horizon=1, max_no_lags=5)

features = [f"close_log_return_lag_{i}" for i in range(1, 6)]
target = "close_log_return"
```

### 3. Train a model, get trade results

```python
from quant_research.utils import set_seed
from quant_research.models import LinearModel
from quant_research.backtest import learn_model_trades

set_seed(42)
model = LinearModel(len(features))
trades = learn_model_trades(
    df, features=features, target=target, model=model,
    test_size=0.25, optimizer_type="lbfgs",
)
```

`trades` is a Polars frame with `y_pred`, `y_true`, `is_won`, `position`,
`trade_log_return`, `equity_curve`, `drawdown_log_return`.

### 4. Apply fees + leverage

```python
from quant_research.backtest import add_compounding_trades

# Requires 'open', 'dir_signal', 'cum_trade_log_return' columns
trades = add_compounding_trades(
    trades, capital=10_000, leverage=1.0,
    maker_fee=0.0001, taker_fee=0.0003,
)
```

### 5. Annualized metrics

```python
from quant_research.backtest import sharpe_annualization_factor, eval_model_performance

ann = sharpe_annualization_factor("1h")
perf = eval_model_performance(
    y_actual=trades["y_true"], y_pred=trades["y_pred"],
    feature_names=features, target_name=target, annualized_rate=ann,
)
```

---

## Path C — Bring your own candles

Neither connector quite fits? Feed any polars OHLCV frame into the event-driven path. Required columns: `datetime`, `open`, `high`, `low`, `close` (plus `volume` optional).

```python
import polars as pl
df = pl.read_parquet("my_data.parquet").sort("datetime")
strategy = EnvelopeStrategy(params={...}, ohlcv=df)
```

## Next steps

- [`API Reference → Backtest`](api/backtest.md) — vectorized + event-driven functions.
- [`API Reference → Strategies`](api/strategies.md) — `EnvelopeStrategy`, `SimpleSMAStrategy`, indicators.
- Runnable notebooks in `accelerator/02_classical_strategies/` (pandas momentum / mean reversion), `accelerator/03_ml_strategy/` (vectorized ML) and `accelerator/04_event_driven_strategies/` (envelope/SMA).
