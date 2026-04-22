# 03 — Envelope & Simple SMA Strategies

Event-driven, class-based strategies on Polars OHLCV bars. Built on
`quant_research.strategies` and `quant_research.backtest.BacktestAnalysis`.

## Notebooks

| # | File | What it does |
|---|------|--------------|
| 1 | `data_engine.ipynb` | Download candles via `CCXTLoader`, cache as parquet |
| 2 | `run_envelope.ipynb` | Backtest `EnvelopeStrategy` (multi-band mean reversion) |
| 3 | `run_sma.ipynb` | Backtest `SimpleSMAStrategy` (triple-SMA trend following) |

## Run order

1. Install deps at repo root (`pip install -e .`) — pulls in `ccxt`, polars, altair.
2. Open `data_engine.ipynb`, pick exchange + pairs + timeframe, run all cells.
   Populates `data/cache/ccxt/<exchange>/<timeframe>/<symbol>.parquet`.
3. Open `run_envelope.ipynb` or `run_sma.ipynb`, tweak `strategy_params`, run all.

## Key concepts

- **Event-driven** — bar-by-bar state machine. Different from
  `accelerator/02_ml_strategy/` which is vectorized ML-signal PnL.
- **Position** model lives in `src/quant_research/backtest/event_driven/position.py`
  and handles fees, leverage, SL/TP, approximated liquidation.
- **BacktestAnalysis** computes Sharpe/Sortino/Calmar, drawdown, profit
  factor, hodl benchmark, monthly bars, and an Altair candlestick chart
  with trade markers.

## Strategy params (quick reference)

### EnvelopeStrategy

```python
{
    "average_type": "SMA",   # 'SMA' | 'EMA' | 'WMA' | 'DCM'
    "average_period": 6,
    "envelopes": [0.07, 0.11, 0.14],
    "stop_loss_pct": 0.3,
    # "price_jump_pct": 0.3,                # optional gap-open exit
    "position_size_percentage": 100,        # or "position_size_fixed_amount"
    # "mode": "long" | "short" | "both"     # default "both"
}
```

### SimpleSMAStrategy

```python
{
    "fast_ma_period": 100,
    "slow_ma_period": 200,
    "trend_ma_period": 300,
    "position_size_percentage": 100,        # or "position_size_fixed_amount"
    # "position_size_exposure": 2,          # risk-based (% of wallet at SL)
    # "mode": "long" | "short" | "both"
}
```

## Extending

Subclass `quant_research.strategies.base.BaseStrategy` and implement:

- `populate_indicators`
- `populate_long_signals` / `populate_short_signals`
- `evaluate_orders(time, row)`

Then call `run_backtest()` followed by `BacktestAnalysis(strategy)`.
