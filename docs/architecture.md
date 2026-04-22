# Architecture

Two complementary backtest paradigms coexist inside `quant_research`. Each leans on the same data engineering and connector layers but diverges at trade generation.

## Module map

```
quant_research/
├── connectors/           Exchange data layer — two tiers:
│   ├── {binance,bybit,coinbase,kraken,okx}.py   tick-level trades (public dumps)
│   └── ccxt_loader.py                           OHLCV candles via CCXT (any exchange)
│
├── engineering/          OHLC aggregation, log-return + lag features, correlations
│
├── models/               PyTorch: LinearModel, NonLinearModel, DeepModel, LSTMModel,
│                         AttentionModel + trainer, validation, inspection, inference
│
├── backtest/
│   ├── vectorized/       ML-pipeline PnL — predictions become trade log-returns,
│   │                     fees, equity curves via Polars expressions
│   └── event_driven/     Bar-by-bar state machine — Position, SL/TP/liquidation,
│                         BacktestAnalysis (Sharpe, Sortino, Calmar, drawdowns, plots)
│
├── strategies/           Ready-made event-driven strategies (envelope, sma) +
│                         indicators (sma, ema, wma, donchian) in pure Polars
│
├── utils/                set_seed, to_tensor, altair/matplotlib plot helpers
└── config.py             SEED, TRADING_DAYS_PER_YEAR (365 for crypto), DEFAULT_EPOCHS, ...
```

## The two backtest paradigms

|                       | Vectorized (`backtest.vectorized`)          | Event-driven (`backtest.event_driven`)      |
|-----------------------|---------------------------------------------|---------------------------------------------|
| **Input shape**        | One row per bar with features + target     | One row per bar (OHLCV)                     |
| **Signal source**      | Model prediction (`y_hat`)                  | Rules coded in a `BaseStrategy` subclass    |
| **Engine**             | Polars `with_columns` chain; no Python loop | Python `for row in df.iter_rows(...)`       |
| **Position model**     | Implicit (sign of `y_hat` → +1/-1 long/short) | Explicit `Position` object with SL/TP/liq |
| **Fees**               | Maker + taker applied in batch after PnL   | Open fee + close fee applied per trade     |
| **Scale-in / pyramid** | Not supported                               | Yes (`Position.add`)                        |
| **Liquidation**        | Not modelled                                | Approximated (`price * (1 ± 1/leverage)`)  |
| **Sharpe / metrics**   | `eval_model_performance`, annualization factor | Full `BacktestAnalysis` metric dict + plots |
| **When to use**        | Rapid ML iteration; prediction-driven strats | Discretionary rules; multi-band entries; SL/TP/liq concerns |

Both publish through `from quant_research.backtest import ...` — the submodule `__init__.py` re-exports keep paths short.

## Typical data flow

### Vectorized path

```
connector.download_*   →   load_ohlc_timeseries   →   add_log_return_features / add_lags
                                                                       ↓
                                                          LinearModel / DeepModel / ...
                                                                       ↓
                                                    learn_model_trades (split, train, predict)
                                                                       ↓
                                       add_compounding_trades  or  add_tx_fees + add_equity_curve
                                                                       ↓
                                      sharpe_annualization_factor,  eval_model_performance
```

### Event-driven path

```
CCXTLoader.download   →   CCXTLoader.load   →   EnvelopeStrategy(params, ohlcv)
                                                            ↓
                                           strategy.run_backtest(initial_balance, leverage, fees)
                                                            ↓
                                                    BacktestAnalysis(strategy)
                                                            ↓
                       print_metrics  /  plot_equity  /  plot_drawdown  /  plot_candlestick
```

## Why two paradigms?

**Vectorized** is fast to iterate on for ML research: train a model, eyeball cumulative log returns, swap features, repeat. It assumes constant holding period (one bar), uniform position sizing, no entry/exit rules beyond the sign of a prediction.

**Event-driven** is honest about execution reality: stop losses, take profits, pyramid-in bands, liquidation, gap-open circuit breakers, separate maker/taker fees, variable position duration. The cost is a Python `for`-loop over bars — slower per run, but orders of magnitude richer per trade.

Most research starts vectorized and graduates to event-driven once a signal looks promising.

## Data directories

```
data/
├── cache/
│   ├── BTCUSDT-trades-YYYY-MM-DD.parquet      tick trades (Binance, Bybit, ...)
│   └── ccxt/<exchange>/<timeframe>/*.parquet  OHLCV candles (CCXTLoader)
└── models/                                     saved model weights
```

All gitignored.

## Accelerator mapping

| Dir | Paradigm |
|-----|----------|
| `accelerator/01_fundamentals/`              | Python + numpy + stats warmups, paradigm-agnostic |
| `accelerator/02_ml_strategy/`               | **Vectorized** — log returns, PyTorch, `learn_model_trades` |
| `accelerator/03_event_driven_strategies/`   | **Event-driven** — `CCXTLoader`, `EnvelopeStrategy`, `BacktestAnalysis` |

## Sharpe annualization

Crypto trades 24/7/365 → `TRADING_DAYS_PER_YEAR = 365`, `TRADING_HOURS_PER_DAY = 24`. For US equities override at the call site with `sharpe_annualization_factor(interval, 252, 6.5)`.
