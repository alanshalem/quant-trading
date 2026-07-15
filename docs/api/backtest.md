# Backtest

Two sibling paradigms:

- **Vectorized** (`quant_research.backtest.vectorized`) — ML-pipeline PnL. Model
  predictions become trade log-returns, fees, and equity curves via Polars
  expressions. No per-bar state machine.
- **Event-driven** (`quant_research.backtest.event_driven`) — bar-by-bar state
  machine. `Position` primitives, SL/TP/liquidation, and `BacktestAnalysis`
  metrics consumed by `quant_research.strategies`.

Both submodules re-export their public symbols from `quant_research.backtest`
so callers can use the short form (`from quant_research.backtest import Position`).

## Vectorized — Engine

::: quant_research.backtest.vectorized.engine

## Vectorized — Performance

::: quant_research.backtest.vectorized.performance

## Event-Driven — Position

::: quant_research.backtest.event_driven.position

## Event-Driven — Analysis

::: quant_research.backtest.event_driven.analysis
