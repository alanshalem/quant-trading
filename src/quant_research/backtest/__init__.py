"""
backtest - Backtest module
===========================
Two paradigms, sibling submodules:

- :mod:`quant_research.backtest.vectorized` — ML-pipeline PnL: model predictions
  turn into trade log-returns, fees, equity curves via Polars expressions.
- :mod:`quant_research.backtest.event_driven` — bar-by-bar state machine:
  ``Position`` primitives, SL/TP/liquidation, and ``BacktestAnalysis`` metrics.

Public symbols from both submodules are re-exported here so callers can keep
the short ``from quant_research.backtest import Position`` form.
"""

from .vectorized import (
    sharpe_annualization_factor,
    model_trade_results,
    eval_model_performance,
    learn_model_trades,
    learn_model_trade_pnl,
    add_tx_fee,
    add_tx_fees,
    add_tx_fees_log,
    add_trade_log_returns,
    add_equity_curve,
    add_compounding_trades,
)

from .event_driven import (
    Position,
    PositionBehavior,
    LongPositionBehavior,
    ShortPositionBehavior,
    update_equity_record,
    BacktestAnalysis,
)

__all__ = [
    # vectorized
    "sharpe_annualization_factor",
    "model_trade_results",
    "eval_model_performance",
    "learn_model_trades",
    "learn_model_trade_pnl",
    "add_tx_fee",
    "add_tx_fees",
    "add_tx_fees_log",
    "add_trade_log_returns",
    "add_equity_curve",
    "add_compounding_trades",
    # event_driven
    "Position",
    "PositionBehavior",
    "LongPositionBehavior",
    "ShortPositionBehavior",
    "update_equity_record",
    "BacktestAnalysis",
]
