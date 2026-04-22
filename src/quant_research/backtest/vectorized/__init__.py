"""
Vectorized ML-pipeline backtest
================================
Functions that turn model predictions into trade-level PnL and equity curves,
operating on Polars DataFrames end to end (no per-bar state machine).
"""

from .performance import (
    sharpe_annualization_factor,
    model_trade_results,
    eval_model_performance,
)

from .engine import (
    learn_model_trades,
    learn_model_trade_pnl,
    add_tx_fee,
    add_tx_fees,
    add_tx_fees_log,
    add_trade_log_returns,
    add_equity_curve,
    add_compounding_trades,
)

__all__ = [
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
]
