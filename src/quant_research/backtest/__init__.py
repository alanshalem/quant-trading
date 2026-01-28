"""
backtest - Backtesting Module
==============================
Functions for simulating trades and calculating performance metrics.
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
    # performance.py
    'sharpe_annualization_factor',
    'model_trade_results',
    'eval_model_performance',
    # engine.py
    'learn_model_trades',
    'learn_model_trade_pnl',
    'add_tx_fee',
    'add_tx_fees',
    'add_tx_fees_log',
    'add_trade_log_returns',
    'add_equity_curve',
    'add_compounding_trades',
]
