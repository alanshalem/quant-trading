"""
performance.py - Performance Metrics
=====================================
Functions for calculating trading performance metrics.
"""

from typing import Any, Dict, List
import re
import numpy as np
import polars as pl

from ..config import TRADING_DAYS_PER_YEAR, TRADING_HOURS_PER_DAY


def sharpe_annualization_factor(interval: str,
                                trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
                                trading_hours_per_day: float = TRADING_HOURS_PER_DAY) -> float:
    """
    Compute annualization factor (sqrt of periods per year) given a return interval.

    interval : str
        Frequency string like '1d', '1h', '30m', '15s'.
    trading_days_per_year : int
        Number of trading days in a year (default 252).
    trading_hours_per_day : float
        Number of trading hours in a trading day (default 6.5).

    Returns
    -------
    float : annualization factor
    """
    match = re.match(r"(\d+)([dhms])", interval.lower())
    if not match:
        raise ValueError("Interval must be like '1d', '2h', '15m', '30s'")

    value, unit = int(match.group(1)), match.group(2)

    # periods per year
    if unit == 'd':
        periods = trading_days_per_year / value
    elif unit == 'h':
        periods = trading_days_per_year * (trading_hours_per_day / value)
    elif unit == 'm':
        periods = trading_days_per_year * (trading_hours_per_day * 60 / value)
    elif unit == 's':
        periods = trading_days_per_year * (trading_hours_per_day * 3600 / value)
    else:
        raise ValueError(f"Unsupported unit: {unit}")

    return np.sqrt(periods)


def model_trade_results(y_true, y_pred) -> pl.DataFrame:
    """Generate trade-level results from model predictions."""
    trade_results = pl.DataFrame({
        'y_pred': y_pred.squeeze(),
        'y_true': y_true.squeeze()
    }).with_columns([
        (pl.col('y_pred').sign() == pl.col('y_true').sign()).alias('is_won'),
        pl.col('y_pred').sign().alias('position')
    ]).with_columns([
        (pl.col('position') * pl.col('y_true')).alias('trade_log_return')
    ]).with_columns([
        pl.col('trade_log_return').cum_sum().alias('equity_curve')
    ]).with_columns(
        (pl.col('equity_curve')-pl.col('equity_curve').cum_max()).alias('drawdown_log_return'),
    )
    return trade_results


def eval_model_performance(y_actual, y_pred, feature_names: List[str], target_name: str, annualized_rate: float) -> Dict[str, Any]:
    """Calculate performance metrics for the trading model."""
    trade_results = model_trade_results(y_actual, y_pred)

    accuracy = trade_results['is_won'].mean()
    avg_win = trade_results.filter(pl.col('is_won'))['trade_log_return'].mean()
    avg_loss = trade_results.filter(~pl.col('is_won'))['trade_log_return'].mean()
    expected_value = accuracy * avg_win + (1 - accuracy) * avg_loss
    drawdown = (trade_results['equity_curve'] - trade_results['equity_curve'].cum_max())
    max_drawdown = drawdown.min()
    sharpe = trade_results['trade_log_return'].mean() / trade_results['trade_log_return'].std() if trade_results['trade_log_return'].std() > 0 else 0
    annualized_sharpe = sharpe * annualized_rate
    equity_trough = trade_results['equity_curve'].min()
    equity_peak = trade_results['equity_curve'].max()
    total_log_return = trade_results['trade_log_return'].sum()
    std = trade_results['trade_log_return'].std()
    return {
        'features': ','.join(list(feature_names)),
        'target': target_name,
        'no_trades': len(trade_results),
        'win_rate': accuracy,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'best_trade': trade_results['trade_log_return'].max(),
        'worst_trade': trade_results['trade_log_return'].min(),
        'ev': expected_value,
        'std': std,
        'total_log_return': total_log_return,
        'compound_return': np.exp(total_log_return),
        'max_drawdown': max_drawdown,
        'equity_trough': equity_trough,
        'equity_peak': equity_peak,
        'sharpe': annualized_sharpe,
    }
