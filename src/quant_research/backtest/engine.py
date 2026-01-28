"""
engine.py - Backtest Engine
============================
Functions for simulating trades and calculating PnL.
"""

from typing import List, Union
import numpy as np
import numpy.typing as npt
import polars as pl
import torch.nn as nn

from ..config import DEFAULT_EPOCHS
from ..models.validation import timeseries_split, _prepare_train_test_tensors
from ..models.trainer import batch_train_reg
from .performance import model_trade_results


def learn_model_trades(df: pl.DataFrame, features: List[str], target: str, model: nn.Module, test_size=0.25, loss=None, optimizer=None, optimizer_type: str = 'lbfgs', no_epochs=None, log=False, lr=None):
    """
    Train a model and return trade-level results DataFrame.

    This function trains a model and generates a detailed trade-by-trade
    analysis including predictions, actual returns, win/loss, and equity curves.

    Args:
        df: DataFrame with features and target columns
        features: List of feature column names
        target: Target column name
        model: PyTorch model to train
        test_size: Test set proportion (default: 0.25)
        loss: Optional loss function (default: L1Loss)
        optimizer: Optional optimizer (overrides optimizer_type)
        optimizer_type: Type of optimizer - 'lbfgs' or 'adam' (default: 'lbfgs')
        no_epochs: Training epochs (default: DEFAULT_EPOCHS)
        log: Whether to log training progress (default: False)
        lr: Learning rate (default: depends on optimizer_type)

    Returns:
        DataFrame with columns:
        - y_pred: Model predictions
        - y_true: Actual returns
        - is_won: Whether trade was profitable
        - position: Direction (+1 long, -1 short)
        - trade_log_return: Return on trade
        - equity_curve: Cumulative equity
        - drawdown_log_return: Drawdown at each point

    Example:
        >>> model = LinearModel(3)
        >>> trades = learn_model_trades(
        ...     df, ['lag_1', 'lag_2', 'lag_3'], 'close_log_return', model
        ... )
        >>> print(trades[['y_pred', 'y_true', 'is_won']].head())
    """
    df = df.drop_nulls()
    df_train, df_test = timeseries_split(df, test_size=test_size)
    if no_epochs is None:
        no_epochs = DEFAULT_EPOCHS

    X_train, X_test, y_train, y_test = _prepare_train_test_tensors(df_train, df_test, features, target)

    y_hat = batch_train_reg(model, X_train, X_test, y_train, y_test, no_epochs, criterion=loss, optimizer=optimizer, optimizer_type=optimizer_type, lr=lr, logging=log)

    return model_trade_results(y_test, y_hat)


def learn_model_trade_pnl(df: pl.DataFrame, features: List[str], target: str, model: nn.Module, test_size=0.25, loss=None, optimizer=None, optimizer_type: str = 'lbfgs', no_epochs=None, log=False, lr=None):
    """
    Train a model and return trade-level PnL results.

    Similar to learn_model_trades but specifically focused on PnL analysis.
    This function trains a model and returns detailed trade results for
    profit/loss analysis.

    Args:
        df: DataFrame with features and target columns
        features: List of feature column names
        target: Target column name
        model: PyTorch model to train
        test_size: Test set proportion (default: 0.25)
        loss: Optional loss function (default: L1Loss)
        optimizer: Optional optimizer (overrides optimizer_type)
        optimizer_type: Type of optimizer - 'lbfgs' or 'adam' (default: 'lbfgs')
        no_epochs: Training epochs (default: DEFAULT_EPOCHS)
        log: Whether to log training progress (default: False)
        lr: Learning rate (default: depends on optimizer_type)

    Returns:
        DataFrame with trade-level results including PnL metrics

    Example:
        >>> model = LinearModel(3)
        >>> pnl_results = learn_model_trade_pnl(
        ...     df, ['lag_1', 'lag_2', 'lag_3'], 'close_log_return', model
        ... )
        >>> total_pnl = pnl_results['trade_log_return'].sum()
    """
    df_train, df_test = timeseries_split(df, test_size=test_size)
    if no_epochs is None:
        no_epochs = DEFAULT_EPOCHS

    X_train, X_test, y_train, y_test = _prepare_train_test_tensors(df_train, df_test, features, target)

    y_hat = batch_train_reg(model, X_train, X_test, y_train, y_test, no_epochs, loss, optimizer, optimizer_type, lr=lr, logging=log)

    trade_results = model_trade_results(y_test, y_hat)
    return trade_results


def add_tx_fee(trades: pl.DataFrame, tx_fee: float, name: str) -> pl.DataFrame:
    """Add transaction fee column to trades DataFrame."""
    tx_fee_col = (pl.col('exit_trade_value') * tx_fee + pl.col('entry_trade_value') * tx_fee).alias(f"tx_fee_{name}")
    return trades.with_columns(tx_fee_col)


def add_tx_fees(trades: pl.DataFrame, maker_fee: float, taker_fee: float) -> pl.DataFrame:
    """Add maker and taker transaction fee columns to trades DataFrame."""
    trades = add_tx_fee(trades, maker_fee, 'maker')
    trades = add_tx_fee(trades, taker_fee, 'taker')
    return trades


def add_tx_fees_log(trades: pl.DataFrame, maker_fee: float, taker_fee: float) -> pl.DataFrame:
    """Add transaction fees in log space for log returns."""
    return trades.with_columns(
        (pl.col('trade_log_return') + np.log(maker_fee)).alias('trade_log_return_net_maker'),
        (pl.col('trade_log_return') + np.log(taker_fee)).alias('trade_log_return_net_taker'),
    ).with_columns(
        pl.col('trade_log_return_net_maker').cum_sum().alias('equity_curve_net_maker'),
        pl.col('trade_log_return_net_taker').cum_sum().alias('equity_curve_net_taker'),
    )


def add_trade_log_returns(trades: pl.DataFrame, pre_trade_values: Union[List[float], npt.NDArray[np.float32]], tx_fee: float, initial_capital: float) -> pl.DataFrame:
    """
    Calculate trade-level returns, PnL, and equity curves with transaction fees.

    This function transforms model predictions into a complete trading simulation
    including position sizing, transaction costs, and cumulative performance.

    Args:
        trades: DataFrame with 'y_hat' (predictions), 'close_log_return', 'open' columns
        pre_trade_values: Array of capital allocated to each trade
        tx_fee: Transaction fee rate (e.g., 0.0003 for 0.03%)
        initial_capital: Starting capital amount

    Returns:
        DataFrame with added columns:
        - dir_signal: +1 for long, -1 for short
        - trade_log_return: Return on each trade
        - signed_trade_qty: Position size (signed)
        - trade_gross_pnl: PnL before fees
        - trade_net_pnl: PnL after fees
        - equity_curve_gross: Cumulative wealth (before fees)
        - equity_curve_net: Cumulative wealth (after fees)

    Example:
        >>> pre_trade_vals = np.full(len(df), 10000.0)  # $10k per trade
        >>> df_trades = add_trade_log_returns(
        ...     df, pre_trade_vals, tx_fee=0.0003, initial_capital=10000
        ... )
        >>> print(df_trades['equity_curve_net'].iloc[-1])  # Final equity

    Note:
        - Transaction fees are applied on both entry and exit
        - Log returns are used for mathematical convenience
        - Equity curves show cumulative performance over time
    """
    # add directional signal to indicate if we're going long or short
    trades = trades.with_columns(pl.col('y_hat').sign().alias('dir_signal'))
    # calculate trade log return
    trades = trades.with_columns((pl.col('close_log_return') * pl.col('dir_signal')).alias('trade_log_return'))
    # calculate the cumulative sum of the trade log returns - this is the equity curves in log space
    trades = trades.with_columns(pl.col('trade_log_return').cum_sum().alias('cum_trade_log_return'))
    trades = trades.with_columns(
        # add pre trade values
        pre_trade_values.alias('pre_trade_value'),
        # add post trade values
        (pre_trade_values * pl.col('trade_log_return').exp()).alias('post_trade_value'),
        # add trade qty
        (pre_trade_values / pl.col('open')).alias('trade_qty'),
    )

    trades = trades.with_columns(
        # add signed trade quantities (the main output of our strategy)
        (pl.col('trade_qty') * pl.col('dir_signal')).alias('signed_trade_qty'),
        # add trade gross pnl
        (pl.col('post_trade_value') - pl.col('pre_trade_value')).alias('trade_gross_pnl'),
        # add tx fees
        (pl.col('pre_trade_value') * tx_fee + pl.col('post_trade_value') * tx_fee).alias('tx_fees')
    )
    trades = trades.with_columns(
        # calculate each trade's profit after fees (net)
        (pl.col('trade_gross_pnl')-pl.col('tx_fees')).alias('trade_net_pnl')
    )
    trades = trades.with_columns(
        # calculate equity curve for gross profit
        (initial_capital + pl.col('trade_gross_pnl').cum_sum()).alias('equity_curve_gross'),
        # calculate equity curve for net profit
        (initial_capital + pl.col('trade_net_pnl').cum_sum()).alias('equity_curve_net')
    )


def add_equity_curve(trades: pl.DataFrame, initial_capital: float, col_name: str, suffix: str) -> pl.DataFrame:
    """Add equity curve column to trades DataFrame."""
    return trades.with_columns(
        (initial_capital + pl.col(col_name).cum_sum()).alias(f'equity_curve_{suffix}')
    )


def add_compounding_trades(trades: pl.DataFrame, capital: float, leverage: float, maker_fee: float, taker_fee: float) -> pl.DataFrame:
    """
    Calculate compounding trade returns with leverage and transaction fees.

    This function implements a compounding strategy where profits are reinvested,
    allowing the position size to grow (or shrink) with account performance.
    Includes leverage and separate maker/taker fee calculations.

    Args:
        trades: DataFrame with 'cum_trade_log_return', 'open', 'dir_signal' columns
        capital: Initial capital amount
        leverage: Leverage multiplier (e.g., 2.0 for 2x leverage)
        maker_fee: Maker fee rate (e.g., 0.0001 for 0.01%)
        taker_fee: Taker fee rate (e.g., 0.0003 for 0.03%)

    Returns:
        DataFrame with added columns:
        - entry_trade_value: Capital allocated at trade entry
        - exit_trade_value: Capital value at trade exit
        - signed_trade_qty: Position size with direction
        - trade_gross_pnl: PnL before fees
        - tx_fee_maker/tx_fee_taker: Fee amounts
        - trade_net_maker_pnl/trade_net_taker_pnl: PnL after fees
        - equity_curve_gross/taker/maker: Cumulative performance curves

    Example:
        >>> # Backtest with 2x leverage and compounding
        >>> trades_with_compounding = add_compounding_trades(
        ...     trades, capital=10000, leverage=2.0,
        ...     maker_fee=0.0001, taker_fee=0.0003
        ... )
        >>> final_equity = trades_with_compounding['equity_curve_taker'].iloc[-1]
        >>> print(f"Final equity: ${final_equity:,.2f}")

    Note:
        - Compounding amplifies both gains and losses
        - Leverage multiplies both returns and risk
        - Maker fees apply when providing liquidity, taker when taking
        - Equity curves show performance under different fee scenarios
    """
    lev_capital = capital * leverage
    # calculate entry and exit trade value and size
    trades = trades.with_columns(
        ((pl.col('cum_trade_log_return').exp()) * lev_capital).shift().fill_null(lev_capital).alias('entry_trade_value'),
        ((pl.col('cum_trade_log_return').exp()) * lev_capital).alias('exit_trade_value'),
    ).with_columns(
        (pl.col('entry_trade_value') / pl.col('open') * pl.col('dir_signal')).alias('signed_trade_qty'),
        (pl.col('exit_trade_value')-pl.col('entry_trade_value')).alias('trade_gross_pnl'),
    )
    # add transaction fee
    trades = add_tx_fees(trades, maker_fee, taker_fee)
    # add net trade pnl
    trades = trades.with_columns(
        (pl.col('trade_gross_pnl') - pl.col('tx_fee_taker')).alias('trade_net_taker_pnl'),
        (pl.col('trade_gross_pnl') - pl.col('tx_fee_maker')).alias('trade_net_maker_pnl'),
    )
    trades = add_equity_curve(trades, capital, 'trade_gross_pnl', 'gross')
    # add net equity curves (both taker and maker)
    trades = add_equity_curve(trades, capital, 'trade_net_taker_pnl', 'taker')
    trades = add_equity_curve(trades, capital, 'trade_net_maker_pnl', 'maker')
    return trades
