"""
processors.py - Data Transformations
=====================================
Functions for transforming and processing time series data using Polars.
"""

from typing import List, Union, Optional
import polars as pl

from .loaders import OHLC_AGGS


def timeseries(
    df: pl.DataFrame,
    time_interval: str,
    aggs: Union[List[pl.Expr], pl.Expr]
) -> pl.DataFrame:
    """
    Generic function for aggregating data into regular time intervals.

    This is a flexible time series aggregation framework that can handle
    any custom aggregation expressions. Used as the foundation for OHLC
    bars and other time-based features.

    Args:
        df: DataFrame with a 'datetime' column
        time_interval: Aggregation period (Polars duration string)
            Examples: '1m', '5m', '15m', '1h', '4h', '1d'
        aggs: List of Polars expressions defining aggregations to compute

    Returns:
        DataFrame with time-aggregated data

    Technical Details:
        - Uses left-closed intervals: [start_time, end_time)
        - Bars start at round times (e.g., 09:00, 09:15, 09:30)
        - Missing bars (no trades) are automatically excluded

    Example:
        >>> # Custom aggregation for volatility analysis
        >>> custom_aggs = [
        ...     pl.col("price").std().alias("price_volatility"),
        ...     pl.col("volume").sum().alias("total_volume"),
        ... ]
        >>> df_volatility = timeseries(df, '1h', custom_aggs)
    """
    return df.group_by_dynamic(
        "datetime",              # Column to group by (must be datetime type)
        every=time_interval,     # Aggregation frequency
        offset="0m"              # No offset (bars align to round times)
    ).agg(aggs)


def ohlc_timeseries(df: pl.DataFrame, time_interval: str) -> pl.DataFrame:
    """
    Convert tick-level trade data into OHLC (Open, High, Low, Close) bars.

    This function aggregates raw trade data into standardized price bars
    with basic volume and trade statistics.

    Args:
        df: DataFrame containing trade data with columns:
            - datetime: Timestamp of each trade
            - price: Execution price
            - quote_qty: Trade size in quote currency (e.g., USDT)
            - is_short: Boolean indicating if trade was a short sale
        time_interval: Aggregation period (e.g., '1m', '5m', '15m', '1h', '1d')

    Returns:
        DataFrame with OHLC bars containing:
            - datetime: Bar timestamp
            - open: First price in interval
            - high: Highest price in interval
            - low: Lowest price in interval
            - close: Last price in interval (most important for ML)

    Example:
        >>> # Create 15-minute OHLC bars
        >>> bars_15m = ohlc_timeseries(trades_df, '15m')
        >>>
        >>> # Create hourly bars for longer-term analysis
        >>> bars_1h = ohlc_timeseries(trades_df, '1h')
    """
    return timeseries(df, time_interval, OHLC_AGGS)


def lag_col_names(col: str, n: int) -> List[str]:
    """Generate lag column names (e.g., ['col_lag_1', 'col_lag_2', ...])."""
    return [f'{col}_lag_{i}' for i in range(1, n+1)]


def log_returns_col(name: str, step_size: int = 1) -> pl.Expr:
    """Create Polars expression for log returns calculation."""
    return (pl.col(name)/pl.col(name).shift(step_size)).log().alias(f'{name}_log_return')


def log_return_col(col: str) -> str:
    """Generate log return column name from base column name."""
    return f"{col}_log_return"


def log_return(col: str, shift_size: int = 1) -> pl.Expr:
    """Create Polars expression for log return calculation."""
    return (pl.col(col)/pl.col(col).shift(shift_size)).log().alias(log_return_col(col))


def lag_cols(col: str, forecast_horizon: int, no_lags: int) -> List[pl.Expr]:
    """Generate list of lag column expressions."""
    return [pl.col(col).shift(forecast_horizon * i).alias(f'{col}_lag_{i}') for i in range(1, no_lags + 1)]


def add_lags(df: pl.DataFrame, col: str, max_n_lags: int, forecast_step: int) -> pl.DataFrame:
    """Add lag columns to DataFrame."""
    return df.with_columns([pl.col(col).shift(i * forecast_step).alias(f'{col}_lag_{i}') for i in range(1, max_n_lags + 1)])


def add_log_return_features(df: pl.DataFrame, col: str, forecast_horizon: int, max_no_lags: Optional[int] = None) -> pl.DataFrame:
    """Add log return and lag features to DataFrame."""
    if max_no_lags is None:
        max_no_lags = 0
    df = df.with_columns(log_return(col, forecast_horizon))
    if max_no_lags > 0:
        df = add_lags(df, log_return_col('close'), max_no_lags, forecast_horizon)
    return df


def _prefix_cols(df: pl.DataFrame, prefix: str) -> pl.DataFrame:
    """Prefix all column names in a DataFrame."""
    return df.rename({col: f"{prefix}_{col}" for col in df.columns})


def _prefix_close_ts(trades: pl.DataFrame, time_interval: str, prefix: str) -> pl.DataFrame:
    """Convert trades to OHLC timeseries and prefix column names."""
    return _prefix_cols(ohlc_timeseries(trades, time_interval), prefix)
