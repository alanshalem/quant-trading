"""
engineering - Data Engineering Module
======================================
Functions for loading, processing, and analyzing time series data.
"""

from .loaders import (
    OHLC_AGGS,
    get_trade_files,
    load_timeseries,
    load_ohlc_timeseries,
    load_timeseries_range,
    load_ohlc_timeseries_range,
    _get_file_hash,
    _read_parquet_cached,
    _load_and_aggregate_file,
    _load_and_aggregate_date_file,
)

from .processors import (
    timeseries,
    ohlc_timeseries,
    lag_col_names,
    log_returns_col,
    log_return_col,
    log_return,
    lag_cols,
    add_lags,
    add_log_return_features,
    _prefix_cols,
    _prefix_close_ts,
)

from .analysis import (
    auto_reg_corr_matrx,
    compare_ts_corr,
)

__all__ = [
    # loaders.py
    'OHLC_AGGS',
    'get_trade_files',
    'load_timeseries',
    'load_ohlc_timeseries',
    'load_timeseries_range',
    'load_ohlc_timeseries_range',
    '_get_file_hash',
    '_read_parquet_cached',
    '_load_and_aggregate_file',
    '_load_and_aggregate_date_file',
    # processors.py
    'timeseries',
    'ohlc_timeseries',
    'lag_col_names',
    'log_returns_col',
    'log_return_col',
    'log_return',
    'lag_cols',
    'add_lags',
    'add_log_return_features',
    '_prefix_cols',
    '_prefix_close_ts',
    # analysis.py
    'auto_reg_corr_matrx',
    'compare_ts_corr',
]
