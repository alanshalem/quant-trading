"""
quant_research - Quantitative Trading Research Library
=======================================================
A comprehensive library for quantitative trading research and model development.

Modules:
--------
- config: Shared configuration and constants
- utils: Common utilities (reproducibility, tensor conversion, plotting)
- engineering: Data loading, processing, and analysis
- models: ML training, validation, inspection, and inference
- backtest: Trade simulation and performance metrics

Usage:
------
    # Import specific functions
    from quant_research.engineering import load_ohlc_timeseries, add_lags
    from quant_research.models import train_reg_model, benchmark_linear_models
    from quant_research.backtest import eval_model_performance
    from quant_research.utils import set_seed, plot_dyn_timeseries

    # Or import entire modules
    from quant_research import engineering, models, backtest, utils

Author: MemLabs
Course: Build a Quant Trading System
"""

# Configuration
from .config import (
    SEED,
    IS_WINDOWS,
    DEFAULT_PARALLEL,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LBFGS_LR,
    DEFAULT_EPOCHS,
    DEFAULT_TEST_SIZE,
    TRADING_DAYS_PER_YEAR,
    TRADING_HOURS_PER_DAY,
    LOG_INTERVAL_DIVISOR,
)

# Utils - Common utilities
from .utils import (
    set_seed,
    to_tensor,
    init_weights,
    plot,
    plot_distribution,
    plot_static_timeseries,
    plot_multiple_lines,
    plot_dyn_timeseries,
    plot_column,
)

# Engineering - Data loading and processing
from .engineering import (
    OHLC_AGGS,
    get_trade_files,
    load_timeseries,
    load_ohlc_timeseries,
    load_timeseries_range,
    load_ohlc_timeseries_range,
    timeseries,
    ohlc_timeseries,
    lag_col_names,
    log_returns_col,
    log_return_col,
    log_return,
    lag_cols,
    add_lags,
    add_log_return_features,
    auto_reg_corr_matrx,
    compare_ts_corr,
)

# Models - ML training and validation
from .models import (
    timeseries_split,
    timeseries_train_test_split,
    total_model_params,
    print_model_info,
    print_model_complexity_ratio,
    get_linear_params,
    print_model_params,
    batch_train_reg,
    train_reg_model,
    benchmark_reg_model,
    benchmark_linear_models,
    add_model_predictions,
    # Model architectures
    LinearModel,
    NonLinearModel,
    DeepModel,
    LSTMModel,
    AttentionModel,
)

# Backtest - Performance and simulation
from .backtest import (
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

__version__ = "1.0.0"
__author__ = "MemLabs"

__all__ = [
    # Config
    'SEED',
    'IS_WINDOWS',
    'DEFAULT_PARALLEL',
    'DEFAULT_LEARNING_RATE',
    'DEFAULT_LBFGS_LR',
    'DEFAULT_EPOCHS',
    'DEFAULT_TEST_SIZE',
    'TRADING_DAYS_PER_YEAR',
    'TRADING_HOURS_PER_DAY',
    'LOG_INTERVAL_DIVISOR',
    # Utils
    'set_seed',
    'to_tensor',
    'init_weights',
    'plot',
    'plot_distribution',
    'plot_static_timeseries',
    'plot_multiple_lines',
    'plot_dyn_timeseries',
    'plot_column',
    # Engineering
    'OHLC_AGGS',
    'get_trade_files',
    'load_timeseries',
    'load_ohlc_timeseries',
    'load_timeseries_range',
    'load_ohlc_timeseries_range',
    'timeseries',
    'ohlc_timeseries',
    'lag_col_names',
    'log_returns_col',
    'log_return_col',
    'log_return',
    'lag_cols',
    'add_lags',
    'add_log_return_features',
    'auto_reg_corr_matrx',
    'compare_ts_corr',
    # Models
    'timeseries_split',
    'timeseries_train_test_split',
    'total_model_params',
    'print_model_info',
    'print_model_complexity_ratio',
    'get_linear_params',
    'print_model_params',
    'batch_train_reg',
    'train_reg_model',
    'benchmark_reg_model',
    'benchmark_linear_models',
    'add_model_predictions',
    # Model architectures
    'LinearModel',
    'NonLinearModel',
    'DeepModel',
    'LSTMModel',
    'AttentionModel',
    # Backtest
    'sharpe_annualization_factor',
    'model_trade_results',
    'eval_model_performance',
    'learn_model_trades',
    'learn_model_trade_pnl',
    'add_tx_fee',
    'add_tx_fees',
    'add_tx_fees_log',
    'add_trade_log_returns',
    'add_equity_curve',
    'add_compounding_trades',
]
