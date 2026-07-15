"""
quant_research - Quantitative Trading Research Library
=======================================================

Two complementary backtest paradigms on Polars-native OHLCV data:

- **Event-driven** (torch-free): ``Position``, ``BacktestAnalysis``,
  ``EnvelopeStrategy``, ``SimpleSMAStrategy``, polars indicators.
- **Vectorized ML-pipeline** (pulls PyTorch): ``LinearModel`` …
  ``AttentionModel``, training utilities, ``learn_model_trades``,
  ``add_compounding_trades``, ``sharpe_annualization_factor``.

The ML symbols load lazily (PEP 562) so event-driven users don't pay
torch's ~200–500 ms import cost on startup.

Usage
-----
    # Event-driven — no torch loaded
    from quant_research.strategies import EnvelopeStrategy
    from quant_research.backtest import Position, BacktestAnalysis

    # Vectorized ML — torch loads on first access
    from quant_research.models import LinearModel
    from quant_research.backtest import learn_model_trades

Modules
-------
- :mod:`config`       Shared constants (SEED, TRADING_DAYS_PER_YEAR, …)
- :mod:`connectors`   Exchange data (raw trades + CCXT OHLCV)
- :mod:`engineering`  Polars loaders, OHLC bars, lag / log-return features
- :mod:`strategies`   Event-driven strategies + indicator expressions
- :mod:`backtest`     ``event_driven`` (eager) + ``vectorized`` (lazy)
- :mod:`models`       PyTorch architectures + trainer (lazy)
- :mod:`utils`        ``plotting`` (eager) + ``common`` reproducibility helpers (lazy)

Author: MemLabs
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# Connectors (submodule ref; concrete connectors self-import their deps)
from . import connectors

# Backtest event-driven (Position, BacktestAnalysis) — torch-free.
# Vectorized symbols are re-exported lazily by the backtest submodule itself.
from .backtest import (
    BacktestAnalysis,
    LongPositionBehavior,
    Position,
    PositionBehavior,
    ShortPositionBehavior,
    update_equity_record,
)

# --- Eager re-exports: torch-free, cheap to import ---
# Configuration
from .config import (
    DEFAULT_EPOCHS,
    DEFAULT_LBFGS_LR,
    DEFAULT_LEARNING_RATE,
    DEFAULT_PARALLEL,
    DEFAULT_TEST_SIZE,
    IS_WINDOWS,
    LOG_INTERVAL_DIVISOR,
    SEED,
    TRADING_DAYS_PER_YEAR,
    TRADING_HOURS_PER_DAY,
)

# Engineering - polars-only, no torch
from .engineering import (
    OHLC_AGGS,
    add_lags,
    add_log_return_features,
    auto_reg_corr_matrx,
    compare_ts_corr,
    get_trade_files,
    lag_col_names,
    lag_cols,
    load_ohlc_timeseries,
    load_ohlc_timeseries_range,
    load_timeseries,
    load_timeseries_range,
    log_return,
    log_return_col,
    log_returns_col,
    ohlc_timeseries,
    timeseries,
)

# Strategies - event-driven (polars only)
from .strategies import (
    AVERAGE_TYPES,
    BaseStrategy,
    EnvelopeStrategy,
    SimpleSMAStrategy,
    donchian_mid,
    ema,
    moving_average,
    sma,
    wma,
)

# Plotting helpers (altair + matplotlib only)
from .utils.plotting import (
    plot,
    plot_column,
    plot_distribution,
    plot_dyn_timeseries,
    plot_multiple_lines,
    plot_static_timeseries,
)

# --- Lazy re-exports (symbols whose modules pull PyTorch) ---
#
# Resolved on first attribute access via PEP 562 __getattr__, then cached
# in module globals so we only pay the import penalty once.
_LAZY_MODELS_SYMBOLS = frozenset({
    # Trainer / validation / inspection / inference
    "timeseries_split",
    "timeseries_train_test_split",
    "total_model_params",
    "print_model_info",
    "print_model_complexity_ratio",
    "get_linear_params",
    "print_model_params",
    "batch_train_reg",
    "train_reg_model",
    "benchmark_reg_model",
    "benchmark_linear_models",
    "add_model_predictions",
    # Architectures
    "LinearModel",
    "NonLinearModel",
    "DeepModel",
    "LSTMModel",
    "AttentionModel",
})

_LAZY_UTILS_COMMON_SYMBOLS = frozenset({
    "set_seed",
    "to_tensor",
    "init_weights",
})

_LAZY_BACKTEST_VECTORIZED_SYMBOLS = frozenset({
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
})


def __getattr__(name: str) -> Any:
    # Resolve heavy (torch-loading) symbols on demand.
    if name in _LAZY_MODELS_SYMBOLS:
        from . import models  # noqa: PLC0415
        attr = getattr(models, name)
    elif name in _LAZY_UTILS_COMMON_SYMBOLS:
        from .utils import common  # noqa: PLC0415
        attr = getattr(common, name)
    elif name in _LAZY_BACKTEST_VECTORIZED_SYMBOLS:
        from . import backtest  # noqa: PLC0415
        attr = getattr(backtest, name)
    else:
        raise AttributeError(f"module 'quant_research' has no attribute {name!r}")
    globals()[name] = attr
    return attr


def __dir__() -> list[str]:
    return sorted(
        set(globals())
        | _LAZY_MODELS_SYMBOLS
        | _LAZY_UTILS_COMMON_SYMBOLS
        | _LAZY_BACKTEST_VECTORIZED_SYMBOLS
    )


if TYPE_CHECKING:
    from .backtest.vectorized import (  # noqa: F401
        add_compounding_trades,
        add_equity_curve,
        add_trade_log_returns,
        add_tx_fee,
        add_tx_fees,
        add_tx_fees_log,
        eval_model_performance,
        learn_model_trade_pnl,
        learn_model_trades,
        model_trade_results,
        sharpe_annualization_factor,
    )
    from .models import (  # noqa: F401
        AttentionModel,
        DeepModel,
        LinearModel,
        LSTMModel,
        NonLinearModel,
        add_model_predictions,
        batch_train_reg,
        benchmark_linear_models,
        benchmark_reg_model,
        get_linear_params,
        print_model_complexity_ratio,
        print_model_info,
        print_model_params,
        timeseries_split,
        timeseries_train_test_split,
        total_model_params,
        train_reg_model,
    )
    from .utils.common import init_weights, set_seed, to_tensor  # noqa: F401


__version__ = "1.0.0"
__author__ = "MemLabs"

__all__ = [
    # Config
    "SEED",
    "IS_WINDOWS",
    "DEFAULT_PARALLEL",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_LBFGS_LR",
    "DEFAULT_EPOCHS",
    "DEFAULT_TEST_SIZE",
    "TRADING_DAYS_PER_YEAR",
    "TRADING_HOURS_PER_DAY",
    "LOG_INTERVAL_DIVISOR",
    # Utils (plotting is eager; common is lazy)
    "plot",
    "plot_distribution",
    "plot_static_timeseries",
    "plot_multiple_lines",
    "plot_dyn_timeseries",
    "plot_column",
    "set_seed",
    "to_tensor",
    "init_weights",
    # Engineering
    "OHLC_AGGS",
    "get_trade_files",
    "load_timeseries",
    "load_ohlc_timeseries",
    "load_timeseries_range",
    "load_ohlc_timeseries_range",
    "timeseries",
    "ohlc_timeseries",
    "lag_col_names",
    "log_returns_col",
    "log_return_col",
    "log_return",
    "lag_cols",
    "add_lags",
    "add_log_return_features",
    "auto_reg_corr_matrx",
    "compare_ts_corr",
    # Models (lazy)
    "timeseries_split",
    "timeseries_train_test_split",
    "total_model_params",
    "print_model_info",
    "print_model_complexity_ratio",
    "get_linear_params",
    "print_model_params",
    "batch_train_reg",
    "train_reg_model",
    "benchmark_reg_model",
    "benchmark_linear_models",
    "add_model_predictions",
    "LinearModel",
    "NonLinearModel",
    "DeepModel",
    "LSTMModel",
    "AttentionModel",
    # Backtest vectorized (lazy) + event-driven (eager)
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
    "Position",
    "PositionBehavior",
    "LongPositionBehavior",
    "ShortPositionBehavior",
    "update_equity_record",
    "BacktestAnalysis",
    # Strategies
    "BaseStrategy",
    "EnvelopeStrategy",
    "SimpleSMAStrategy",
    "AVERAGE_TYPES",
    "sma",
    "ema",
    "wma",
    "donchian_mid",
    "moving_average",
    # Connectors (submodule)
    "connectors",
]
