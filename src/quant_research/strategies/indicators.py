"""
indicators.py - Polars-native Technical Indicators
===================================================
Lightweight replacements for the ``ta`` library so ``quant_research`` has no
external dependency for the handful of indicators used by the bundled
strategies. All functions operate on Polars ``Series`` / ``Expr`` and return
a ``pl.Expr`` that can be plugged into ``with_columns``.
"""

from functools import lru_cache
from typing import List, Optional, Tuple

import polars as pl


def sma(col: str = "close", window: int = 20, name: Optional[str] = None) -> pl.Expr:
    """Simple moving average of ``col`` over ``window`` periods."""
    return pl.col(col).rolling_mean(window_size=window).alias(name or f"sma_{window}")


def ema(col: str = "close", window: int = 20, name: Optional[str] = None) -> pl.Expr:
    """Exponential moving average (``adjust=False`` to match classic EMA)."""
    return pl.col(col).ewm_mean(span=window, adjust=False).alias(name or f"ema_{window}")


@lru_cache(maxsize=64)
def _wma_weights(window: int) -> Tuple[float, ...]:
    """Return normalized linear weights ``1..window`` (cached per window)."""
    total = window * (window + 1) / 2  # sum of 1..window
    return tuple(i / total for i in range(1, window + 1))


def wma(col: str = "close", window: int = 20, name: Optional[str] = None) -> pl.Expr:
    """Linear weighted moving average: weights ``1..window`` (most recent heaviest)."""
    weights: List[float] = list(_wma_weights(window))
    return pl.col(col).rolling_mean(
        window_size=window, weights=weights
    ).alias(name or f"wma_{window}")


def donchian_mid(
    high: str = "high",
    low: str = "low",
    window: int = 20,
    name: str = "donchian_mid",
) -> pl.Expr:
    """Middle band of the Donchian Channel — mean of rolling max(high) and rolling min(low)."""
    upper = pl.col(high).rolling_max(window_size=window)
    lower = pl.col(low).rolling_min(window_size=window)
    return ((upper + lower) / 2).alias(name)


AVERAGE_TYPES = {"SMA", "EMA", "WMA", "DCM"}


def moving_average(
    df: pl.DataFrame,
    average_type: str,
    period: int,
    close: str = "close",
    high: str = "high",
    low: str = "low",
    name: str = "average",
    shift: int = 1,
) -> pl.DataFrame:
    """
    Add a moving-average column to ``df``.

    ``average_type`` is one of ``'SMA'``, ``'EMA'``, ``'WMA'``, ``'DCM'``.
    The indicator is shifted by ``shift`` bars (default 1) to avoid look-ahead.
    """
    if average_type not in AVERAGE_TYPES:
        raise ValueError(f"average_type must be one of {sorted(AVERAGE_TYPES)}")

    if average_type == "SMA":
        expr = sma(close, period, name=name)
    elif average_type == "EMA":
        expr = ema(close, period, name=name)
    elif average_type == "WMA":
        expr = wma(close, period, name=name)
    else:  # DCM
        expr = donchian_mid(high, low, period, name=name)

    if shift:
        expr = expr.shift(shift).alias(name)
    return df.with_columns(expr)
