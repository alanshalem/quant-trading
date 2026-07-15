"""
indicators.py - Polars-native Technical Indicators
===================================================
Lightweight replacements for the ``ta`` library so ``quant_research`` has no
external dependency for the indicators used by the bundled strategies.

All functions return a ``pl.Expr`` (or a tuple of Exprs for multi-line
indicators like Bollinger / MACD) that can be plugged into ``with_columns``.

Indicators
----------
- Trend:          :func:`sma`, :func:`ema`, :func:`wma`, :func:`donchian_mid`
- Momentum:       :func:`rsi`, :func:`macd`
- Volatility:     :func:`atr`, :func:`bollinger`

Example
-------
>>> import polars as pl
>>> from quant_research.strategies.indicators import rsi, bollinger
>>> df = df.with_columns(rsi("close", window=14))
>>> mid, upper, lower = bollinger("close", window=20, std=2.0)
>>> df = df.with_columns(mid, upper, lower)
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


# ---------- Momentum ----------

def rsi(col: str = "close", window: int = 14, name: Optional[str] = None) -> pl.Expr:
    """
    Relative Strength Index (Wilder).

    ``100 - 100 / (1 + avg_gain / avg_loss)`` where ``avg_*`` are exponential
    moving averages with ``alpha = 1/window`` (Wilder smoothing), matching
    the canonical definition used by most charting libraries.
    """
    delta = pl.col(col).diff()
    gain = pl.when(delta > 0).then(delta).otherwise(0.0)
    loss = pl.when(delta < 0).then(-delta).otherwise(0.0)
    # Wilder smoothing is an EMA with alpha = 1/window.
    avg_gain = gain.ewm_mean(alpha=1.0 / window, adjust=False)
    avg_loss = loss.ewm_mean(alpha=1.0 / window, adjust=False)
    rs = avg_gain / avg_loss
    return (100 - 100 / (1 + rs)).alias(name or f"rsi_{window}")


def macd(
    col: str = "close",
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    prefix: str = "macd",
) -> Tuple[pl.Expr, pl.Expr, pl.Expr]:
    """
    MACD line, signal line, and histogram.

    Returns a ``(macd, signal, histogram)`` tuple of Exprs. Canonical
    12 / 26 / 9 exponential spans.
    """
    fast_ema = pl.col(col).ewm_mean(span=fast, adjust=False)
    slow_ema = pl.col(col).ewm_mean(span=slow, adjust=False)
    macd_line = (fast_ema - slow_ema).alias(prefix)
    signal_line = macd_line.ewm_mean(span=signal, adjust=False).alias(f"{prefix}_signal")
    hist = (macd_line - signal_line).alias(f"{prefix}_hist")
    return macd_line, signal_line, hist


# ---------- Volatility ----------

def atr(
    high: str = "high",
    low: str = "low",
    close: str = "close",
    window: int = 14,
    name: Optional[str] = None,
) -> pl.Expr:
    """
    Average True Range (Wilder).

    ``TR = max(high - low, |high - prev_close|, |low - prev_close|)``
    smoothed with Wilder's EMA (``alpha = 1/window``).
    """
    prev_close = pl.col(close).shift(1)
    tr = pl.max_horizontal(
        pl.col(high) - pl.col(low),
        (pl.col(high) - prev_close).abs(),
        (pl.col(low) - prev_close).abs(),
    )
    return tr.ewm_mean(alpha=1.0 / window, adjust=False).alias(name or f"atr_{window}")


def bollinger(
    col: str = "close",
    window: int = 20,
    std: float = 2.0,
    prefix: str = "bb",
) -> Tuple[pl.Expr, pl.Expr, pl.Expr]:
    """
    Bollinger Bands.

    Returns ``(mid, upper, lower)`` Exprs where ``mid`` is an SMA and
    ``upper/lower = mid ± std * rolling_std``.
    """
    mid = pl.col(col).rolling_mean(window_size=window).alias(f"{prefix}_mid")
    sd = pl.col(col).rolling_std(window_size=window)
    upper = (mid + std * sd).alias(f"{prefix}_upper")
    lower = (mid - std * sd).alias(f"{prefix}_lower")
    return mid, upper, lower
