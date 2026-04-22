"""
strategies - Event-driven Trading Strategies
=============================================
Polars-native, class-based strategies driven by an OHLCV DataFrame.

Modules
-------
- :mod:`indicators`   Polars technical indicators (SMA, EMA, WMA, Donchian)
- :mod:`base`         :class:`BaseStrategy` abstract base class
- :mod:`envelope`     :class:`EnvelopeStrategy` — multi-band mean reversion
- :mod:`sma`          :class:`SimpleSMAStrategy` — triple-SMA trend following
"""

from .base import BaseStrategy
from .envelope import EnvelopeStrategy
from .sma import SimpleSMAStrategy
from .indicators import (
    AVERAGE_TYPES,
    donchian_mid,
    ema,
    moving_average,
    sma,
    wma,
)

__all__ = [
    "BaseStrategy",
    "EnvelopeStrategy",
    "SimpleSMAStrategy",
    "AVERAGE_TYPES",
    "sma",
    "ema",
    "wma",
    "donchian_mid",
    "moving_average",
]
