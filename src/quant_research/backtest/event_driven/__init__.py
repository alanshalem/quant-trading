"""
Event-driven backtest
======================
Bar-by-bar state machine: ``Position`` primitives plus ``BacktestAnalysis``
metrics/plots consumed by :mod:`quant_research.strategies`.
"""

from .analysis import BacktestAnalysis
from .position import (
    LongPositionBehavior,
    Position,
    PositionBehavior,
    ShortPositionBehavior,
    update_equity_record,
)

__all__ = [
    "Position",
    "PositionBehavior",
    "LongPositionBehavior",
    "ShortPositionBehavior",
    "update_equity_record",
    "BacktestAnalysis",
]
