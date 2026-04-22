"""
Event-driven backtest
======================
Bar-by-bar state machine: ``Position`` primitives plus ``BacktestAnalysis``
metrics/plots consumed by :mod:`quant_research.strategies`.
"""

from .position import (
    Position,
    PositionBehavior,
    LongPositionBehavior,
    ShortPositionBehavior,
    update_equity_record,
)

from .analysis import BacktestAnalysis

__all__ = [
    "Position",
    "PositionBehavior",
    "LongPositionBehavior",
    "ShortPositionBehavior",
    "update_equity_record",
    "BacktestAnalysis",
]
