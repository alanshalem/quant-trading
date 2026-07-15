"""
utils - Common Utilities Module
================================
Transversal helper functions: visualization (eager; altair + matplotlib)
and reproducibility / tensor helpers (lazy; pulls PyTorch).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# Plotting helpers — torch-free, import eagerly.
from .plotting import (
    plot,
    plot_column,
    plot_distribution,
    plot_dyn_timeseries,
    plot_multiple_lines,
    plot_static_timeseries,
)

# ``common`` imports ``torch`` at module load. Defer to first access so
# event-driven users don't pay the startup cost.
_LAZY_COMMON_SYMBOLS = frozenset({"set_seed", "to_tensor", "init_weights"})


def __getattr__(name: str) -> Any:
    if name in _LAZY_COMMON_SYMBOLS:
        from . import common  # noqa: PLC0415
        attr = getattr(common, name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module 'quant_research.utils' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LAZY_COMMON_SYMBOLS)


if TYPE_CHECKING:
    from .common import init_weights, set_seed, to_tensor  # noqa: F401


__all__ = [
    # plotting.py (eager)
    "plot",
    "plot_distribution",
    "plot_static_timeseries",
    "plot_multiple_lines",
    "plot_dyn_timeseries",
    "plot_column",
    # common.py (lazy)
    "set_seed",
    "to_tensor",
    "init_weights",
]
