"""
backtest - Backtest module
===========================
Two paradigms, sibling submodules:

- :mod:`quant_research.backtest.vectorized` — ML-pipeline PnL: model predictions
  turn into trade log-returns, fees, equity curves via Polars expressions.
  Pulls PyTorch transitively (trainer + engine need ``torch.nn`` /
  ``torch.optim``).
- :mod:`quant_research.backtest.event_driven` — bar-by-bar state machine:
  ``Position`` primitives, SL/TP/liquidation, and ``BacktestAnalysis`` metrics.
  Torch-free; only depends on polars + altair + matplotlib.

Event-driven symbols are imported eagerly. Vectorized symbols are resolved
lazily on first attribute access (PEP 562) so users who import only
``Position`` / ``BacktestAnalysis`` don't pay the ~200–500 ms torch startup
cost.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# --- Eager (torch-free) event-driven re-exports ---
from .event_driven import (
    BacktestAnalysis,
    LongPositionBehavior,
    Position,
    PositionBehavior,
    ShortPositionBehavior,
    update_equity_record,
)

# --- Lazy vectorized re-exports (defer torch import to first access) ---
_VECTORIZED_SYMBOLS = frozenset({
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
    if name in _VECTORIZED_SYMBOLS:
        from . import vectorized  # noqa: PLC0415 — intentional lazy import
        attr = getattr(vectorized, name)
        globals()[name] = attr  # cache so we only pay __getattr__ once
        return attr
    raise AttributeError(f"module 'quant_research.backtest' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _VECTORIZED_SYMBOLS)


if TYPE_CHECKING:
    # Type checkers need eager visibility for the lazy symbols.
    from .vectorized import (  # noqa: F401
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


__all__ = [
    # event_driven (eager)
    "Position",
    "PositionBehavior",
    "LongPositionBehavior",
    "ShortPositionBehavior",
    "update_equity_record",
    "BacktestAnalysis",
    # vectorized (lazy)
    *sorted(_VECTORIZED_SYMBOLS),
]
