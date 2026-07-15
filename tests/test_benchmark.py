"""Performance guards for the event-driven loop.

Soft budget: synthetic 500-bar run should complete in < 500 ms on
CI-class hardware. Asserts a conservative ceiling so accidental O(n^2)
or per-bar allocations show up in CI.
"""

import time

import polars as pl

from quant_research.backtest import BacktestAnalysis
from quant_research.strategies import EnvelopeStrategy

# Ceiling in ms for 500-bar backtest + metrics. Bump if the test legitimately
# grows; halve for a tighter gate.
BUDGET_MS = 2000


def test_envelope_500_bar_loop_fits_budget(synthetic_ohlcv: pl.DataFrame):
    params = {
        "average_type": "SMA",
        "average_period": 10,
        "envelopes": [0.02, 0.04],
        "stop_loss_pct": 0.3,
        "position_size_percentage": 100,
    }

    t0 = time.perf_counter()
    strat = EnvelopeStrategy(params, synthetic_ohlcv)
    strat.run_backtest(
        initial_balance=1000,
        leverage=1,
        open_fee_rate=0.0002,
        close_fee_rate=0.0006,
    )
    if strat.trades_info.height > 0:
        BacktestAnalysis(strat)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert elapsed_ms < BUDGET_MS, (
        f"Event-driven 500-bar run took {elapsed_ms:.0f}ms; budget {BUDGET_MS}ms"
    )
