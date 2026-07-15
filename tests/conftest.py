"""Shared pytest fixtures."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest


@pytest.fixture
def synthetic_ohlcv() -> pl.DataFrame:
    """
    Deterministic synthetic 500-bar hourly OHLCV frame.

    Seeded RNG with a smooth sinusoidal trend + Gaussian noise to give
    envelope strategies something to trade. Seed 0 produces >=10 trades
    with default params (asserted in test_strategies.py).
    """
    rng = np.random.default_rng(0)
    n = 500
    start = datetime(2023, 1, 1)
    times = [start + timedelta(hours=i) for i in range(n)]
    close = 100 + np.cumsum(rng.standard_normal(n) * 0.5) + 5 * np.sin(np.arange(n) / 30)
    high = close + np.abs(rng.standard_normal(n) * 0.3)
    low = close - np.abs(rng.standard_normal(n) * 0.3)
    op = close + rng.standard_normal(n) * 0.1
    volume = rng.random(n) * 100
    return pl.DataFrame({
        "datetime": times,
        "open": op,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })
