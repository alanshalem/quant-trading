"""Tests for CCXTLoader — patches ccxt to avoid real network calls."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock

import polars as pl
import pytest

from quant_research.connectors import CCXTLoader


def _fake_ohlcv_rows(start: datetime, n: int, step_ms: int) -> List[List[Any]]:
    rows = []
    ts = int(start.timestamp() * 1000)
    for i in range(n):
        rows.append([ts + i * step_ms, 100 + i, 101 + i, 99 + i, 100.5 + i, 10.0])
    return rows


@pytest.fixture
def loader(tmp_path: Path) -> CCXTLoader:
    # Skip the hasattr check by patching ccxt. `binance` is real, use it as a
    # harmless namespace; we override `.load_markets` + `.fetch_ohlcv`.
    loader = CCXTLoader("binance", cache_dir=str(tmp_path))
    loader.exchange = MagicMock()
    loader.exchange.load_markets = MagicMock(return_value={
        "BTC/USDT": {"limits": {"amount": {"min": 0.0001}}},
    })
    return loader


def test_rejects_unknown_exchange():
    with pytest.raises(ValueError, match="not supported"):
        CCXTLoader("not_a_real_exchange")


def test_download_writes_parquet_and_meta(loader, tmp_path):
    start = datetime(2024, 1, 1)
    rows = _fake_ohlcv_rows(start, n=10, step_ms=3_600_000)
    # Return full batch once, then empty to end pagination.
    loader.exchange.fetch_ohlcv = MagicMock(side_effect=[rows, []])

    df = loader.download("BTC/USDT", "1h", start_date="2024-01-01")
    assert df.height == 10
    assert set(df.columns) >= {"datetime", "open", "high", "low", "close", "volume"}

    cache_file = tmp_path / "binance" / "1h" / "BTC-USDT.parquet"
    assert cache_file.exists()

    meta = tmp_path / "binance" / "1h" / "_meta.json"
    assert meta.exists()
    stored = json.loads(meta.read_text())
    assert stored["schema_version"] >= 1
    assert stored["exchange"] == "binance"


def test_load_roundtrip(loader):
    start = datetime(2024, 1, 1)
    rows = _fake_ohlcv_rows(start, n=5, step_ms=3_600_000)
    loader.exchange.fetch_ohlcv = MagicMock(side_effect=[rows, []])
    loader.download("BTC/USDT", "1h", start_date="2024-01-01")

    df = loader.load("BTC/USDT", "1h")
    assert df.height == 5


def test_load_raises_without_cache(loader):
    with pytest.raises(FileNotFoundError):
        loader.load("BTC/USDT", "1h")


def test_date_range_filter(loader):
    start = datetime(2024, 1, 1)
    rows = _fake_ohlcv_rows(start, n=10, step_ms=3_600_000)
    loader.exchange.fetch_ohlcv = MagicMock(side_effect=[rows, []])
    loader.download("BTC/USDT", "1h", start_date="2024-01-01")

    # Get the actual stored datetime range to sidestep tz gotchas.
    full = loader.load("BTC/USDT", "1h")
    assert full.height == 10
    first_ts = full[0, "datetime"]
    cutoff_ts = full[2, "datetime"]  # keep first 3 rows

    df = loader.load(
        "BTC/USDT", "1h",
        start_date=first_ts.strftime("%Y-%m-%d %H:%M:%S"),
        end_date=cutoff_ts.strftime("%Y-%m-%d %H:%M:%S"),
    )
    assert df.height == 3


def test_graceful_shutdown_flushes_partial_cache(loader, tmp_path):
    start = datetime(2024, 1, 1)
    rows = _fake_ohlcv_rows(start, n=5, step_ms=3_600_000)

    calls = {"n": 0}

    def flaky_fetch(*_args, **_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return rows
        # Second call simulates a Ctrl+C mid-download.
        raise KeyboardInterrupt()

    loader.exchange.fetch_ohlcv = MagicMock(side_effect=flaky_fetch)

    with pytest.raises(KeyboardInterrupt):
        loader.download("BTC/USDT", "1h", start_date="2024-01-01")

    # First batch should have landed in the cache despite the interrupt.
    cache_file = tmp_path / "binance" / "1h" / "BTC-USDT.parquet"
    assert cache_file.exists()
    df = pl.read_parquet(cache_file)
    assert df.height == 5


def test_invalid_timeframe_raises(loader):
    loader.exchange.load_markets.return_value = {"BTC/USDT": {}}
    with pytest.raises(ValueError, match="Timeframe"):
        loader.download("BTC/USDT", "99z", start_date="2024-01-01")


def test_invalid_date_format_raises(loader):
    # Populate cache first so load() reaches the date-format check.
    start = datetime(2024, 1, 1)
    rows = _fake_ohlcv_rows(start, n=3, step_ms=3_600_000)
    loader.exchange.fetch_ohlcv = MagicMock(side_effect=[rows, []])
    loader.download("BTC/USDT", "1h", start_date="2024-01-01")

    with pytest.raises(ValueError, match="Date"):
        loader.load("BTC/USDT", "1h", start_date="not-a-date")
