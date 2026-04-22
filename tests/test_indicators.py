"""Unit tests for polars-native indicators."""

import numpy as np
import polars as pl
import pytest

from quant_research.strategies.indicators import (
    atr,
    bollinger,
    donchian_mid,
    ema,
    macd,
    moving_average,
    rsi,
    sma,
    wma,
)


@pytest.fixture
def ohlcv() -> pl.DataFrame:
    rng = np.random.default_rng(42)
    n = 200
    close = 100 + np.cumsum(rng.standard_normal(n) * 0.5)
    high = close + np.abs(rng.standard_normal(n) * 0.3)
    low = close - np.abs(rng.standard_normal(n) * 0.3)
    return pl.DataFrame({"open": close, "high": high, "low": low, "close": close})


def test_sma_matches_rolling_mean(ohlcv):
    out = ohlcv.with_columns(sma("close", 10))
    expected = ohlcv.select(pl.col("close").rolling_mean(10)).to_series()
    assert out["sma_10"].equals(expected, null_equal=True)


def test_ema_non_null_after_first(ohlcv):
    out = ohlcv.with_columns(ema("close", 10))
    # EMA with adjust=False starts producing values on row 0
    assert out["ema_10"].null_count() == 0


def test_wma_weighted_correctly():
    # 4-bar WMA of [1, 2, 3, 4] with weights 1..4 normalized = (1+4+9+16)/10 = 3.0
    df = pl.DataFrame({"close": [1.0, 2.0, 3.0, 4.0]})
    out = df.with_columns(wma("close", 4))
    assert abs(out[-1, "wma_4"] - 3.0) < 1e-9


def test_donchian_mid_shape(ohlcv):
    out = ohlcv.with_columns(donchian_mid("high", "low", 20))
    # First 19 are null; rest are finite
    assert out["donchian_mid"].null_count() == 19
    tail = out["donchian_mid"].drop_nulls()
    assert tail.is_finite().all()


def test_rsi_bounded(ohlcv):
    out = ohlcv.with_columns(rsi("close", 14))
    vals = out["rsi_14"].drop_nulls()
    assert vals.min() >= 0
    assert vals.max() <= 100


def test_rsi_all_up_is_100():
    df = pl.DataFrame({"close": list(range(1, 50))})  # strictly increasing
    out = df.with_columns(rsi("close", 14))
    # After warmup, strictly-up series pushes RSI to 100 (no losses).
    assert out[-1, "rsi_14"] == pytest.approx(100.0)


def test_macd_triple_length(ohlcv):
    m, s, h = macd("close", fast=12, slow=26, signal=9)
    out = ohlcv.with_columns(m, s, h)
    assert "macd" in out.columns
    assert "macd_signal" in out.columns
    assert "macd_hist" in out.columns
    # hist == macd - signal
    diff = (out["macd"] - out["macd_signal"]) - out["macd_hist"]
    assert diff.abs().max() < 1e-9


def test_atr_non_negative(ohlcv):
    out = ohlcv.with_columns(atr("high", "low", "close", 14))
    vals = out["atr_14"].drop_nulls()
    assert (vals >= 0).all()


def test_bollinger_bands_order(ohlcv):
    mid, up, lo = bollinger("close", window=20, std=2.0)
    out = ohlcv.with_columns(mid, up, lo).drop_nulls()
    assert (out["bb_upper"] >= out["bb_mid"]).all()
    assert (out["bb_mid"] >= out["bb_lower"]).all()


def test_moving_average_dispatch(ohlcv):
    for avg in ("SMA", "EMA", "WMA", "DCM"):
        out = moving_average(ohlcv, avg, period=10, name=f"avg_{avg}", shift=0)
        assert f"avg_{avg}" in out.columns


def test_moving_average_rejects_unknown(ohlcv):
    with pytest.raises(ValueError):
        moving_average(ohlcv, "BOGUS", period=10)
