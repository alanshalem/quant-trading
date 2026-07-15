"""End-to-end tests for the event-driven strategies."""

import polars as pl
import pytest

from quant_research.backtest import BacktestAnalysis
from quant_research.strategies import (
    EnvelopeStrategy,
    SimpleSMAStrategy,
    donchian_mid,
    ema,
    moving_average,
    sma,
    wma,
)

# ----- indicators -----

def test_indicators_shapes(synthetic_ohlcv: pl.DataFrame):
    df = synthetic_ohlcv.with_columns(
        sma("close", 10),
        ema("close", 10),
        wma("close", 10),
        donchian_mid("high", "low", 10),
    )
    for col in ["sma_10", "ema_10", "wma_10", "donchian_mid"]:
        assert col in df.columns
        assert df[col].drop_nulls().len() > 0


def test_moving_average_rejects_unknown_type(synthetic_ohlcv: pl.DataFrame):
    with pytest.raises(ValueError):
        moving_average(synthetic_ohlcv, "BOGUS", 5)


# ----- EnvelopeStrategy -----

def test_envelope_runs_and_returns_trades(synthetic_ohlcv: pl.DataFrame):
    params = {
        "average_type": "SMA", "average_period": 10,
        "envelopes": [0.02, 0.04], "stop_loss_pct": 0.3,
        "position_size_percentage": 100,
    }
    strat = EnvelopeStrategy(params, synthetic_ohlcv)
    strat.run_backtest(initial_balance=1000, leverage=1, open_fee_rate=0.0002, close_fee_rate=0.0006)
    assert isinstance(strat.trades_info, pl.DataFrame)
    assert strat.equity_record.height > 0
    # Shouldn't raise even if no trades (guarded).
    if strat.trades_info.height > 0:
        required = {"open_time", "close_time", "net_pnl", "open_reason", "close_reason"}
        assert required.issubset(set(strat.trades_info.columns))


def test_envelope_invalid_mode(synthetic_ohlcv: pl.DataFrame):
    params = {
        "average_type": "SMA", "average_period": 10,
        "envelopes": [0.02], "stop_loss_pct": 0.3,
        "position_size_percentage": 100, "mode": "bogus",
    }
    with pytest.raises(ValueError):
        EnvelopeStrategy(params, synthetic_ohlcv)


# ----- SimpleSMAStrategy -----

def test_simple_sma_runs(synthetic_ohlcv: pl.DataFrame):
    params = {
        "fast_ma_period": 10, "slow_ma_period": 20, "trend_ma_period": 50,
        "position_size_percentage": 100,
    }
    strat = SimpleSMAStrategy(params, synthetic_ohlcv)
    strat.run_backtest(initial_balance=1000, leverage=1, fee_rate=0.0006)
    assert strat.equity_record.height > 0


# ----- BacktestAnalysis -----

def test_backtest_analysis_produces_metrics(synthetic_ohlcv: pl.DataFrame):
    params = {
        "average_type": "SMA", "average_period": 10,
        "envelopes": [0.02, 0.04], "stop_loss_pct": 0.3,
        "position_size_percentage": 100,
    }
    strat = EnvelopeStrategy(params, synthetic_ohlcv)
    strat.run_backtest(initial_balance=1000, leverage=1, open_fee_rate=0.0002, close_fee_rate=0.0006)
    assert strat.trades_info.height > 0, "synthetic_ohlcv fixture must produce trades"

    analysis = BacktestAnalysis(strat)
    for key in [
        "initial_balance", "final_balance", "roi", "win_rate", "sharpe_ratio",
        "max_drawdown_equity", "total_trades", "profit_factor",
    ]:
        assert key in analysis.metrics
    assert analysis.metrics["total_trades"] == strat.trades_info.height
