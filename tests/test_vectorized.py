"""Tests for backtest.vectorized fee / equity helpers."""

import numpy as np
import polars as pl

from quant_research.backtest.vectorized import (
    add_equity_curve,
    add_tx_fee,
    add_tx_fees,
    add_tx_fees_log,
    sharpe_annualization_factor,
)


def test_add_tx_fee_applies_rate_to_both_legs():
    trades = pl.DataFrame({
        "entry_trade_value": [1000.0, 2000.0],
        "exit_trade_value": [1050.0, 1950.0],
    })
    out = add_tx_fee(trades, tx_fee=0.001, name="maker")
    # Fee = (entry + exit) * rate
    assert out["tx_fee_maker"][0] == 2.05
    assert out["tx_fee_maker"][1] == 3.95


def test_add_tx_fees_emits_both_maker_and_taker():
    import pytest as _pytest
    trades = pl.DataFrame({
        "entry_trade_value": [1000.0],
        "exit_trade_value": [1010.0],
    })
    out = add_tx_fees(trades, maker_fee=0.0001, taker_fee=0.0005)
    assert "tx_fee_maker" in out.columns
    assert "tx_fee_taker" in out.columns
    # Taker fee is 5x larger than maker (within FP tolerance).
    assert out["tx_fee_taker"][0] == _pytest.approx(5 * out["tx_fee_maker"][0])


def test_add_equity_curve_starts_at_initial_capital():
    trades = pl.DataFrame({"trade_pnl": [10.0, -5.0, 3.0]})
    out = add_equity_curve(trades, initial_capital=1000.0, col_name="trade_pnl", suffix="gross")
    assert out["equity_curve_gross"].to_list() == [1010.0, 1005.0, 1008.0]


def test_add_tx_fees_log_produces_both_columns():
    trades = pl.DataFrame({"trade_log_return": [0.01, -0.02, 0.005]})
    out = add_tx_fees_log(trades, maker_fee=0.0001, taker_fee=0.0003)
    for col in [
        "trade_log_return_net_maker",
        "trade_log_return_net_taker",
        "equity_curve_net_maker",
        "equity_curve_net_taker",
    ]:
        assert col in out.columns


def test_sharpe_annualization_daily_vs_hourly():
    import pytest as _pytest
    daily = sharpe_annualization_factor("1d", 365, 24)
    hourly = sharpe_annualization_factor("1h", 365, 24)
    # hourly factor = daily factor * sqrt(24)
    assert hourly == _pytest.approx(np.sqrt(24) * daily, rel=1e-9)


def test_sharpe_annualization_rejects_bad_interval():
    import pytest
    with pytest.raises(ValueError):
        sharpe_annualization_factor("abc")
