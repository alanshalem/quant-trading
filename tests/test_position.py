"""Tests for event-driven Position and equity tracker."""

from datetime import datetime, timedelta

import pytest

from quant_research.backtest import Position, update_equity_record


@pytest.fixture
def long_position() -> Position:
    pos = Position(leverage=1, fee_rate=0.001)
    pos.open(
        open_time=datetime(2024, 1, 1),
        side="long",
        initial_margin=1000.0,
        open_price=100.0,
        open_reason="entry",
        sl_price=90.0,
        tp_price=130.0,
    )
    return pos


def test_open_populates_metrics(long_position: Position):
    assert long_position.side == "long"
    assert long_position.open_fee == pytest.approx(1000 * 0.001)
    # Amount = (notional - fee) / price
    assert long_position.amount == pytest.approx((1000 - 1) / 100)
    assert long_position.liquidation_price == pytest.approx(0.0)  # leverage=1 -> 0


def test_long_close_profitable(long_position: Position):
    long_position.close(datetime(2024, 1, 2), price=110.0, reason="exit")
    assert long_position.side is None
    # pnl = amount * (110-100)
    expected_gross = long_position.amount * 10
    expected_net = expected_gross - long_position.open_fee - long_position.close_fee
    assert long_position.net_pnl == pytest.approx(expected_net)
    assert long_position.net_pnl_pct == pytest.approx(expected_net / 1000 * 100)


def test_long_sl_tp_checks(long_position: Position):
    assert long_position.check_for_sl({"low": 89.0, "high": 120.0}) is True
    assert long_position.check_for_sl({"low": 95.0, "high": 120.0}) is False
    assert long_position.check_for_tp({"low": 95.0, "high": 131.0}) is True
    assert long_position.check_for_tp({"low": 95.0, "high": 120.0}) is False


def test_short_position_pnl():
    pos = Position(leverage=1, fee_rate=0.001)
    pos.open(datetime(2024, 1, 1), "short", 1000.0, 100.0, "entry", sl_price=110.0)
    pos.close(datetime(2024, 1, 2), 90.0, "exit")
    expected_gross = pos.amount * 10  # price drop = profit for short
    expected_net = expected_gross - pos.open_fee - pos.close_fee
    assert pos.net_pnl == pytest.approx(expected_net)


def test_position_add_updates_average_price(long_position: Position):
    before_amount = long_position.amount
    long_position.add(500.0, price=80.0, reason="add")
    assert long_position.initial_margin == pytest.approx(1500.0)
    # Average price should move towards 80.
    assert long_position.open_price < 100.0
    assert long_position.amount > before_amount


def test_update_equity_record_skips_within_interval():
    pos = Position()
    record: list = []
    t0 = datetime(2024, 1, 1)
    update = update_equity_record(t0, pos, 1000.0, 100.0, datetime(1900, 1, 1), timedelta(hours=6), record)
    assert update == t0
    assert len(record) == 1

    # Less than 6 hours later - no new record
    next_update = update_equity_record(t0 + timedelta(minutes=30), pos, 1000.0, 100.0, update, timedelta(hours=6), record)
    assert next_update == t0
    assert len(record) == 1

    # 6 hours later - new record
    final = update_equity_record(t0 + timedelta(hours=6), pos, 1000.0, 100.0, update, timedelta(hours=6), record)
    assert final == t0 + timedelta(hours=6)
    assert len(record) == 2
