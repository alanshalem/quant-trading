"""
position.py - Trade Position Primitives
========================================
``Position`` models a single long/short trade with leverage, transaction fees,
stop loss, take profit and (approximated) liquidation price.

Used by the discrete event-driven strategies in
:mod:`quant_research.strategies`. Engine-level, vectorized backtests live in
:mod:`quant_research.backtest.engine`.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional


class PositionBehavior(ABC):
    """Abstract long/short behavior for a ``Position``."""

    @abstractmethod
    def calculate_pnl(self, position: "Position", close_price: float, amount: float) -> float: ...

    @abstractmethod
    def calculate_liquidation_price(self, position: "Position", price: float) -> float: ...

    @abstractmethod
    def check_for_sl(self, position: "Position", row: Dict[str, float]) -> bool: ...

    @abstractmethod
    def check_for_tp(self, position: "Position", row: Dict[str, float]) -> bool: ...

    @abstractmethod
    def check_for_liquidation(self, position: "Position", row: Dict[str, float]) -> bool: ...


class LongPositionBehavior(PositionBehavior):
    def calculate_pnl(self, position, close_price, amount):
        return amount * (close_price - position.open_price)

    def calculate_liquidation_price(self, position, price):
        return price * (1 - 1 / position.leverage)

    def check_for_sl(self, position, row):
        return position.sl_price is not None and row["low"] <= position.sl_price

    def check_for_tp(self, position, row):
        return position.tp_price is not None and row["high"] >= position.tp_price

    def check_for_liquidation(self, position, row):
        return row["low"] <= position.liquidation_price


class ShortPositionBehavior(PositionBehavior):
    def calculate_pnl(self, position, close_price, amount):
        return amount * (position.open_price - close_price)

    def calculate_liquidation_price(self, position, price):
        return price * (1 + 1 / position.leverage)

    def check_for_sl(self, position, row):
        return position.sl_price is not None and row["high"] >= position.sl_price

    def check_for_tp(self, position, row):
        return position.tp_price is not None and row["low"] <= position.tp_price

    def check_for_liquidation(self, position, row):
        return row["high"] >= position.liquidation_price


class Position:
    """
    Single leveraged trade with fees, stop loss, take profit and liquidation.

    Lifecycle
    ---------
    - :meth:`open` — set side, margin, entry price, SL/TP; compute open fee,
      notional, contract quantity (``amount``) and liquidation price.
    - :meth:`add` — scale into the same side (pyramid-in): recomputes a
      weighted-average entry and refreshes the liquidation price.
    - :meth:`close` — settle at ``price``: computes close fee and net PnL
      (``gross_pnl - open_fee - close_fee``) and clears ``side``.

    Fees
    ----
    Separate ``open_fee_rate`` and ``close_fee_rate`` (default to the single
    ``fee_rate`` if unspecified). Fees apply to notional at each leg.

    Liquidation
    -----------
    Approximated as ``open_price * (1 - 1 / leverage)`` for longs and
    ``open_price * (1 + 1 / leverage)`` for shorts. Exchanges use
    maintenance-margin tables — **check exchange specifics before trading live**.

    Side behavior (SL/TP/liq checks) is dispatched through
    :class:`LongPositionBehavior` / :class:`ShortPositionBehavior`.
    """

    def __init__(
        self,
        leverage: int = 1,
        fee_rate: float = 0.001,
        open_fee_rate: Optional[float] = None,
        close_fee_rate: Optional[float] = None,
    ) -> None:
        self.leverage = leverage
        self.open_fee_rate = fee_rate if open_fee_rate is None else open_fee_rate
        self.close_fee_rate = fee_rate if close_fee_rate is None else close_fee_rate
        self.side: Optional[str] = None
        self.open_time: Optional[datetime] = None
        self.close_time: Optional[datetime] = None
        self.open_price: Optional[float] = None
        self.close_price: Optional[float] = None
        self.open_reason: Optional[str] = None
        self.close_reason: Optional[str] = None
        self.open_fee: Optional[float] = None
        self.close_fee: Optional[float] = None
        self.initial_margin: Optional[float] = None
        self.open_notional_value: Optional[float] = None
        self.close_notional_value: Optional[float] = None
        self.amount: Optional[float] = None
        self.net_pnl: Optional[float] = None
        self.net_pnl_pct: Optional[float] = None
        self.sl_price: Optional[float] = None
        self.tp_price: Optional[float] = None
        self.liquidation_price: Optional[float] = None
        self.behavior: Optional[PositionBehavior] = None

    def _opening_metrics(self, initial_margin: float, open_price: float) -> Dict[str, float]:
        open_notional = initial_margin * self.leverage
        open_fee = open_notional * self.open_fee_rate
        open_notional -= open_fee
        amount = open_notional / open_price
        return {"open_notional_value": open_notional, "open_fee": open_fee, "amount": amount}

    def open(
        self,
        open_time: datetime,
        side: str,
        initial_margin: float,
        open_price: float,
        open_reason: str,
        sl_price: Optional[float] = None,
        tp_price: Optional[float] = None,
    ) -> None:
        self.open_time = open_time
        self.side = side
        self.initial_margin = initial_margin
        self.open_price = open_price
        self.open_reason = open_reason
        self.sl_price = sl_price
        self.tp_price = tp_price
        self.behavior = LongPositionBehavior() if side == "long" else ShortPositionBehavior()
        metrics = self._opening_metrics(initial_margin, open_price)
        self.open_fee = metrics["open_fee"]
        self.open_notional_value = metrics["open_notional_value"]
        self.amount = metrics["amount"]
        self.liquidation_price = self.behavior.calculate_liquidation_price(self, open_price)

    def add(self, initial_margin: float, price: float, reason: str) -> None:
        self.open_reason = reason
        metrics = self._opening_metrics(initial_margin, price)
        self.open_price = (
            self.open_price * self.amount + price * metrics["amount"]
        ) / (self.amount + metrics["amount"])
        self.initial_margin += initial_margin
        self.open_fee += metrics["open_fee"]
        self.open_notional_value += metrics["open_notional_value"]
        self.amount += metrics["amount"]
        self.liquidation_price = self.behavior.calculate_liquidation_price(self, self.open_price)

    def close(self, time: datetime, price: float, reason: str) -> None:
        self.close_time = time
        self.close_price = price
        self.close_reason = reason
        pnl = self.behavior.calculate_pnl(self, price, self.amount)
        self.close_notional_value = self.open_notional_value + pnl
        self.close_fee = self.close_notional_value * self.close_fee_rate
        self.net_pnl = pnl - self.open_fee - self.close_fee
        self.net_pnl_pct = self.net_pnl / self.initial_margin * 100
        self.side = None

    # --- convenience wrappers ---

    def check_for_sl(self, row: Dict[str, float]) -> bool:
        return self.behavior.check_for_sl(self, row)

    def check_for_tp(self, row: Dict[str, float]) -> bool:
        return self.behavior.check_for_tp(self, row)

    def check_for_liquidation(self, row: Dict[str, float]) -> bool:
        return self.behavior.check_for_liquidation(self, row)

    def calculate_pnl(self, price: float, amount: float) -> float:
        return self.behavior.calculate_pnl(self, price, amount)

    def info(self) -> Dict[str, Any]:
        return {
            "open_time": self.open_time,
            "close_time": self.close_time,
            "open_reason": self.open_reason,
            "close_reason": self.close_reason,
            "open_price": self.open_price,
            "close_price": self.close_price,
            "initial_margin": self.initial_margin,
            "net_pnl": self.net_pnl,
            "net_pnl_pct": self.net_pnl_pct,
            "open_notional_value": self.open_notional_value,
            "close_notional_value": self.close_notional_value,
            "amount": self.amount,
            "open_fee": self.open_fee,
            "close_fee": self.close_fee,
            "sl_price": self.sl_price,
            "tp_price": self.tp_price,
            "liquidation_price": self.liquidation_price,
        }


def update_equity_record(
    time: datetime,
    position: Position,
    balance: float,
    price: float,
    previous_update_time: datetime,
    update_interval: timedelta,
    record: List[Dict[str, Any]],
) -> datetime:
    """Append an equity snapshot to ``record`` if enough time has elapsed."""
    if time - previous_update_time < update_interval:
        return previous_update_time

    equity = balance
    if position.side:
        unrealized_pnl = position.calculate_pnl(price, position.amount)
        close_fee = (position.open_notional_value + unrealized_pnl) * position.close_fee_rate
        equity += position.initial_margin + unrealized_pnl - position.open_fee - close_fee

    record.append({"time": time, "price": price, "equity": equity})
    return time
