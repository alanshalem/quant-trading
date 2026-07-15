"""
envelope.py - Envelope mean-reversion strategy
================================================
Multi-band envelope strategy around a moving average. Opens a staggered
long (or short) position each time price pierces a lower (upper) band, and
exits when price reverts to the central moving average. Stop-loss and an
optional "price-jump" circuit breaker protect against gap risk.
"""

from datetime import datetime
from typing import Any, Dict

import polars as pl

from .base import BaseStrategy
from .indicators import moving_average


class EnvelopeStrategy(BaseStrategy):
    """
    Multi-band envelope mean-reversion strategy.

    Builds a moving average of ``close`` and N symmetric bands around it. On
    each bar, if price pierces the ``k``-th lower band while flat or long, a
    fraction of wallet is deployed at that band price (scale-in). Symmetric
    logic applies on short side. Position is closed when price crosses back
    through the moving average.

    Protective layers, from fastest to slowest:

    1. ``price_jump_pct`` (optional) — flat-exit if bar opens past
       ``open_price * (1 ± price_jump_pct)`` (gap / circuit breaker).
    2. ``stop_loss_pct`` — hard stop at ``avg_entry * (1 ± stop_loss_pct)``.
    3. Liquidation — approximated at ``open * (1 ± 1 / leverage)``.

    After a losing exit (SL or price-jump), trading pauses until price
    reverts past the moving average (controlled by ``good_to_trade``).

    Params
    ------
    average_type : {'SMA', 'EMA', 'WMA', 'DCM'}
    average_period : int
    envelopes : list[float]            # e.g. [0.07, 0.11, 0.14]
    stop_loss_pct : float              # 0.3 == 30 %
    price_jump_pct : float, optional   # gap-open circuit breaker
    position_size_percentage : float   # % of wallet, split across bands
    position_size_fixed_amount : float # alternative: fixed quote amount
    mode : {'long', 'short', 'both'}, default 'both'
    """

    def __init__(self, params: Dict[str, Any], ohlcv) -> None:
        self.good_to_trade = True
        self.position_was_closed = False
        self.n_bands_hit = 0
        self.last_position_side: str | None = None
        super().__init__(params, ohlcv)

    # ----- indicators -----

    def populate_indicators(self) -> None:
        self.data = moving_average(
            self.data,
            average_type=self.params["average_type"],
            period=self.params["average_period"],
            name="average",
            shift=1,
        )
        envelopes = self.params["envelopes"]
        exprs = []
        for i, e in enumerate(envelopes):
            exprs.append((pl.col("average") / (1 - e)).alias(f"band_high_{i + 1}"))
            exprs.append((pl.col("average") * (1 - e)).alias(f"band_low_{i + 1}"))
        self.data = self.data.with_columns(exprs)

    # ----- signals -----

    def populate_long_signals(self) -> None:
        exprs = [(pl.col("high") >= pl.col("average")).alias("close_long")]
        for i in range(len(self.params["envelopes"])):
            exprs.append(
                (pl.col("low") <= pl.col(f"band_low_{i + 1}")).alias(f"open_long_{i + 1}")
            )
        self.data = self.data.with_columns(exprs)

    def populate_short_signals(self) -> None:
        exprs = [(pl.col("low") <= pl.col("average")).alias("close_short")]
        for i in range(len(self.params["envelopes"])):
            exprs.append(
                (pl.col("high") >= pl.col(f"band_high_{i + 1}")).alias(f"open_short_{i + 1}")
            )
        self.data = self.data.with_columns(exprs)

    # ----- SL helper -----

    def _stop_loss_price(self, side: str, avg_open_price: float) -> float:
        sign = -1 if side == "long" else 1
        return avg_open_price * (1 + sign * self.params["stop_loss_pct"])

    # ----- main decision loop -----

    def evaluate_orders(self, time: datetime, row: Dict[str, Any]) -> None:
        self.position_was_closed = False

        if not self.good_to_trade and (
            (self.last_position_side == "long" and row["close"] > row["average"])
            or (self.last_position_side == "short" and row["close"] < row["average"])
        ):
            self.good_to_trade = True

        # --- exits ---
        if self.position.side == "long":
            if self._price_jump_long(row):
                self._exit_on_jump(time, row["open"], "CA long")
            elif self.position.check_for_sl(row):
                self._exit_on_sl(time, self.position.sl_price, "SL long")
            elif self.position.check_for_liquidation(row):
                raise RuntimeError(
                    f"Long liquidated at {time} (price={self.position.liquidation_price})"
                )
            elif row.get("close_long"):
                self.close_trade(time, row["average"], "Exit long")
                self.position_was_closed = True
                self.n_bands_hit = 0

        elif self.position.side == "short":
            if self._price_jump_short(row):
                self._exit_on_jump(time, row["open"], "CA short")
            elif self.position.check_for_sl(row):
                self._exit_on_sl(time, self.position.sl_price, "SL short")
            elif self.position.check_for_liquidation(row):
                raise RuntimeError(
                    f"Short liquidated at {time} (price={self.position.liquidation_price})"
                )
            elif row.get("close_short"):
                self.close_trade(time, row["average"], "Exit short")
                self.position_was_closed = True
                self.n_bands_hit = 0

        # --- entries ---
        if self.good_to_trade and not self.position_was_closed:
            self._try_entries(time, row)

    def _price_jump_long(self, row: Dict[str, Any]) -> bool:
        pj = self.params.get("price_jump_pct")
        return pj is not None and row["open"] <= self.position.open_price * (1 - pj)

    def _price_jump_short(self, row: Dict[str, Any]) -> bool:
        pj = self.params.get("price_jump_pct")
        return pj is not None and row["open"] >= self.position.open_price * (1 + pj)

    def _exit_on_jump(self, time: datetime, price: float, reason: str) -> None:
        self.close_trade(time, price, reason)
        self.good_to_trade = False
        self.n_bands_hit = 0

    def _exit_on_sl(self, time: datetime, price: float, reason: str) -> None:
        self.close_trade(time, price, reason)
        self.good_to_trade = False
        self.n_bands_hit = 0

    def _try_entries(self, time: datetime, row: Dict[str, Any]) -> None:
        envelopes = self.params["envelopes"]
        for i in range(self.n_bands_hit, len(envelopes)):
            if (
                not self.ignore_longs
                and self.position.side != "short"
                and row.get(f"open_long_{i + 1}")
            ):
                side = "long"
                price = row[f"band_low_{i + 1}"]
            elif (
                not self.ignore_shorts
                and self.position.side != "long"
                and row.get(f"open_short_{i + 1}")
            ):
                side = "short"
                price = row[f"band_high_{i + 1}"]
            else:
                continue

            self.last_position_side = side
            initial_margin = self._size_margin(envelopes)
            self.balance -= initial_margin
            self.n_bands_hit += 1

            if i == 0:
                self.position.open(
                    time, side, initial_margin, price, f"Open {side} {i + 1}",
                    sl_price=self._stop_loss_price(side, price),
                )
            else:
                self.position.add(initial_margin, price, f"Open {side} {i + 1}")
                self.position.sl_price = self._stop_loss_price(side, self.position.open_price)

    def _size_margin(self, envelopes: list[float]) -> float:
        n = len(envelopes)
        if "position_size_percentage" in self.params:
            return self.balance * round(self.params["position_size_percentage"] / 100 / n, 4)
        if "position_size_fixed_amount" in self.params:
            return round(self.params["position_size_fixed_amount"] / n, 4)
        raise KeyError("Need 'position_size_percentage' or 'position_size_fixed_amount' in params")
