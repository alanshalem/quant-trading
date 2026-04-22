"""
sma.py - Simple triple-SMA trend-following strategy
====================================================
Two crossing SMAs (fast/slow) define entries; a third, longer SMA filters
for trend direction. Exits are either on the reverse crossover, SL, TP or
liquidation.
"""

from datetime import datetime
from typing import Any, Dict

import polars as pl

from .base import BaseStrategy
from .indicators import sma


class SimpleSMAStrategy(BaseStrategy):
    """
    Triple-SMA trend-following strategy.

    Entries
    -------
    Long when the fast SMA crosses **above** the slow SMA **and** price is
    above the longer-period trend SMA (regime filter). Short is symmetric.

    Exits
    -----
    Fixed-ratio SL / TP set at entry (see class constants
    ``TP_LONG`` / ``SL_LONG`` / ``TP_SHORT`` / ``SL_SHORT``), plus a
    crossover-reversal signal (fast back through slow).

    Params
    ------
    fast_ma_period : int
    slow_ma_period : int
    trend_ma_period : int
    position_size_percentage | position_size_exposure | position_size_fixed_amount
    mode : {'long', 'short', 'both'}, default 'both'
    """

    # Fixed-ratio TP/SL levels (can be overridden per-subclass).
    TP_LONG: float = 1.3     # +30 %
    SL_LONG: float = 0.85    # -15 %
    TP_SHORT: float = 0.7    # -30 %
    SL_SHORT: float = 1.15   # +15 %

    # ----- indicators -----

    def populate_indicators(self) -> None:
        self.data = self.data.with_columns(
            sma("close", self.params["fast_ma_period"], name="fastMA").shift(1).alias("fastMA"),
            sma("close", self.params["slow_ma_period"], name="slowMA").shift(1).alias("slowMA"),
            sma("close", self.params["trend_ma_period"], name="trend").shift(1).alias("trend"),
        )

    # ----- signals -----

    def populate_long_signals(self) -> None:
        self.data = self.data.with_columns(
            (
                (pl.col("fastMA") > pl.col("slowMA"))
                & (pl.col("fastMA").shift(1) <= pl.col("slowMA").shift(1))
                & (pl.col("close") > pl.col("trend"))
            ).alias("open_long"),
            (pl.col("fastMA") < pl.col("slowMA")).alias("close_long"),
        )

    def populate_short_signals(self) -> None:
        self.data = self.data.with_columns(
            (
                (pl.col("fastMA") < pl.col("slowMA"))
                & (pl.col("fastMA").shift(1) >= pl.col("slowMA").shift(1))
                & (pl.col("close") < pl.col("trend"))
            ).alias("open_short"),
            (pl.col("fastMA") > pl.col("slowMA")).alias("close_short"),
        )

    # ----- SL / TP -----

    def _long_tp(self, row: Dict[str, Any]) -> float:
        return row["close"] * self.TP_LONG

    def _long_sl(self, row: Dict[str, Any]) -> float:
        return row["close"] * self.SL_LONG

    def _short_tp(self, row: Dict[str, Any]) -> float:
        return row["close"] * self.TP_SHORT

    def _short_sl(self, row: Dict[str, Any]) -> float:
        return row["close"] * self.SL_SHORT

    # ----- sizing -----

    def _size_margin(self, price: float, sl_price: float) -> float:
        if "position_size_percentage" in self.params:
            return self.balance * self.params["position_size_percentage"] / 100
        if "position_size_exposure" in self.params:
            amount_at_risk = self.balance * self.params["position_size_exposure"] / 100
            return amount_at_risk * price / abs(price - sl_price)
        if "position_size_fixed_amount" in self.params:
            return self.params["position_size_fixed_amount"]
        raise KeyError("Need a 'position_size_*' param")

    # ----- loop -----

    def evaluate_orders(self, time: datetime, row: Dict[str, Any]) -> None:
        if self.position.side == "long":
            if self.position.check_for_sl(row):
                self.close_trade(time, self.position.sl_price, "SL long")
            elif self.position.check_for_liquidation(row):
                raise RuntimeError(f"Long liquidated at {time}")
            elif self.position.check_for_tp(row):
                self.close_trade(time, self.position.tp_price, "TP long")
            elif row.get("close_long"):
                self.close_trade(time, row["close"], "Exit long")

        elif self.position.side == "short":
            if self.position.check_for_sl(row):
                self.close_trade(time, self.position.sl_price, "SL short")
            elif self.position.check_for_liquidation(row):
                raise RuntimeError(f"Short liquidated at {time}")
            elif self.position.check_for_tp(row):
                self.close_trade(time, self.position.tp_price, "TP short")
            elif row.get("close_short"):
                self.close_trade(time, row["close"], "Exit short")

        else:
            price = row["close"]
            if not self.ignore_longs and row.get("open_long"):
                sl = self._long_sl(row)
                tp = self._long_tp(row)
                margin = self._size_margin(price, sl)
                self.balance -= margin
                self.position.open(time, "long", margin, price, "Open long", sl, tp)
            elif not self.ignore_shorts and row.get("open_short"):
                sl = self._short_sl(row)
                tp = self._short_tp(row)
                margin = self._size_margin(price, sl)
                self.balance -= margin
                self.position.open(time, "short", margin, price, "Open short", sl, tp)
