"""
base.py - Base class for event-driven strategies
=================================================
Provides :class:`BaseStrategy`, a thin abstract base that owns an OHLCV
DataFrame, a ``Position`` and a main bar-by-bar backtest loop. Concrete
strategies (envelope, SMA, ...) implement ``populate_indicators``,
``populate_long_signals``, ``populate_short_signals`` and
``evaluate_orders``.

Input can be a Polars DataFrame (recommended) or a Pandas DataFrame; Pandas
is converted to Polars internally. The OHLCV frame must expose a
``datetime`` column plus ``open``, ``high``, ``low``, ``close`` and
(optionally) ``volume``.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Union

import polars as pl

from ..backtest import Position, update_equity_record

OHLCVInput = Union[pl.DataFrame, "pandas.DataFrame"]  # noqa: F821


def _to_polars(df: OHLCVInput) -> pl.DataFrame:
    """Accept either a Polars or Pandas OHLCV frame, normalize to Polars."""
    if isinstance(df, pl.DataFrame):
        out = df.clone()
    else:  # assume pandas
        pdf = df.copy()
        if "datetime" not in pdf.columns:
            pdf = pdf.reset_index().rename(columns={pdf.index.name or "index": "datetime"})
        out = pl.from_pandas(pdf)
    required = {"datetime", "open", "high", "low", "close"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"OHLCV frame missing columns: {sorted(missing)}")
    return out.sort("datetime")


class BaseStrategy(ABC):
    """Abstract event-driven strategy."""

    valid_modes = ("long", "short", "both")

    def __init__(self, params: Dict[str, Any], ohlcv: OHLCVInput) -> None:
        self.params = dict(params)
        self.data = _to_polars(ohlcv)

        self.populate_indicators()
        self.set_trade_mode()

        self.trades_info: List[Dict[str, Any]] = []
        self.equity_record: List[Dict[str, Any]] = []
        self.initial_balance: float = 0.0
        self.balance: float = 0.0
        self.position: Position = Position()
        self.final_equity: float = 0.0

    # ----- trade mode -----

    def set_trade_mode(self) -> None:
        mode = self.params.setdefault("mode", "both")
        if mode not in self.valid_modes:
            raise ValueError(f"mode must be one of {self.valid_modes}")
        self.ignore_shorts = mode == "long"
        self.ignore_longs = mode == "short"

        if not self.ignore_longs:
            self.populate_long_signals()
        if not self.ignore_shorts:
            self.populate_short_signals()

    # ----- hooks -----

    @abstractmethod
    def populate_indicators(self) -> None: ...

    @abstractmethod
    def populate_long_signals(self) -> None: ...

    @abstractmethod
    def populate_short_signals(self) -> None: ...

    @abstractmethod
    def evaluate_orders(self, time: datetime, row: Dict[str, Any]) -> None: ...

    # ----- helpers for subclasses -----

    def with_columns(self, *exprs: pl.Expr) -> None:
        """Convenience: add columns to ``self.data`` in place."""
        self.data = self.data.with_columns(*exprs)

    def close_trade(self, time: datetime, price: float, reason: str) -> None:
        self.position.close(time, price, reason)
        open_balance = self.balance
        self.balance += self.position.initial_margin + self.position.net_pnl
        info = self.position.info()
        info["open_balance"] = open_balance
        info["close_balance"] = self.balance
        self.trades_info.append(info)

    # ----- main loop -----

    def run_backtest(
        self,
        initial_balance: float = 1000.0,
        leverage: int = 1,
        fee_rate: float = 0.001,
        open_fee_rate: float | None = None,
        close_fee_rate: float | None = None,
        equity_update_interval: timedelta = timedelta(hours=6),
    ) -> pl.DataFrame:
        """
        Iterate the OHLCV frame bar by bar and simulate the strategy.

        Calls :meth:`evaluate_orders` on each row and snapshots equity every
        ``equity_update_interval``.

        Parameters
        ----------
        initial_balance : float
            Starting wallet balance in quote currency.
        leverage : int
            Position leverage (used for notional sizing and liquidation).
        fee_rate : float
            Default open/close fee rate (e.g. ``0.001`` = 10 bps).
        open_fee_rate, close_fee_rate : float, optional
            Per-leg overrides; fall back to ``fee_rate`` if ``None``.
        equity_update_interval : timedelta
            Minimum wall-clock gap between equity record snapshots.

        Returns
        -------
        polars.DataFrame
            ``self.trades_info`` — closed trades with PnL and metadata.
            Also populates ``self.equity_record`` (time-indexed equity curve)
            and ``self.final_equity``.
        """
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = Position(
            leverage=leverage,
            fee_rate=fee_rate,
            open_fee_rate=open_fee_rate,
            close_fee_rate=close_fee_rate,
        )
        previous_update_time = datetime(1900, 1, 1)
        self.trades_info = []
        self.equity_record = []

        for row in self.data.iter_rows(named=True):
            time = row["datetime"]
            self.evaluate_orders(time, row)
            previous_update_time = update_equity_record(
                time=time,
                position=self.position,
                balance=self.balance,
                price=row["close"],
                previous_update_time=previous_update_time,
                update_interval=equity_update_interval,
                record=self.equity_record,
            )

        trades_df = pl.DataFrame(self.trades_info) if self.trades_info else pl.DataFrame()
        equity_df = (
            pl.DataFrame(self.equity_record).sort("time")
            if self.equity_record else pl.DataFrame()
        )
        self.trades_info = trades_df
        self.equity_record = equity_df
        if not equity_df.is_empty():
            self.final_equity = round(float(equity_df[-1, "equity"]), 2)
        return trades_df

    # ----- persistence -----

    def save_equity_record(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.equity_record.write_csv(f"{path}_equity_record.csv")

    def save_trades_info(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.trades_info.write_csv(f"{path}_trades_info.csv")
