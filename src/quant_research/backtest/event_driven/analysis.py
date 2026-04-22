"""
analysis.py - Event-driven Backtest Analysis
=============================================
Compute metrics and produce plots from the ``trades_info`` and
``equity_record`` DataFrames produced by
:class:`quant_research.strategies.base.BaseStrategy`.

All metrics are computed with Polars. Plots use Altair (interactive) by
default with a Matplotlib fallback.
"""

from __future__ import annotations

import io
from typing import Any, Dict, Optional

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import polars as pl


class BacktestAnalysis:
    """Compute and visualise performance from a strategy's backtest output."""

    def __init__(self, strategy) -> None:
        if not hasattr(strategy, "trades_info") or not hasattr(strategy, "equity_record"):
            raise ValueError("Expected a strategy object with .trades_info and .equity_record")

        self.trades: pl.DataFrame = strategy.trades_info
        self.equity: pl.DataFrame = strategy.equity_record
        self.data: pl.DataFrame = getattr(strategy, "data", pl.DataFrame())

        if self.trades.is_empty():
            raise ValueError("trades_info is empty, run the backtest first")
        if self.equity.is_empty():
            raise ValueError("equity_record is empty, run the backtest first")

        self.metrics: Dict[str, Any] = {}
        self._compute_metrics()

    # ----- metrics -----

    def _compute_metrics(self) -> None:
        trades = self.trades.with_columns(
            (pl.col("close_time") - pl.col("open_time")).alias("duration"),
        )
        trades = trades.with_columns(
            pl.col("close_balance").cum_max().alias("balance_ath"),
        ).with_columns(
            (pl.col("balance_ath") - pl.col("close_balance")).alias("drawdown"),
        ).with_columns(
            (pl.col("drawdown") / pl.col("balance_ath")).alias("drawdown_pct"),
        )
        self.trades = trades

        total_trades = trades.height
        wins = trades.filter(pl.col("net_pnl") > 0)
        losses = trades.filter(pl.col("net_pnl") < 0)
        total_good = wins.height
        total_bad = losses.height

        pnl_pos = wins["net_pnl"].sum() or 0.0
        pnl_neg = losses["net_pnl"].sum() or 0.0
        profit_factor = abs(pnl_pos / pnl_neg) if pnl_neg else float("inf")

        fees = (trades["open_fee"] + trades["close_fee"]).sum()
        biggest_fee = (trades["open_fee"] + trades["close_fee"]).max()
        avg_fee = (trades["open_fee"] + trades["close_fee"]).mean()

        # streaks
        max_win_streak = max_lose_streak = 0
        wins_flags = (trades["net_pnl"] > 0).to_list()
        cur_win = cur_lose = 0
        for flag in wins_flags:
            if flag:
                cur_win += 1
                cur_lose = 0
                max_win_streak = max(max_win_streak, cur_win)
            else:
                cur_lose += 1
                cur_win = 0
                max_lose_streak = max(max_lose_streak, cur_lose)

        # equity / drawdown on equity record
        equity = self.equity.sort("time").with_columns(
            pl.col("equity").cum_max().alias("equity_ath"),
        ).with_columns(
            (pl.col("equity_ath") - pl.col("equity")).alias("drawdown"),
        ).with_columns(
            (pl.col("drawdown") / pl.col("equity_ath")).alias("drawdown_pct"),
        )
        self.equity = equity

        initial = float(equity[0, "equity"])
        final = float(equity[-1, "equity"])
        roi = final / initial - 1

        returns = equity["equity"].pct_change().drop_nulls()
        r_mean = returns.mean() or 0.0
        r_std = returns.std() or 0.0
        neg_std = returns.filter(returns < 0).std() or 0.0
        max_dd_equity = float(equity["drawdown_pct"].max() or 0.0)

        sharpe = np.sqrt(365) * r_mean / r_std if r_std else 0.0
        sortino = np.sqrt(365) * r_mean / neg_std if neg_std else 0.0
        calmar = (r_mean * 365 / max_dd_equity) if max_dd_equity else 0.0

        first_price = float(equity[0, "price"])
        last_price = float(equity[-1, "price"])
        hodl_pct = last_price / first_price - 1
        hodl = initial * (1 + hodl_pct)
        perf_vs_hodl = (final - hodl) / hodl if hodl else 0.0

        time_span = equity[-1, "time"] - equity[0, "time"]
        total_days = max(time_span.days + 1, 1)
        trade_per_day = total_trades / total_days

        total_in_pos = trades["duration"].sum().total_seconds()
        total_secs = time_span.total_seconds() or 1
        time_in_pos_ratio = total_in_pos / total_secs

        best_idx = int(trades["net_pnl"].arg_max())
        worst_idx = int(trades["net_pnl_pct"].arg_min())

        self.metrics = {
            "period_start": equity[0, "time"],
            "period_end": equity[-1, "time"],
            "initial_balance": initial,
            "final_balance": final,
            "roi": roi,
            "hodl_pct": hodl_pct,
            "performance_vs_hodl": perf_vs_hodl,
            "total_trades": total_trades,
            "total_good_trades": total_good,
            "total_bad_trades": total_bad,
            "win_rate": total_good / total_trades if total_trades else 0.0,
            "avg_pnl_pct": float(trades["net_pnl_pct"].mean() or 0.0),
            "avg_pnl_pct_good": float(wins["net_pnl_pct"].mean() or 0.0) if total_good else 0.0,
            "avg_pnl_pct_bad": float(losses["net_pnl_pct"].mean() or 0.0) if total_bad else 0.0,
            "mean_trade_duration": trades["duration"].mean(),
            "mean_good_duration": wins["duration"].mean() if total_good else None,
            "mean_bad_duration": losses["duration"].mean() if total_bad else None,
            "max_drawdown_trades": float(trades["drawdown_pct"].max() or 0.0),
            "max_drawdown_equity": max_dd_equity,
            "profit_factor": profit_factor,
            "return_over_max_drawdown": roi / abs(max_dd_equity) if max_dd_equity else 0.0,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "total_fee": float(fees or 0.0),
            "biggest_fee": float(biggest_fee or 0.0),
            "avg_fee": float(avg_fee or 0.0),
            "best_trade": trades.row(best_idx, named=True),
            "worst_trade": trades.row(worst_idx, named=True),
            "max_win_streak": max_win_streak,
            "max_lose_streak": max_lose_streak,
            "mean_trades_per_day": trade_per_day,
            "time_in_position_ratio": time_in_pos_ratio,
        }

    # ----- reporting -----

    def print_metrics(self, path: Optional[str] = None) -> None:
        m = self.metrics
        buf = io.StringIO()
        p = lambda *a: print(*a, file=buf)

        p("--- General ---")
        p(f"Period: [{m['period_start']}] -> [{m['period_end']}]")
        p(f"Initial balance: {round(m['initial_balance'], 2)} $")
        p(f"Final balance:   {round(m['final_balance'], 2)} $")
        p(f"Performance:     {round(m['roi'] * 100, 2)} %")
        p(f"Hodl perf:       {round(m['hodl_pct'] * 100, 2)} %")
        p(f"Perf vs Hodl:    {round(m['performance_vs_hodl'] * 100, 2)} %")
        p(f"Total trades:    {m['total_trades']}")
        p(f"Time in pos:     {round(m['time_in_position_ratio'] * 100, 2)} %")

        p("\n--- Health ---")
        p(f"Win rate:        {round(m['win_rate'] * 100, 2)} %")
        p(f"Max DD trades:   -{round(m['max_drawdown_trades'] * 100, 2)} %")
        p(f"Max DD equity:   -{round(m['max_drawdown_equity'] * 100, 2)} %")
        p(f"Profit factor:   {round(m['profit_factor'], 2)}")
        p(f"ROI / MaxDD:     {round(m['return_over_max_drawdown'], 2)}")
        p(f"Sharpe:          {round(m['sharpe_ratio'], 2)}")
        p(f"Sortino:         {round(m['sortino_ratio'], 2)}")
        p(f"Calmar:          {round(m['calmar_ratio'], 2)}")

        p("\n--- Trades ---")
        p(f"Avg net PnL:     {round(m['avg_pnl_pct'], 2)} %")
        p(f"Avg trades/day:  {round(m['mean_trades_per_day'], 3)}")
        p(f"Avg duration:    {m['mean_trade_duration']}")
        p(f"Best trade:      +{round(m['best_trade']['net_pnl_pct'], 2)} % "
          f"{m['best_trade']['open_time']} -> {m['best_trade']['close_time']}")
        p(f"Worst trade:     {round(m['worst_trade']['net_pnl_pct'], 2)} % "
          f"{m['worst_trade']['open_time']} -> {m['worst_trade']['close_time']}")
        p(f"Winners:         {m['total_good_trades']}")
        p(f"Losers:          {m['total_bad_trades']}")
        p(f"Avg win pnl %:   {round(m['avg_pnl_pct_good'], 2)}")
        p(f"Avg lose pnl %:  {round(m['avg_pnl_pct_bad'], 2)}")
        p(f"Max win streak:  {m['max_win_streak']}")
        p(f"Max lose streak: {m['max_lose_streak']}")

        p("\nOpen reasons:")
        for row in self.trades.group_by("open_reason").len().sort("len", descending=True).iter_rows(named=True):
            p(f"  {row['open_reason']}: {row['len']}")
        p("Close reasons:")
        for row in self.trades.group_by("close_reason").len().sort("len", descending=True).iter_rows(named=True):
            p(f"  {row['close_reason']}: {row['len']}")

        p("\n--- Fees (quote) ---")
        p(f"Total:    {round(m['total_fee'], 2)}")
        p(f"Biggest:  {round(m['biggest_fee'], 2)}")
        p(f"Average:  {round(m['avg_fee'], 2)}")

        text = buf.getvalue()
        if path:
            with open(f"{path}_metrics.txt", "w") as f:
                f.write(text)
        else:
            print(text)

    # ----- plots -----

    def plot_equity(self, path: Optional[str] = None, plot_price: bool = True) -> alt.Chart:
        eq = self.equity.select(["time", "price", "equity"])
        price_chart = alt.Chart(eq).mark_line(color="#800080", strokeWidth=1).encode(
            x=alt.X("time:T", title="time"),
            y=alt.Y("price:Q", scale=alt.Scale(zero=False), title="asset price"),
        )
        equity_chart = alt.Chart(eq).mark_area(color="#089981", opacity=0.25).encode(
            x="time:T",
            y=alt.Y("equity:Q", scale=alt.Scale(zero=False), title="equity"),
        ) + alt.Chart(eq).mark_line(color="#089981", strokeWidth=1).encode(
            x="time:T", y="equity:Q"
        )
        chart = (equity_chart + price_chart).resolve_scale(y="independent").properties(
            width=800, height=320, title="Equity vs Asset Price" if plot_price else "Equity",
        )
        if path:
            chart.save(f"{path}_plot_equity.html")
        return chart

    def plot_drawdown(self, path: Optional[str] = None) -> alt.Chart:
        eq = self.equity.with_columns(
            (-pl.col("drawdown_pct") * 100).alias("drawdown_%")
        ).select(["time", "drawdown_%"])
        chart = alt.Chart(eq).mark_area(color="indianred", opacity=0.3).encode(
            x="time:T", y=alt.Y("drawdown_%:Q", title="drawdown (%)"),
        ) + alt.Chart(eq).mark_line(color="indianred", strokeWidth=1).encode(
            x="time:T", y="drawdown_%:Q",
        )
        chart = chart.properties(width=800, height=280, title="Drawdown")
        if path:
            chart.save(f"{path}_plot_drawdown.html")
        return chart

    def plot_monthly_performance(self, year: int | str = "all", path: Optional[str] = None):
        eq = self.equity
        years = sorted(eq["time"].dt.year().unique().to_list())
        targets = years if year == "all" else [int(year)]
        figs = []
        for yr in targets:
            subset = eq.filter(pl.col("time").dt.year() == yr).sort("time")
            if subset.is_empty():
                continue
            # Compute monthly performance (first-to-last equity in each month).
            subset = subset.with_columns(pl.col("time").dt.month().alias("month"))
            monthly = subset.group_by("month").agg(
                pl.col("equity").first().alias("first"),
                pl.col("equity").last().alias("last"),
            ).with_columns(
                ((pl.col("last") - pl.col("first")) / pl.col("first") * 100).alias("perf_pct"),
            ).sort("month")

            fig, ax = plt.subplots(figsize=(10, 4))
            colors = ["#089981" if p >= 0 else "#F23645" for p in monthly["perf_pct"].to_list()]
            ax.bar(monthly["month"].to_list(), monthly["perf_pct"].to_list(), color=colors)
            ax.axhline(0, color="black", alpha=0.4)
            ax.set_title(f"{yr} monthly performance (%)")
            ax.set_xlabel("month")
            ax.set_ylabel("perf %")
            plt.tight_layout()
            if path:
                plt.savefig(f"{path}_plot_performance_{yr}.png")
            else:
                plt.show()
            figs.append(fig)
        return figs

    def plot_candlestick(
        self,
        indicators: Optional[Dict[str, Dict[str, Any]]] = None,
        show_volume: bool = False,
    ) -> alt.LayerChart:
        """Interactive candlestick chart with trade markers (Altair-based)."""
        if self.data.is_empty():
            raise ValueError("Strategy data is empty; candlestick plot unavailable")

        ohlc = self.data.select(["datetime", "open", "high", "low", "close"])
        base = alt.Chart(ohlc).encode(x="datetime:T")
        rule = base.mark_rule().encode(y="low:Q", y2="high:Q")
        bar = base.mark_bar().encode(
            y="open:Q",
            y2="close:Q",
            color=alt.condition("datum.open <= datum.close",
                                alt.value("#089981"),
                                alt.value("#F23645")),
        )
        layers = [rule, bar]

        if indicators:
            for name, ind in indicators.items():
                df = ind["df"] if isinstance(ind["df"], pl.DataFrame) else pl.from_pandas(ind["df"])
                layers.append(
                    alt.Chart(df).mark_line(color=ind.get("color", "white"), strokeWidth=1).encode(
                        x="time:T", y=f"{name}:Q"
                    )
                )

        open_markers = alt.Chart(self.trades.select("open_time", "open_reason")).mark_point(
            shape="triangle-up", color="white", size=120, filled=True
        ).encode(x="open_time:T", tooltip=["open_reason"])
        close_markers = alt.Chart(self.trades.select("close_time", "close_reason")).mark_point(
            shape="triangle-down", color="white", size=120, filled=True
        ).encode(x="close_time:T", tooltip=["close_reason"])
        layers.extend([open_markers, close_markers])

        return alt.layer(*layers).properties(width=900, height=400).interactive()
