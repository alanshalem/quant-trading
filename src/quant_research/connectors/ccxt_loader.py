"""
ccxt_loader.py - CCXT-based OHLCV Loader
=========================================
Download and cache OHLCV candles via the CCXT unified API.

Unlike the per-exchange connectors in this package (which download raw
tick-level trades from Binance-compatible public dumps), this loader fetches
pre-aggregated OHLCV candles through CCXT, so it works with any exchange
CCXT supports (binance, bitget, okx, bybit, kraken, ...).

Data is cached as parquet files, one per (exchange, timeframe, symbol).

Usage:
------
    from connectors.ccxt_loader import CCXTLoader

    loader = CCXTLoader(exchange="binanceusdm")
    loader.download("BTC/USDT:USDT", "1h", start_date="2023-01-01")
    df = loader.load("BTC/USDT:USDT", "1h")
"""

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import ccxt
import polars as pl

from .._logging import get_logger

logger = get_logger(__name__)

# Bump whenever the cached parquet schema changes (column names, dtypes, order).
CACHE_SCHEMA_VERSION = 1

TIMEFRAMES: Dict[str, Dict[str, int]] = {
    "1m":  {"ms": 60_000},
    "2m":  {"ms": 120_000},
    "5m":  {"ms": 300_000},
    "15m": {"ms": 900_000},
    "30m": {"ms": 1_800_000},
    "1h":  {"ms": 3_600_000},
    "2h":  {"ms": 7_200_000},
    "4h":  {"ms": 14_400_000},
    "12h": {"ms": 43_200_000},
    "1d":  {"ms": 86_400_000},
    "1w":  {"ms": 604_800_000},
}

EXCHANGE_LIMITS: Dict[str, int] = {
    "binance": 1000,
    "binanceusdm": 1000,
    "bitget": 200,
    "okx": 300,
    "bybit": 1000,
    "kraken": 720,
    "coinbase": 300,
}


class CCXTLoader:
    """
    Download and cache OHLCV candles via CCXT.

    Parameters
    ----------
    exchange : str
        CCXT exchange id (e.g. 'binance', 'binanceusdm', 'bitget', 'okx').
    cache_dir : str | Path, optional
        Directory for parquet cache. Defaults to ``data/cache/ccxt``.

    Example
    -------
    >>> loader = CCXTLoader("binanceusdm")
    >>> loader.download("BTC/USDT:USDT", "1h", start_date="2023-01-01")
    >>> df = loader.load("BTC/USDT:USDT", "1h")
    >>> df.columns
    ['datetime', 'open', 'high', 'low', 'close', 'volume']
    """

    def __init__(self, exchange: str, cache_dir: Optional[str] = None) -> None:
        if not hasattr(ccxt, exchange):
            raise ValueError(f"Exchange '{exchange}' not supported by ccxt")
        self.exchange_id = exchange
        self.exchange = getattr(ccxt, exchange)({"enableRateLimit": True})
        self.limit = EXCHANGE_LIMITS.get(exchange, 500)

        if cache_dir is None:
            self.cache_dir = Path(__file__).resolve().parents[2] / "data" / "cache" / "ccxt"
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._markets: Optional[Dict[str, Any]] = None

    # ---------- markets ----------

    def fetch_markets(self) -> Dict[str, Any]:
        if self._markets is None:
            self._markets = self.exchange.load_markets()
        return self._markets

    @property
    def available_symbols(self) -> List[str]:
        return list(self.fetch_markets().keys())

    def market_info(self, symbol: str) -> Dict[str, Any]:
        return self.fetch_markets()[symbol]

    def market_limits(self, symbol: str) -> Dict[str, Any]:
        return self.fetch_markets()[symbol]["limits"]

    def ticker(self, symbol: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.exchange.fetch_ticker(symbol, params or {})

    # ---------- download / load ----------

    def download(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pl.DataFrame:
        """
        Download OHLCV candles to parquet cache and return the DataFrame.

        Pagination checkpoints every N batches so a KeyboardInterrupt
        (Ctrl+C) during a long-running download preserves the partial
        result: the accumulated rows are flushed to the parquet cache and
        re-raised so the user knows the download was cut short.
        """
        self._validate_symbol(symbol)
        self._validate_timeframe(timeframe)

        start_dt = self._parse_date(start_date) if start_date else datetime(2017, 1, 1)
        end_dt = self._parse_date(end_date) if end_date else datetime.now()

        try:
            rows = self._fetch_ohlcv(symbol, timeframe, start_dt, end_dt)
        except KeyboardInterrupt:
            # Flush whatever we accumulated before re-raising. Helper stores
            # partial results on self so we can recover them here.
            partial = getattr(self, "_partial_rows", [])
            if partial:
                logger.warning(
                    "Interrupted after %d candles for %s %s; flushing partial cache.",
                    len(partial), symbol, timeframe,
                )
                self._write_cache(symbol, timeframe, self._rows_to_df(partial))
            raise
        df = self._rows_to_df(rows)
        self._write_cache(symbol, timeframe, df)
        return df

    def load(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pl.DataFrame:
        """Load cached OHLCV candles, optionally filtered by date range."""
        path = self._cache_path(symbol, timeframe)
        if not path.exists():
            raise FileNotFoundError(
                f"No cached data for {symbol} {timeframe} on {self.exchange_id}. "
                "Run .download() first."
            )
        self._check_cache_version(path.parent)
        df = pl.read_parquet(path)
        if df.is_empty():
            raise ValueError(f"Cached data for {symbol} {timeframe} is empty")

        if start_date is not None:
            df = df.filter(pl.col("datetime") >= self._parse_date(start_date))
        if end_date is not None:
            df = df.filter(pl.col("datetime") <= self._parse_date(end_date))
        return df

    def _check_cache_version(self, dir_path: Path) -> None:
        """Warn if cached parquet was written by a different schema version."""
        meta = dir_path / "_meta.json"
        if not meta.exists():
            return
        try:
            stored = json.loads(meta.read_text()).get("schema_version")
        except (json.JSONDecodeError, OSError):
            return
        if stored != CACHE_SCHEMA_VERSION:
            warnings.warn(
                f"Cache schema mismatch at {dir_path}: "
                f"stored={stored}, current={CACHE_SCHEMA_VERSION}. "
                "Re-run .download() to refresh.",
                stacklevel=3,
            )

    # ---------- internals ----------

    def _fetch_ohlcv(
        self, symbol: str, timeframe: str, start_dt: datetime, end_dt: datetime
    ) -> List[List[Any]]:
        cur_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)
        step_ms = TIMEFRAMES[timeframe]["ms"]
        all_rows: List[List[Any]] = []
        # Expose to .download() so Ctrl+C handler can still flush a cache.
        self._partial_rows = all_rows

        while cur_ms < end_ms:
            batch = self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=cur_ms,
                limit=self.limit,
            )
            if not batch:
                break
            all_rows.extend(batch)
            last_ts = batch[-1][0]
            next_ms = last_ts + step_ms
            if next_ms <= cur_ms:
                break
            cur_ms = next_ms
        self._partial_rows = []  # clear — normal completion
        return all_rows

    @staticmethod
    def _rows_to_df(rows: List[List[Any]]) -> pl.DataFrame:
        if not rows:
            return pl.DataFrame(schema={
                "datetime": pl.Datetime("ms"),
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
            })
        df = pl.DataFrame(
            rows,
            schema=["ts", "open", "high", "low", "close", "volume"],
            orient="row",
        ).with_columns(
            pl.from_epoch("ts", time_unit="ms").alias("datetime")
        ).drop("ts").select(["datetime", "open", "high", "low", "close", "volume"])
        return df.unique(subset=["datetime"]).sort("datetime")

    def _cache_path(self, symbol: str, timeframe: str) -> Path:
        safe_symbol = symbol.replace("/", "-").replace(":", "-")
        dir_path = self.cache_dir / self.exchange_id / timeframe
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path / f"{safe_symbol}.parquet"

    def _write_cache(self, symbol: str, timeframe: str, df: pl.DataFrame) -> None:
        path = self._cache_path(symbol, timeframe)
        df.write_parquet(path)
        meta = path.parent / "_meta.json"
        meta.write_text(json.dumps({
            "schema_version": CACHE_SCHEMA_VERSION,
            "columns": df.columns,
            "exchange": self.exchange_id,
            "timeframe": timeframe,
        }))

    def _validate_symbol(self, symbol: str) -> None:
        if symbol not in self.available_symbols:
            raise ValueError(
                f"Symbol '{symbol}' not available on {self.exchange_id}. "
                f"Check loader.available_symbols."
            )

    @staticmethod
    def _validate_timeframe(timeframe: str) -> None:
        if timeframe not in TIMEFRAMES:
            raise ValueError(
                f"Timeframe '{timeframe}' not supported. Choose from {list(TIMEFRAMES)}."
            )

    @staticmethod
    def _parse_date(value: str) -> datetime:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
        raise ValueError(f"Date '{value}' must be 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'")
