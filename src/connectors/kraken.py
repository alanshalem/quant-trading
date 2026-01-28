"""
kraken.py - Kraken Exchange Connector
=====================================
Connector for downloading Kraken Futures trade data.

Note: Kraken provides historical data through their REST API
and bulk data downloads.
"""

from typing import List, Optional
import requests
from pathlib import Path
import polars as pl
from datetime import datetime, timedelta
from tqdm import tqdm

from .base import BaseConnector

# Fee constants (as of 2024 for regular users - Kraken Futures)
MAKER_FEE = 0.000200  # 0.02%
TAKER_FEE = 0.000500  # 0.05%


class KrakenConnector(BaseConnector):
    """
    Connector for Kraken Futures.

    Kraken provides historical OHLC data through their REST API.
    For tick-level trade data, you may need to use their WebSocket API.

    Attributes:
        MAKER_FEE: 0.02% (0.000200)
        TAKER_FEE: 0.05% (0.000500)

    Note:
        Kraken Futures API documentation:
        https://docs.kraken.com/api/

    Example:
        >>> connector = KrakenConnector()
        >>> df = connector.download_ohlc_timeseries('PF_XBTUSD', no_days=30, time_interval='12h')
    """

    MAKER_FEE = MAKER_FEE
    TAKER_FEE = TAKER_FEE
    exchange_name = "Kraken"
    base_url = "https://api.kraken.com/0/public"
    futures_url = "https://futures.kraken.com/derivatives/api/v3"

    def _convert_interval(self, time_interval: str) -> int:
        """Convert common interval format to Kraken format (minutes)."""
        mapping = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '4h': 240, '1d': 1440, '1w': 10080,
        }
        return mapping.get(time_interval, 60)

    def download_and_unzip(
        self,
        symbol: str,
        date: str | datetime,
        download_dir: str = "data",
        cache_dir: str = "data/cache"
    ) -> pl.DataFrame:
        """
        Download Kraken trade data for a given symbol and date.

        Note: Kraken's public trades endpoint returns recent trades.
        For historical data, consider using their data export or WebSocket.
        """
        date_str = date.strftime('%Y-%m-%d') if isinstance(date, datetime) else date

        cache_dir_path = Path(cache_dir) / "kraken"
        cache_dir_path.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir_path / f"{symbol}-trades-{date_str}.parquet"

        if cache_path.exists():
            return pl.read_parquet(cache_path)

        # Kraken spot trades endpoint
        url = f"{self.base_url}/Trades"
        params = {"pair": symbol}

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if data.get("error"):
                raise ValueError(f"Kraken API error: {data['error']}")

            # Find the result key (it varies based on symbol)
            result_key = [k for k in data["result"].keys() if k != "last"][0]
            trades = data["result"][result_key]

            if not trades:
                raise ValueError(f"No trades found for {symbol}")

            # Kraken format: [price, volume, time, buy/sell, market/limit, misc, trade_id]
            df = pl.DataFrame(
                trades,
                schema=["price", "qty", "time", "side", "order_type", "misc", "trade_id"],
                orient="row"
            ).with_columns(
                pl.col("price").cast(pl.Float64),
                pl.col("qty").cast(pl.Float64),
                pl.col("time").cast(pl.Float64),
                (pl.col("price").cast(pl.Float64) * pl.col("qty").cast(pl.Float64)).alias("quoteQty"),
                (pl.col("side") == "s").alias("isBuyerMaker"),
            ).with_columns(
                pl.from_epoch(pl.col("time"), time_unit="s").alias("datetime")
            )

            df.write_parquet(cache_path)
            return df

        except requests.exceptions.HTTPError as e:
            raise ValueError(f"Failed to download data for {symbol}: {e}")

    def download_date_range(
        self,
        symbol: str,
        start_date: str | datetime,
        end_date: str | datetime,
        download_dir: str = "data",
        cache_dir: str = "data/cache"
    ) -> None:
        """Download trade data for a range of dates."""
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')

        num_days = (end_date - start_date).days + 1

        for i in tqdm(range(num_days), desc=f"[Kraken] Downloading {symbol}"):
            current_date = start_date + timedelta(days=i)
            try:
                self.download_and_unzip(symbol, current_date, download_dir, cache_dir)
            except Exception as e:
                tqdm.write(f"[ERROR] {symbol} {current_date.date()}: {e}")

    def download_trades(
        self,
        symbol: str,
        no_days: int,
        download_dir: str = "data",
        cache_dir: str = "data/cache",
        return_trades: bool = False
    ) -> Optional[pl.DataFrame]:
        """Download trades for the last N days."""
        yesterday = datetime.now() - timedelta(days=1)
        start_date = yesterday - timedelta(days=no_days - 1)

        dfs = []
        for i in tqdm(range(no_days), desc=f"[Kraken] Downloading {symbol}"):
            current_date = start_date + timedelta(days=i)
            try:
                if return_trades:
                    dfs.append(self.download_and_unzip(symbol, current_date, download_dir, cache_dir))
                else:
                    self.download_and_unzip(symbol, current_date, download_dir, cache_dir)
            except Exception as e:
                tqdm.write(f"[ERROR] {symbol} {current_date.date()}: {e}")

        return pl.concat(dfs) if return_trades and dfs else None

    def download_ohlc_timeseries(
        self,
        symbol: str,
        no_days: int,
        time_interval: str,
        download_dir: str = "data",
        cache_dir: str = "data/cache"
    ) -> pl.DataFrame:
        """
        Download OHLC candlestick data directly from Kraken API.

        This uses Kraken's OHLC endpoint which is more efficient
        than downloading trades and aggregating.
        """
        cache_dir_path = Path(cache_dir) / "kraken"
        cache_dir_path.mkdir(parents=True, exist_ok=True)

        interval = self._convert_interval(time_interval)
        url = f"{self.base_url}/OHLC"

        # Calculate since timestamp
        since = int((datetime.now() - timedelta(days=no_days)).timestamp())

        params = {
            "pair": symbol,
            "interval": interval,
            "since": since
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if data.get("error"):
                raise ValueError(f"Kraken API error: {data['error']}")

            # Find the result key
            result_key = [k for k in data["result"].keys() if k != "last"][0]
            ohlc_data = data["result"][result_key]

            if not ohlc_data:
                return pl.DataFrame()

            # Kraken format: [time, open, high, low, close, vwap, volume, count]
            df = pl.DataFrame(
                ohlc_data,
                schema=["time", "open", "high", "low", "close", "vwap", "volume", "count"],
                orient="row"
            ).with_columns(
                pl.from_epoch(pl.col("time").cast(pl.Int64), time_unit="s").alias("datetime"),
                pl.col("open").cast(pl.Float64),
                pl.col("high").cast(pl.Float64),
                pl.col("low").cast(pl.Float64),
                pl.col("close").cast(pl.Float64),
            ).select(["datetime", "open", "high", "low", "close"]).sort("datetime")

            return df

        except Exception as e:
            raise ValueError(f"Failed to download OHLC data for {symbol}: {e}")

    def download_timeseries(
        self,
        symbol: str,
        no_days: int,
        time_interval: str,
        aggs: List[pl.Expr],
        download_dir: str = "data",
        cache_dir: str = "data/cache"
    ) -> pl.DataFrame:
        """Download and aggregate with custom aggregations."""
        from src.quant_research import timeseries

        yesterday = datetime.now() - timedelta(days=1)
        start_date = yesterday - timedelta(days=no_days - 1)

        time_series_list = []
        for i in tqdm(range(no_days), desc=f"[Kraken] Downloading {symbol}"):
            current_date = start_date + timedelta(days=i)
            try:
                trades = self.download_and_unzip(symbol, current_date, download_dir, cache_dir)
                time_series_list.append(timeseries(trades, time_interval, aggs))
            except Exception as e:
                tqdm.write(f"[ERROR] {symbol} {current_date.date()}: {e}")

        return pl.concat(time_series_list) if time_series_list else pl.DataFrame()
