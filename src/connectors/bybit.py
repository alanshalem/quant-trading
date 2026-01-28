"""
bybit.py - Bybit Exchange Connector
===================================
Connector for downloading Bybit Linear Perpetuals trade data.

Note: This is a template implementation. Bybit provides historical data
through their API and public data repository.
"""

from typing import List, Optional
import requests
from pathlib import Path
import polars as pl
from datetime import datetime, timedelta
from tqdm import tqdm

from .base import BaseConnector

# Fee constants (as of 2024 for regular users)
MAKER_FEE = 0.000200  # 0.02%
TAKER_FEE = 0.000550  # 0.055%


class BybitConnector(BaseConnector):
    """
    Connector for Bybit Linear Perpetuals.

    Downloads historical trade data from Bybit's public data repository.

    Attributes:
        MAKER_FEE: 0.02% (0.000200)
        TAKER_FEE: 0.055% (0.000550)

    Note:
        Bybit provides historical data at:
        https://public.bybit.com/trading/

    Example:
        >>> connector = BybitConnector()
        >>> df = connector.download_ohlc_timeseries('BTCUSDT', no_days=30, time_interval='12h')
    """

    MAKER_FEE = MAKER_FEE
    TAKER_FEE = TAKER_FEE
    exchange_name = "Bybit"
    base_url = "https://public.bybit.com/trading"

    def download_and_unzip(
        self,
        symbol: str,
        date: str | datetime,
        download_dir: str = "data",
        cache_dir: str = "data/cache"
    ) -> pl.DataFrame:
        """
        Download Bybit trade data for a given symbol and date.

        Bybit data format:
        - Files are gzip compressed CSV
        - Columns: timestamp, symbol, side, size, price, tickDirection, trdMatchID, grossValue, homeNotional, foreignNotional
        """
        date_str = date.strftime('%Y-%m-%d') if isinstance(date, datetime) else date

        cache_dir_path = Path(cache_dir) / "bybit"
        cache_dir_path.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir_path / f"{symbol}-trades-{date_str}.parquet"

        if cache_path.exists():
            return pl.read_parquet(cache_path)

        # Bybit uses a different URL structure
        # Format: https://public.bybit.com/trading/{symbol}/{symbol}{YYYY-MM-DD}.csv.gz
        url = f"{self.base_url}/{symbol}/{symbol}{date_str}.csv.gz"

        download_dir_path = Path(download_dir) / "bybit"
        download_dir_path.mkdir(parents=True, exist_ok=True)

        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            # Read directly from gzipped response
            import gzip
            import io

            content = gzip.decompress(response.content)
            df = pl.read_csv(
                io.BytesIO(content),
                schema={
                    "timestamp": pl.Float64,
                    "symbol": pl.Utf8,
                    "side": pl.Utf8,
                    "size": pl.Float64,
                    "price": pl.Float64,
                    "tickDirection": pl.Utf8,
                    "trdMatchID": pl.Utf8,
                    "grossValue": pl.Float64,
                    "homeNotional": pl.Float64,
                    "foreignNotional": pl.Float64,
                }
            ).with_columns(
                pl.from_epoch(pl.col("timestamp"), time_unit="s").alias("datetime"),
                pl.col("size").alias("qty"),
                (pl.col("price") * pl.col("size")).alias("quoteQty"),
                (pl.col("side") == "Sell").alias("isBuyerMaker"),
            )

            df.write_parquet(cache_path)
            return df

        except requests.exceptions.HTTPError as e:
            raise ValueError(f"Failed to download data for {symbol} on {date_str}: {e}")

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

        for i in tqdm(range(num_days), desc=f"[Bybit] Downloading {symbol}"):
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
        for i in tqdm(range(no_days), desc=f"[Bybit] Downloading {symbol}"):
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
        """Download and aggregate into OHLC timeseries."""
        from src.quant_research import timeseries, OHLC_AGGS

        yesterday = datetime.now() - timedelta(days=1)
        start_date = yesterday - timedelta(days=no_days - 1)

        time_series_list = []
        for i in tqdm(range(no_days), desc=f"[Bybit] Downloading {symbol}"):
            current_date = start_date + timedelta(days=i)
            try:
                trades = self.download_and_unzip(symbol, current_date, download_dir, cache_dir)
                time_series_list.append(timeseries(trades, time_interval, OHLC_AGGS))
            except Exception as e:
                tqdm.write(f"[ERROR] {symbol} {current_date.date()}: {e}")

        return pl.concat(time_series_list) if time_series_list else pl.DataFrame()

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
        for i in tqdm(range(no_days), desc=f"[Bybit] Downloading {symbol}"):
            current_date = start_date + timedelta(days=i)
            try:
                trades = self.download_and_unzip(symbol, current_date, download_dir, cache_dir)
                time_series_list.append(timeseries(trades, time_interval, aggs))
            except Exception as e:
                tqdm.write(f"[ERROR] {symbol} {current_date.date()}: {e}")

        return pl.concat(time_series_list) if time_series_list else pl.DataFrame()
