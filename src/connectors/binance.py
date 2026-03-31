"""
binance.py - Binance Exchange Connector
=======================================
Connector for downloading Binance USDT-M Futures trade data.
"""

from typing import List, Optional
import requests
import zipfile
from pathlib import Path
import polars as pl
from datetime import datetime, timedelta
from tqdm import tqdm

from .base import BaseConnector

# Import from quant_research for timeseries aggregation
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.quant_research import timeseries, OHLC_AGGS

# Fee constants (as of 2024 for regular users)
MAKER_FEE = 0.000450  # 0.045%
TAKER_FEE = 0.000450  # 0.045%


class BinanceConnector(BaseConnector):
    """
    Connector for Binance USDT-M Futures.

    Downloads historical trade data from Binance's public data repository.
    Data is cached as parquet files for faster subsequent access.

    Attributes:
        MAKER_FEE: 0.045% (0.000450)
        TAKER_FEE: 0.045% (0.000450)

    Example:
        >>> connector = BinanceConnector()
        >>> df = connector.download_ohlc_timeseries('BTCUSDT', no_days=30, time_interval='12h')
        >>> print(df.head())
    """

    MAKER_FEE = MAKER_FEE
    TAKER_FEE = TAKER_FEE
    exchange_name = "Binance"
    base_url = "https://data.binance.vision/data/futures/um/daily/trades"

    def download_and_unzip(
        self,
        symbol: str,
        date: str | datetime,
        download_dir: str = "data",
        cache_dir: str = "data/cache"
    ) -> pl.DataFrame:
        """
        Download and unzip Binance futures trade data for a given symbol and date.
        Caches results as parquet files to avoid repeated downloads.
        """
        # Normalize date to string
        date_str = date.strftime('%Y-%m-%d') if isinstance(date, datetime) else date

        cache_dir_path = Path(cache_dir)
        cache_dir_path.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir_path / f"{symbol}-trades-{date_str}.parquet"

        if cache_path.exists():
            return pl.read_parquet(cache_path)

        url = f"{self.base_url}/{symbol}/{symbol}-trades-{date_str}.zip"

        download_dir_path = Path(download_dir)
        download_dir_path.mkdir(parents=True, exist_ok=True)
        zip_path = download_dir_path / f"{symbol}-trades-{date_str}.zip"

        # Download zip
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Extract
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(download_dir_path)

        csv_path = download_dir_path / f"{symbol}-trades-{date_str}.csv"

        # Load into Polars
        df = pl.read_csv(
            csv_path,
            schema={
                "id": pl.Int64,
                "price": pl.Float64,
                "qty": pl.Float64,
                "quoteQty": pl.Float64,
                "time": pl.Int64,
                "isBuyerMaker": pl.Boolean,
            }
        ).with_columns(
            pl.from_epoch("time", time_unit="ms").alias("datetime")
        )

        # Cache and clean
        df.write_parquet(cache_path)
        zip_path.unlink(missing_ok=True)
        csv_path.unlink(missing_ok=True)

        return df

    def download_date_range(
        self,
        symbol: str,
        start_date: str | datetime,
        end_date: str | datetime,
        download_dir: str = "data",
        cache_dir: str = "data/cache"
    ) -> None:
        """
        Download trade data for a range of dates with a progress bar.
        """
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')

        num_days = (end_date - start_date).days + 1

        for i in tqdm(range(num_days), desc=f"Downloading {symbol}"):
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
        """
        Download trades for the last N days up to yesterday with a progress bar.
        """
        yesterday = datetime.now() - timedelta(days=1)
        start_date = yesterday - timedelta(days=no_days - 1)

        dfs = []
        for i in tqdm(range(no_days), desc=f"Downloading {symbol}"):
            current_date = start_date + timedelta(days=i)
            try:
                if return_trades:
                    dfs.append(self.download_and_unzip(symbol, current_date, download_dir, cache_dir))
                else:
                    self.download_and_unzip(symbol, current_date, download_dir, cache_dir)
            except Exception as e:
                tqdm.write(f"[ERROR] {symbol} {current_date.date()}: {e}")

        return pl.concat(dfs) if return_trades else None

    def download_ohlc_timeseries(
        self,
        symbol: str,
        no_days: int,
        time_interval: str,
        download_dir: str = "data",
        cache_dir: str = "data/cache"
    ) -> pl.DataFrame:
        """
        Download trades for the last N days and aggregate into OHLC timeseries.
        """
        yesterday = datetime.now() - timedelta(days=1)
        start_date = yesterday - timedelta(days=no_days - 1)

        time_series_list = []
        for i in tqdm(range(no_days), desc=f"Downloading {symbol}"):
            current_date = start_date + timedelta(days=i)
            try:
                trades = self.download_and_unzip(symbol, current_date, download_dir, cache_dir)
                time_series_list.append(timeseries(trades, time_interval, OHLC_AGGS))
            except Exception as e:
                tqdm.write(f"[ERROR] {symbol} {current_date.date()}: {e}")
        return pl.concat(time_series_list)

    def download_timeseries(
        self,
        symbol: str,
        no_days: int,
        time_interval: str,
        aggs: List[pl.Expr],
        download_dir: str = "data",
        cache_dir: str = "data/cache"
    ) -> pl.DataFrame:
        """
        Download trades for the last N days and aggregate with custom aggregations.
        """
        yesterday = datetime.now() - timedelta(days=1)
        start_date = yesterday - timedelta(days=no_days - 1)

        time_series_list = []
        for i in tqdm(range(no_days), desc=f"Downloading {symbol}"):
            current_date = start_date + timedelta(days=i)
            try:
                trades = self.download_and_unzip(symbol, current_date, download_dir, cache_dir)
                time_series_list.append(timeseries(trades, time_interval, aggs))
            except Exception as e:
                tqdm.write(f"[ERROR] {symbol} {current_date.date()}: {e}")
        return pl.concat(time_series_list)


# Backward compatibility - module-level functions
_connector = None

def _get_connector():
    global _connector
    if _connector is None:
        _connector = BinanceConnector()
    return _connector


def download_and_unzip(symbol: str, date: str | datetime,
                       download_dir: str = "data", cache_dir: str = "data/cache") -> pl.DataFrame:
    """Backward compatible function. Use BinanceConnector class instead."""
    return _get_connector().download_and_unzip(symbol, date, download_dir, cache_dir)


def download_date_range(symbol: str, start_date: str | datetime, end_date: str | datetime,
                        download_dir: str = "data", cache_dir: str = "data/cache") -> None:
    """Backward compatible function. Use BinanceConnector class instead."""
    return _get_connector().download_date_range(symbol, start_date, end_date, download_dir, cache_dir)


def download_trades(symbol: str, no_days: int,
                    download_dir: str = "data", cache_dir: str = "data/cache", return_trades: bool = False) -> Optional[pl.DataFrame]:
    """Backward compatible function. Use BinanceConnector class instead."""
    return _get_connector().download_trades(symbol, no_days, download_dir, cache_dir, return_trades)


def download_ohlc_timeseries(symbol: str, no_days: int, time_interval: str,
                             download_dir: str = "data", cache_dir: str = "data/cache") -> pl.DataFrame:
    """Backward compatible function. Use BinanceConnector class instead."""
    return _get_connector().download_ohlc_timeseries(symbol, no_days, time_interval, download_dir, cache_dir)


def download_timeseries(symbol: str, no_days: int, time_interval: str, aggs: List[pl.Expr],
                        download_dir: str = "data", cache_dir: str = "data/cache") -> pl.DataFrame:
    """Backward compatible function. Use BinanceConnector class instead."""
    return _get_connector().download_timeseries(symbol, no_days, time_interval, aggs, download_dir, cache_dir)
