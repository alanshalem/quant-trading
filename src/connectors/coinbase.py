"""
coinbase.py - Coinbase Exchange Connector
=========================================
Connector for downloading Coinbase Advanced Trade API data.

Note: Coinbase provides historical data through their Advanced Trade API.
This implementation uses their public market data endpoints.
"""

from typing import List, Optional
import requests
from pathlib import Path
import polars as pl
from datetime import datetime, timedelta
from tqdm import tqdm

from .base import BaseConnector

# Fee constants (as of 2024 for Advanced Trade - Maker/Taker pricing tier)
MAKER_FEE = 0.004000  # 0.40%
TAKER_FEE = 0.006000  # 0.60%


class CoinbaseConnector(BaseConnector):
    """
    Connector for Coinbase Advanced Trade API.

    Coinbase provides historical candlestick data through their REST API.
    Note that Coinbase is primarily a spot exchange.

    Attributes:
        MAKER_FEE: 0.40% (0.004000)
        TAKER_FEE: 0.60% (0.006000)

    Note:
        Coinbase Advanced Trade API documentation:
        https://docs.cdp.coinbase.com/advanced-trade/docs/welcome

    Example:
        >>> connector = CoinbaseConnector()
        >>> df = connector.download_ohlc_timeseries('BTC-USD', no_days=30, time_interval='1h')
    """

    MAKER_FEE = MAKER_FEE
    TAKER_FEE = TAKER_FEE
    exchange_name = "Coinbase"
    base_url = "https://api.exchange.coinbase.com"

    def _convert_interval(self, time_interval: str) -> int:
        """Convert common interval format to Coinbase format (seconds)."""
        mapping = {
            '1m': 60, '5m': 300, '15m': 900,
            '1h': 3600, '6h': 21600, '1d': 86400,
        }
        return mapping.get(time_interval, 3600)

    def download_and_unzip(
        self,
        symbol: str,
        date: str | datetime,
        download_dir: str = "data",
        cache_dir: str = "data/cache"
    ) -> pl.DataFrame:
        """
        Download Coinbase trade data for a given symbol and date.

        Note: Coinbase's trades endpoint returns recent trades.
        For historical data, consider collecting data over time.
        """
        date_str = date.strftime('%Y-%m-%d') if isinstance(date, datetime) else date

        cache_dir_path = Path(cache_dir) / "coinbase"
        cache_dir_path.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir_path / f"{symbol}-trades-{date_str}.parquet"

        if cache_path.exists():
            return pl.read_parquet(cache_path)

        # Coinbase trades endpoint
        url = f"{self.base_url}/products/{symbol}/trades"
        params = {"limit": 1000}  # Max limit

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            trades = response.json()

            if not trades:
                raise ValueError(f"No trades found for {symbol}")

            # Coinbase format: trade_id, side, size, price, time
            df = pl.DataFrame(trades).with_columns(
                pl.col("price").cast(pl.Float64),
                pl.col("size").cast(pl.Float64).alias("qty"),
                (pl.col("price").cast(pl.Float64) * pl.col("size").cast(pl.Float64)).alias("quoteQty"),
                (pl.col("side") == "sell").alias("isBuyerMaker"),
                pl.col("time").str.to_datetime().alias("datetime"),
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

        for i in tqdm(range(num_days), desc=f"[Coinbase] Downloading {symbol}"):
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
        for i in tqdm(range(no_days), desc=f"[Coinbase] Downloading {symbol}"):
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
        Download OHLC candlestick data directly from Coinbase API.

        This uses Coinbase's candles endpoint which is more efficient
        than downloading trades and aggregating.
        """
        cache_dir_path = Path(cache_dir) / "coinbase"
        cache_dir_path.mkdir(parents=True, exist_ok=True)

        granularity = self._convert_interval(time_interval)
        url = f"{self.base_url}/products/{symbol}/candles"

        # Calculate time range
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=no_days)

        # Coinbase limits to 300 candles per request
        all_candles = []
        current_end = end_time

        with tqdm(total=no_days, desc=f"[Coinbase] Downloading {symbol}") as pbar:
            while current_end > start_time:
                params = {
                    "granularity": granularity,
                    "start": start_time.isoformat(),
                    "end": current_end.isoformat(),
                }

                try:
                    response = requests.get(url, params=params)
                    response.raise_for_status()
                    candles = response.json()

                    if not candles:
                        break

                    all_candles.extend(candles)

                    # Update end time for next iteration (candles are reverse chronological)
                    oldest_time = datetime.utcfromtimestamp(candles[-1][0])
                    if oldest_time >= current_end:
                        break
                    current_end = oldest_time
                    pbar.update(1)

                except Exception as e:
                    tqdm.write(f"[ERROR] {e}")
                    break

        if not all_candles:
            return pl.DataFrame()

        # Coinbase format: [time, low, high, open, close, volume]
        df = pl.DataFrame(
            all_candles,
            schema=["time", "low", "high", "open", "close", "volume"],
            orient="row"
        ).with_columns(
            pl.from_epoch(pl.col("time").cast(pl.Int64), time_unit="s").alias("datetime"),
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
        ).select(["datetime", "open", "high", "low", "close"]).sort("datetime").unique("datetime")

        return df

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
        for i in tqdm(range(no_days), desc=f"[Coinbase] Downloading {symbol}"):
            current_date = start_date + timedelta(days=i)
            try:
                trades = self.download_and_unzip(symbol, current_date, download_dir, cache_dir)
                time_series_list.append(timeseries(trades, time_interval, aggs))
            except Exception as e:
                tqdm.write(f"[ERROR] {symbol} {current_date.date()}: {e}")

        return pl.concat(time_series_list) if time_series_list else pl.DataFrame()
