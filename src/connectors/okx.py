"""
okx.py - OKX Exchange Connector
===============================
Connector for downloading OKX Linear Swaps trade data.

Note: OKX provides historical data through their REST API.
This implementation uses their public market data endpoints.
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
TAKER_FEE = 0.000500  # 0.05%


class OKXConnector(BaseConnector):
    """
    Connector for OKX Linear Swaps (USDT-margined perpetuals).

    OKX provides historical candlestick data through their REST API.
    For tick-level trade data, you may need to use their websocket API
    and store data locally.

    Attributes:
        MAKER_FEE: 0.02% (0.000200)
        TAKER_FEE: 0.05% (0.000500)

    Note:
        OKX API documentation:
        https://www.okx.com/docs-v5/en/

    Example:
        >>> connector = OKXConnector()
        >>> df = connector.download_ohlc_timeseries('BTC-USDT-SWAP', no_days=30, time_interval='12h')
    """

    MAKER_FEE = MAKER_FEE
    TAKER_FEE = TAKER_FEE
    exchange_name = "OKX"
    base_url = "https://www.okx.com/api/v5/market"

    def _convert_interval(self, time_interval: str) -> str:
        """Convert common interval format to OKX format."""
        mapping = {
            '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '1H', '2h': '2H', '4h': '4H', '6h': '6H', '12h': '12H',
            '1d': '1D', '1w': '1W',
        }
        return mapping.get(time_interval, time_interval)

    def download_and_unzip(
        self,
        symbol: str,
        date: str | datetime,
        download_dir: str = "data",
        cache_dir: str = "data/cache"
    ) -> pl.DataFrame:
        """
        Download OKX trade history for a given symbol and date.

        Note: OKX doesn't provide bulk historical trade downloads.
        This method uses the trades endpoint with pagination.
        For production use, consider using their WebSocket API.
        """
        date_str = date.strftime('%Y-%m-%d') if isinstance(date, datetime) else date

        cache_dir_path = Path(cache_dir) / "okx"
        cache_dir_path.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir_path / f"{symbol}-trades-{date_str}.parquet"

        if cache_path.exists():
            return pl.read_parquet(cache_path)

        # OKX trade endpoint - limited to recent trades
        # For historical data, you would need to collect data over time
        url = f"{self.base_url}/trades"
        params = {
            "instId": symbol,
            "limit": "500"  # Max limit per request
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if data["code"] != "0":
                raise ValueError(f"OKX API error: {data['msg']}")

            trades = data["data"]
            if not trades:
                raise ValueError(f"No trades found for {symbol}")

            df = pl.DataFrame(trades).with_columns(
                pl.col("ts").cast(pl.Int64).alias("time"),
                pl.col("px").cast(pl.Float64).alias("price"),
                pl.col("sz").cast(pl.Float64).alias("qty"),
                (pl.col("px").cast(pl.Float64) * pl.col("sz").cast(pl.Float64)).alias("quoteQty"),
                (pl.col("side") == "sell").alias("isBuyerMaker"),
            ).with_columns(
                pl.from_epoch("time", time_unit="ms").alias("datetime")
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
        """
        Download trade data for a range of dates.

        Note: OKX API is rate-limited. For bulk historical data,
        consider using their data export feature or WebSocket API.
        """
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')

        num_days = (end_date - start_date).days + 1

        for i in tqdm(range(num_days), desc=f"[OKX] Downloading {symbol}"):
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
        for i in tqdm(range(no_days), desc=f"[OKX] Downloading {symbol}"):
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
        Download OHLC candlestick data directly from OKX API.

        This is more efficient than downloading trades and aggregating.
        """
        cache_dir_path = Path(cache_dir) / "okx"
        cache_dir_path.mkdir(parents=True, exist_ok=True)

        okx_interval = self._convert_interval(time_interval)
        url = f"{self.base_url}/candles"

        # Calculate timestamps
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=no_days)).timestamp() * 1000)

        all_candles = []
        current_end = end_time

        with tqdm(total=no_days, desc=f"[OKX] Downloading {symbol}") as pbar:
            while current_end > start_time:
                params = {
                    "instId": symbol,
                    "bar": okx_interval,
                    "after": str(current_end),
                    "limit": "300"
                }

                try:
                    response = requests.get(url, params=params)
                    response.raise_for_status()
                    data = response.json()

                    if data["code"] != "0" or not data["data"]:
                        break

                    candles = data["data"]
                    all_candles.extend(candles)

                    # Update end time for next iteration
                    current_end = int(candles[-1][0]) - 1
                    pbar.update(1)

                except Exception as e:
                    tqdm.write(f"[ERROR] {e}")
                    break

        if not all_candles:
            return pl.DataFrame()

        # Convert to DataFrame
        df = pl.DataFrame(
            all_candles,
            schema=["ts", "open", "high", "low", "close", "vol", "volCcy", "volCcyQuote", "confirm"],
            orient="row"
        ).with_columns(
            pl.from_epoch(pl.col("ts").cast(pl.Int64), time_unit="ms").alias("datetime"),
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
        ).select(["datetime", "open", "high", "low", "close"]).sort("datetime")

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
        for i in tqdm(range(no_days), desc=f"[OKX] Downloading {symbol}"):
            current_date = start_date + timedelta(days=i)
            try:
                trades = self.download_and_unzip(symbol, current_date, download_dir, cache_dir)
                time_series_list.append(timeseries(trades, time_interval, aggs))
            except Exception as e:
                tqdm.write(f"[ERROR] {symbol} {current_date.date()}: {e}")

        return pl.concat(time_series_list) if time_series_list else pl.DataFrame()
