"""
base.py - Base Exchange Connector
=================================
Abstract base class for all exchange connectors.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from datetime import datetime
import polars as pl


class BaseConnector(ABC):
    """
    Abstract base class for exchange connectors.

    All exchange connectors should inherit from this class and implement
    the required methods for downloading trade data.

    Attributes:
        MAKER_FEE: Fee rate for maker orders (limit orders that add liquidity)
        TAKER_FEE: Fee rate for taker orders (market orders that remove liquidity)
        exchange_name: Human-readable name of the exchange
        base_url: Base URL for the exchange's data API
    """

    MAKER_FEE: float = 0.0
    TAKER_FEE: float = 0.0
    exchange_name: str = "Base"
    base_url: str = ""

    @abstractmethod
    def download_and_unzip(
        self,
        symbol: str,
        date: str | datetime,
        download_dir: str = "data",
        cache_dir: str = "data/cache"
    ) -> pl.DataFrame:
        """
        Download and process trade data for a given symbol and date.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            date: Date to download data for
            download_dir: Directory for temporary downloads
            cache_dir: Directory for caching processed data

        Returns:
            DataFrame with trade data
        """
        pass

    @abstractmethod
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

        Args:
            symbol: Trading pair symbol
            start_date: Start date of the range
            end_date: End date of the range
            download_dir: Directory for temporary downloads
            cache_dir: Directory for caching processed data
        """
        pass

    @abstractmethod
    def download_trades(
        self,
        symbol: str,
        no_days: int,
        download_dir: str = "data",
        cache_dir: str = "data/cache",
        return_trades: bool = False
    ) -> Optional[pl.DataFrame]:
        """
        Download trades for the last N days.

        Args:
            symbol: Trading pair symbol
            no_days: Number of days to download
            download_dir: Directory for temporary downloads
            cache_dir: Directory for caching processed data
            return_trades: Whether to return concatenated DataFrame

        Returns:
            Concatenated DataFrame if return_trades=True, else None
        """
        pass

    @abstractmethod
    def download_ohlc_timeseries(
        self,
        symbol: str,
        no_days: int,
        time_interval: str,
        download_dir: str = "data",
        cache_dir: str = "data/cache"
    ) -> pl.DataFrame:
        """
        Download and aggregate trades into OHLC time series.

        Args:
            symbol: Trading pair symbol
            no_days: Number of days to download
            time_interval: Aggregation interval (e.g., '1h', '12h', '1d')
            download_dir: Directory for temporary downloads
            cache_dir: Directory for caching processed data

        Returns:
            DataFrame with OHLC bars
        """
        pass

    @abstractmethod
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
        Download and aggregate trades with custom aggregations.

        Args:
            symbol: Trading pair symbol
            no_days: Number of days to download
            time_interval: Aggregation interval
            aggs: List of Polars aggregation expressions
            download_dir: Directory for temporary downloads
            cache_dir: Directory for caching processed data

        Returns:
            DataFrame with aggregated data
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(maker_fee={self.MAKER_FEE}, taker_fee={self.TAKER_FEE})"
