"""
connectors - Exchange Connector Module
======================================
Unified interface for connecting to cryptocurrency exchanges
for downloading trade data and market information.

Supported Exchanges:
-------------------
- Binance (USDT-M Futures)
- Bybit (Linear Perpetuals)
- OKX (Linear Swaps)
- Kraken (Futures)
- Coinbase (Spot - Advanced Trade API)

Usage:
------
    from src.connectors import BinanceConnector, BybitConnector

    # Download trade data
    binance = BinanceConnector()
    df = binance.download_ohlc_timeseries('BTCUSDT', no_days=30, time_interval='12h')

    # Access fee information
    print(f"Maker fee: {binance.MAKER_FEE}")
    print(f"Taker fee: {binance.TAKER_FEE}")
"""

from .base import BaseConnector
from .binance import BinanceConnector, MAKER_FEE, TAKER_FEE
from .bybit import BybitConnector
from .okx import OKXConnector
from .kraken import KrakenConnector
from .coinbase import CoinbaseConnector

__all__ = [
    'BaseConnector',
    'BinanceConnector',
    'BybitConnector',
    'OKXConnector',
    'KrakenConnector',
    'CoinbaseConnector',
    'MAKER_FEE',
    'TAKER_FEE',
]
