"""Tests for exchange connectors."""

import pytest

from src.connectors.binance import BinanceConnector, MAKER_FEE, TAKER_FEE
from src.connectors.bybit import BybitConnector
from src.connectors.okx import OKXConnector
from src.connectors.kraken import KrakenConnector
from src.connectors.coinbase import CoinbaseConnector


class TestBinanceConnector:
    """Tests for Binance connector."""

    def test_fee_constants(self):
        """Test fee constants are defined correctly."""
        assert MAKER_FEE == 0.000450
        assert TAKER_FEE == 0.000450

    def test_connector_initialization(self):
        """Test connector can be initialized."""
        connector = BinanceConnector()
        assert connector.exchange_name == "Binance"
        assert connector.MAKER_FEE == MAKER_FEE
        assert connector.TAKER_FEE == TAKER_FEE


class TestBybitConnector:
    """Tests for Bybit connector."""

    def test_connector_initialization(self):
        """Test connector can be initialized."""
        connector = BybitConnector()
        assert connector.exchange_name == "Bybit"
        assert connector.MAKER_FEE == 0.000200
        assert connector.TAKER_FEE == 0.000550


class TestOKXConnector:
    """Tests for OKX connector."""

    def test_connector_initialization(self):
        """Test connector can be initialized."""
        connector = OKXConnector()
        assert connector.exchange_name == "OKX"


class TestKrakenConnector:
    """Tests for Kraken connector."""

    def test_connector_initialization(self):
        """Test connector can be initialized."""
        connector = KrakenConnector()
        assert connector.exchange_name == "Kraken"


class TestCoinbaseConnector:
    """Tests for Coinbase connector."""

    def test_connector_initialization(self):
        """Test connector can be initialized."""
        connector = CoinbaseConnector()
        assert connector.exchange_name == "Coinbase"
