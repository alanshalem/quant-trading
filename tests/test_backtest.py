"""Tests for backtesting module."""

import pytest
import polars as pl
import numpy as np

from src.quant_research import sharpe_annualization_factor


class TestSharpeAnnualization:
    """Tests for Sharpe ratio annualization."""

    def test_hourly_annualization(self):
        """Test annualization factor for hourly data."""
        factor = sharpe_annualization_factor('1h', 365, 24)
        # sqrt(365 * 24) = sqrt(8760) ≈ 93.6
        assert 93 < factor < 94

    def test_daily_annualization(self):
        """Test annualization factor for daily data."""
        factor = sharpe_annualization_factor('1d', 365, 1)
        # sqrt(365) ≈ 19.1
        assert 19 < factor < 20

    def test_minute_annualization(self):
        """Test annualization factor for minute data."""
        factor = sharpe_annualization_factor('1m', 365, 24 * 60)
        # sqrt(365 * 24 * 60) = sqrt(525600) ≈ 724.9
        assert 724 < factor < 726
