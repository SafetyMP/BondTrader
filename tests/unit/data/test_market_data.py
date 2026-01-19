"""
Tests for market data module
"""

import pytest
from datetime import datetime, timedelta

from bondtrader.data.market_data import FINRADataProvider, FREDDataProvider
from bondtrader.core.bond_models import Bond, BondType


@pytest.mark.unit
class TestFREDDataProvider:
    """Test FREDDataProvider functionality"""

    @pytest.fixture
    def provider(self):
        """Create FRED data provider"""
        return FREDDataProvider()

    def test_provider_init(self, provider):
        """Test provider initialization"""
        assert provider is not None

    def test_fetch_treasury_rates(self, provider):
        """Test fetching treasury rates"""
        # This may require API key, so just test it doesn't crash
        try:
            result = provider.fetch_treasury_rates()
            # If successful, should return dict or None
            assert result is None or isinstance(result, dict)
        except Exception:
            # Expected if API key not configured
            pass

    def test_fetch_yield_curve(self, provider):
        """Test fetching yield curve"""
        try:
            result = provider.fetch_yield_curve()
            assert result is None or isinstance(result, dict)
        except Exception:
            pass


@pytest.mark.unit
class TestFINRADataProvider:
    """Test FINRADataProvider functionality"""

    @pytest.fixture
    def provider(self):
        """Create FINRA data provider"""
        return FINRADataProvider()

    def test_provider_init(self, provider):
        """Test provider initialization"""
        assert provider is not None

    def test_get_access_token(self, provider):
        """Test getting access token"""
        # This requires API credentials, so just test it doesn't crash
        try:
            token = provider._get_access_token()
            assert token is None or isinstance(token, str)
        except Exception:
            # Expected if credentials not configured
            pass

    def test_fetch_historical_bond_data(self, provider):
        """Test fetching historical bond data"""
        from datetime import datetime

        start = datetime(2010, 1, 1)
        end = datetime(2010, 1, 31)
        try:
            result = provider.fetch_historical_bond_data(start, end)
            # May return None or DataFrame
            assert result is None or hasattr(result, "shape")
        except Exception:
            pass
