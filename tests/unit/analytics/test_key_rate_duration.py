"""
Tests for key rate duration
"""

import pytest
from datetime import datetime, timedelta

from bondtrader.analytics.key_rate_duration import KeyRateDuration
from bondtrader.core.bond_models import Bond, BondType


@pytest.mark.unit
class TestKeyRateDuration:
    """Test KeyRateDuration functionality"""

    @pytest.fixture
    def krd(self):
        """Create key rate duration calculator"""
        return KeyRateDuration()

    @pytest.fixture
    def sample_bond(self):
        """Create sample bond"""
        return Bond(
            bond_id="TEST-001",
            bond_type=BondType.CORPORATE,
            face_value=1000,
            coupon_rate=5.0,
            maturity_date=datetime.now() + timedelta(days=1825),
            issue_date=datetime.now() - timedelta(days=365),
            current_price=950,
            credit_rating="BBB",
            issuer="Test Corp",
            frequency=2,
        )

    def test_calculate_krd(self, krd, sample_bond):
        """Test calculating key rate durations"""
        result = krd.calculate_krd(sample_bond)
        assert isinstance(result, dict)
        assert "krd_by_rate" in result or "krd_values" in result

    def test_calculate_portfolio_krd(self, krd, sample_bond):
        """Test calculating portfolio KRD"""
        bonds = [sample_bond]
        weights = [1.0]
        result = krd.calculate_portfolio_krd(bonds, weights)
        assert isinstance(result, dict)

    def test_yield_curve_shock_analysis(self, krd, sample_bond):
        """Test yield curve shock analysis"""
        result = krd.yield_curve_shock_analysis([sample_bond], shock_scenarios=["parallel_shift"])
        assert isinstance(result, dict)
        assert "scenarios" in result
