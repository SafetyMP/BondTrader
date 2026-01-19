"""
Tests for multi-curve framework
"""

from datetime import datetime, timedelta

import pytest

from bondtrader.analytics.multi_curve import MultiCurveFramework
from bondtrader.core.bond_models import Bond, BondType


@pytest.mark.unit
class TestMultiCurveFramework:
    """Test MultiCurveFramework functionality"""

    @pytest.fixture
    def framework(self):
        """Create multi-curve framework"""
        return MultiCurveFramework()

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

    def test_build_ois_curve(self, framework):
        """Test building OIS curve"""
        maturities = [0.25, 0.5, 1.0, 2.0, 5.0]
        rates = [0.01, 0.015, 0.02, 0.025, 0.03]
        result = framework.build_ois_curve(maturities, rates)
        assert isinstance(result, dict)

    def test_build_libor_curve(self, framework):
        """Test building LIBOR curve"""
        maturities = [0.25, 0.5, 1.0, 2.0, 5.0]
        rates = [0.011, 0.016, 0.021, 0.026, 0.031]
        result = framework.build_libor_curve(maturities, rates)
        assert isinstance(result, dict)

    def test_get_discount_factor(self, framework):
        """Test getting discount factor"""
        result = framework.get_discount_factor(maturity=1.0, curve_type="OIS")
        assert isinstance(result, float)
        assert 0 < result < 1

    def test_get_forward_rate(self, framework):
        """Test getting forward rate"""
        result = framework.get_forward_rate(t1=0.5, t2=1.0, curve_type="LIBOR")
        assert isinstance(result, float)
        assert result > 0

    def test_calculate_basis_spread(self, framework):
        """Test calculating basis spread"""
        result = framework.calculate_basis_spread(maturity=1.0)
        assert isinstance(result, float)

    def test_price_bond_with_multi_curve(self, framework, sample_bond):
        """Test pricing bond with multi-curve"""
        result = framework.price_bond_with_multi_curve(sample_bond)
        assert isinstance(result, dict)
        assert "multi_curve_value" in result or "single_curve_value" in result
