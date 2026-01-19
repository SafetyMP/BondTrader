"""
Tests for advanced analytics module
"""

import pytest
from datetime import datetime, timedelta

from bondtrader.analytics.advanced_analytics import AdvancedAnalytics
from bondtrader.core.bond_models import Bond, BondType
from bondtrader.core.bond_valuation import BondValuator


@pytest.mark.unit
class TestAdvancedAnalytics:
    """Test AdvancedAnalytics functionality"""

    @pytest.fixture
    def analytics(self):
        """Create advanced analytics instance"""
        return AdvancedAnalytics()

    @pytest.fixture
    def sample_bonds(self):
        """Create sample bonds"""
        now = datetime.now()
        return [
            Bond(
                bond_id=f"BOND-{i:03d}",
                bond_type=BondType.CORPORATE,
                face_value=1000,
                coupon_rate=4.0 + (i % 3),
                maturity_date=now + timedelta(days=365 * (2 + i % 5)),
                issue_date=now - timedelta(days=365),
                current_price=950 + (i * 10),
                credit_rating=["AAA", "AA", "A", "BBB", "BB"][i % 5],
                issuer=f"Test Corp {i}",
                frequency=2,
            )
            for i in range(5)
        ]

    def test_fit_yield_curve(self, analytics, sample_bonds):
        """Test fitting yield curve"""
        result = analytics.fit_yield_curve(sample_bonds, method="nelson_siegel")
        assert "method" in result
        assert "parameters" in result

    def test_calculate_z_spread(self, analytics, sample_bonds):
        """Test calculating Z-spread"""
        result = analytics.calculate_z_spread(sample_bonds[0])
        assert isinstance(result, dict)
        assert "z_spread" in result

    def test_monte_carlo_scenario(self, analytics, sample_bonds):
        """Test Monte Carlo scenario analysis"""
        result = analytics.monte_carlo_scenario(sample_bonds, num_scenarios=100)
        assert isinstance(result, dict)
        assert "scenarios" in result or "mean_portfolio_value" in result

    def test_relative_value_analysis(self, analytics, sample_bonds):
        """Test relative value analysis"""
        # Need at least 2 bonds for benchmark
        result = analytics.relative_value_analysis(sample_bonds[0], sample_bonds[1:])
        assert isinstance(result, dict)
