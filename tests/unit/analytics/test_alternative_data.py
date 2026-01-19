"""
Tests for alternative data analytics
"""

import pytest
from datetime import datetime, timedelta

from bondtrader.analytics.alternative_data import AlternativeDataAnalyzer
from bondtrader.core.bond_models import Bond, BondType


@pytest.mark.unit
class TestAlternativeDataAnalyzer:
    """Test AlternativeDataAnalyzer functionality"""

    @pytest.fixture
    def analyzer(self):
        """Create alternative data analyzer"""
        return AlternativeDataAnalyzer()

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
            issuer="Tech Corp",
            frequency=2,
        )

    def test_calculate_esg_score(self, analyzer, sample_bond):
        """Test calculating ESG score"""
        result = analyzer.calculate_esg_score(sample_bond)
        assert "esg_score" in result
        assert "environmental_score" in result
        assert "social_score" in result
        assert "governance_score" in result
        assert 0 <= result["esg_score"] <= 100

    def test_sentiment_analysis(self, analyzer, sample_bond):
        """Test sentiment analysis"""
        result = analyzer.sentiment_analysis(sample_bond)
        assert isinstance(result, dict)

    def test_economic_factors_impact(self, analyzer, sample_bond):
        """Test economic factors impact"""
        # economic_factors_impact takes a list of bonds
        result = analyzer.economic_factors_impact([sample_bond])
        assert isinstance(result, dict)

    def test_macro_factor_adjusted_valuation(self, analyzer, sample_bond):
        """Test macro factor adjusted valuation"""
        result = analyzer.macro_factor_adjusted_valuation(sample_bond)
        assert isinstance(result, dict)
