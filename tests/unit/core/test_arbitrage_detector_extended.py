"""
Extended tests for arbitrage detector - additional coverage
"""

from datetime import datetime, timedelta

import pytest

from bondtrader.analytics.transaction_costs import TransactionCostCalculator
from bondtrader.core.arbitrage_detector import ArbitrageDetector
from bondtrader.core.bond_models import Bond, BondType
from bondtrader.core.bond_valuation import BondValuator


@pytest.mark.unit
class TestArbitrageDetectorExtended:
    """Extended tests for ArbitrageDetector"""

    @pytest.fixture
    def detector(self):
        """Create detector"""
        return ArbitrageDetector(valuator=BondValuator(), min_profit_threshold=0.01)

    @pytest.fixture
    def sample_bonds(self):
        """Create sample bonds"""
        now = datetime.now()
        return [
            Bond(
                bond_id="BOND-001",
                bond_type=BondType.CORPORATE,
                face_value=1000,
                coupon_rate=5.0,
                maturity_date=now + timedelta(days=1825),
                issue_date=now - timedelta(days=365),
                current_price=950,
                credit_rating="BBB",
                issuer="Test Corp",
                frequency=2,
            ),
            Bond(
                bond_id="BOND-002",
                bond_type=BondType.TREASURY,
                face_value=1000,
                coupon_rate=4.0,
                maturity_date=now + timedelta(days=1825),
                issue_date=now - timedelta(days=365),
                current_price=980,
                credit_rating="AAA",
                issuer="US Treasury",
                frequency=2,
            ),
            Bond(
                bond_id="BOND-003",
                bond_type=BondType.HIGH_YIELD,
                face_value=1000,
                coupon_rate=8.0,
                maturity_date=now + timedelta(days=1825),
                issue_date=now - timedelta(days=365),
                current_price=900,
                credit_rating="BB",
                issuer="High Yield Corp",
                frequency=2,
            ),
        ]

    def test_classify_arbitrage_type_minor(self, detector, sample_bonds):
        """Test arbitrage type classification - minor mispricing"""
        result = detector._classify_arbitrage_type(sample_bonds[0], 0.5)
        assert result == "Minor Mispricing"

    def test_classify_arbitrage_type_moderate(self, detector, sample_bonds):
        """Test arbitrage type classification - moderate"""
        result = detector._classify_arbitrage_type(sample_bonds[0], 2.0)
        assert result == "Moderate Arbitrage"

    def test_classify_arbitrage_type_significant(self, detector, sample_bonds):
        """Test arbitrage type classification - significant"""
        result = detector._classify_arbitrage_type(sample_bonds[0], 4.0)
        assert result == "Significant Arbitrage"

    def test_classify_arbitrage_type_high(self, detector, sample_bonds):
        """Test arbitrage type classification - high"""
        result = detector._classify_arbitrage_type(sample_bonds[0], 6.0)
        assert result == "High-Arbitrage Opportunity"

    def test_calculate_portfolio_arbitrage(self, detector, sample_bonds):
        """Test calculating portfolio arbitrage"""
        weights = [0.4, 0.3, 0.3]
        result = detector.calculate_portfolio_arbitrage(sample_bonds, weights)
        assert "total_market_value" in result
        assert "total_fair_value" in result
        assert "portfolio_profit" in result
        assert "num_opportunities" in result

    def test_calculate_portfolio_arbitrage_equal_weights(self, detector, sample_bonds):
        """Test calculating portfolio arbitrage with equal weights"""
        result = detector.calculate_portfolio_arbitrage(sample_bonds)
        assert "total_market_value" in result
        assert len(sample_bonds) == 3

    def test_calculate_portfolio_arbitrage_invalid_weights(self, detector, sample_bonds):
        """Test calculating portfolio arbitrage with invalid weights"""
        with pytest.raises(ValueError):
            detector.calculate_portfolio_arbitrage(sample_bonds, weights=[0.5, 0.6])

    def test_find_arbitrage_opportunities_without_ml(self, detector, sample_bonds):
        """Test finding opportunities without ML"""
        opportunities = detector.find_arbitrage_opportunities(sample_bonds, use_ml=False)
        assert isinstance(opportunities, list)

    def test_find_arbitrage_opportunities_with_ml_adjuster(self, detector, sample_bonds):
        """Test finding opportunities with ML adjuster"""
        from bondtrader.ml.ml_adjuster import MLBondAdjuster

        ml_adjuster = MLBondAdjuster(valuator=detector.valuator)
        detector.ml_adjuster = ml_adjuster
        opportunities = detector.find_arbitrage_opportunities(sample_bonds, use_ml=True)
        assert isinstance(opportunities, list)

    def test_find_arbitrage_opportunities_empty_list(self, detector):
        """Test finding opportunities with empty bond list"""
        opportunities = detector.find_arbitrage_opportunities([])
        assert opportunities == []

    def test_compare_equivalent_bonds_all_grouping(self, detector, sample_bonds):
        """Test comparing bonds with 'all' grouping key"""
        result = detector.compare_equivalent_bonds(sample_bonds, grouping_key="all")
        assert isinstance(result, list)

    def test_compare_equivalent_bonds_empty_list(self, detector):
        """Test comparing bonds with empty list"""
        result = detector.compare_equivalent_bonds([], grouping_key="bond_type")
        assert result == []
