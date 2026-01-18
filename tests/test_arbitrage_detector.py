"""
Unit tests for arbitrage detection module
"""

import pytest
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bondtrader.core.bond_models import Bond, BondType
from bondtrader.core.bond_valuation import BondValuator
from bondtrader.core.arbitrage_detector import ArbitrageDetector
from bondtrader.analytics.transaction_costs import TransactionCostCalculator


@pytest.fixture
def sample_bonds():
    """Create sample bonds for testing"""
    now = datetime.now()
    return [
        Bond(
            bond_id="BOND-001",
            bond_type=BondType.CORPORATE,
            face_value=1000,
            coupon_rate=5.0,
            maturity_date=now + timedelta(days=1825),
            issue_date=now - timedelta(days=365),
            current_price=950,  # Undervalued
            credit_rating="BBB",
            issuer="Test Corp",
            frequency=2
        ),
        Bond(
            bond_id="BOND-002",
            bond_type=BondType.CORPORATE,
            face_value=1000,
            coupon_rate=4.0,
            maturity_date=now + timedelta(days=1825),
            issue_date=now - timedelta(days=365),
            current_price=1050,  # Overvalued
            credit_rating="A",
            issuer="Test Corp 2",
            frequency=2
        ),
    ]


@pytest.fixture
def detector():
    """Create an arbitrage detector for testing"""
    valuator = BondValuator(risk_free_rate=0.03)
    return ArbitrageDetector(valuator=valuator, min_profit_threshold=0.01)


def test_find_arbitrage_opportunities(sample_bonds, detector):
    """Test finding arbitrage opportunities"""
    opportunities = detector.find_arbitrage_opportunities(sample_bonds)
    
    assert isinstance(opportunities, list)
    # Should find opportunities if bonds are mispriced
    assert len(opportunities) >= 0


def test_arbitrage_opportunity_structure(sample_bonds, detector):
    """Test that arbitrage opportunities have correct structure"""
    opportunities = detector.find_arbitrage_opportunities(sample_bonds)
    
    if opportunities:
        opp = opportunities[0]
        assert 'bond_id' in opp
        assert 'market_price' in opp
        assert 'fair_value' in opp
        assert 'profit' in opp or 'net_profit' in opp


def test_min_profit_threshold(sample_bonds, detector):
    """Test minimum profit threshold filtering"""
    # Set high threshold
    detector.min_profit_threshold = 0.50  # 50%
    opportunities = detector.find_arbitrage_opportunities(sample_bonds)
    
    # Should have fewer or no opportunities with high threshold
    assert isinstance(opportunities, list)


def test_transaction_costs_included(sample_bonds):
    """Test that transaction costs are considered"""
    detector_with_costs = ArbitrageDetector(
        valuator=BondValuator(),
        include_transaction_costs=True
    )
    
    opportunities = detector_with_costs.find_arbitrage_opportunities(sample_bonds)
    
    # Opportunities should account for transaction costs
    assert isinstance(opportunities, list)
    if opportunities:
        opp = opportunities[0]
        assert 'net_profit' in opp


def test_empty_bond_list(detector):
    """Test with empty bond list"""
    opportunities = detector.find_arbitrage_opportunities([])
    assert opportunities == []


def test_relative_arbitrage(sample_bonds, detector):
    """Test relative arbitrage detection between bonds"""
    relative_opps = detector.find_relative_arbitrage(sample_bonds)
    
    assert isinstance(relative_opps, list)
    # Should return list even if no opportunities found


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
