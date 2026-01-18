"""
Unit tests for arbitrage detection
"""

import os
import sys
from datetime import datetime, timedelta

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bondtrader.analytics.transaction_costs import TransactionCostCalculator
from bondtrader.core.arbitrage_detector import ArbitrageDetector
from bondtrader.core.bond_models import Bond, BondType
from bondtrader.core.bond_valuation import BondValuator


@pytest.fixture
def bonds():
    """Create sample bonds for testing"""
    return [
        Bond(
            bond_id=f"BOND-{i}",
            bond_type=BondType.CORPORATE,
            face_value=1000,
            coupon_rate=5.0 + i * 0.5,
            maturity_date=datetime.now() + timedelta(days=1825 * (i + 1)),
            issue_date=datetime.now() - timedelta(days=365),
            current_price=950 + i * 10,
            credit_rating="BBB",
            issuer=f"Corp {i}",
        )
        for i in range(5)
    ]


@pytest.fixture
def detector():
    """Create arbitrage detector for testing"""
    return ArbitrageDetector(valuator=BondValuator(), min_profit_threshold=0.01)


def test_find_arbitrage_opportunities(bonds, detector):
    """Test finding arbitrage opportunities"""
    opportunities = detector.find_arbitrage_opportunities(bonds, use_ml=False)
    assert isinstance(opportunities, list)
    for opp in opportunities:
        assert "bond_id" in opp
        assert "profit_percentage" in opp
        assert "recommendation" in opp


def test_transaction_costs():
    """Test transaction cost calculations"""
    calculator = TransactionCostCalculator()
    bond = Bond(
        bond_id="TEST",
        bond_type=BondType.CORPORATE,
        face_value=1000,
        coupon_rate=5.0,
        maturity_date=datetime.now() + timedelta(days=1825),
        issue_date=datetime.now() - timedelta(days=365),
        current_price=950,
        credit_rating="BBB",
    )

    costs = calculator.calculate_trading_cost(bond, quantity=1.0, is_buy=True)
    assert "total_cost" in costs
    assert costs["total_cost"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
