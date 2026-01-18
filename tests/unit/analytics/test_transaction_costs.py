"""
Unit tests for transaction costs module
"""

import os
import sys

import pytest

pytestmark = pytest.mark.unit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bondtrader.analytics.transaction_costs import TransactionCostCalculator
from fixtures.bond_factory import create_test_bond


@pytest.fixture
def cost_calculator():
    """Create transaction cost calculator instance"""
    return TransactionCostCalculator()


def test_transaction_cost_calculator_initialization():
    """Test transaction cost calculator initialization"""
    calculator = TransactionCostCalculator()
    assert calculator is not None


def test_calculate_trading_cost(cost_calculator):
    """Test trading cost calculation"""
    bond = create_test_bond()
    quantity = 100
    
    result = cost_calculator.calculate_trading_cost(bond, quantity, is_buy=True)
    
    assert "commission" in result
    assert "bid_ask_spread_cost" in result
    assert "total_cost" in result
    assert result["total_cost"] >= 0


def test_calculate_trading_cost_buy_vs_sell(cost_calculator):
    """Test trading cost difference between buy and sell"""
    bond = create_test_bond()
    quantity = 100
    
    buy_cost = cost_calculator.calculate_trading_cost(bond, quantity, is_buy=True)
    sell_cost = cost_calculator.calculate_trading_cost(bond, quantity, is_buy=False)
    
    assert buy_cost["total_cost"] >= 0
    assert sell_cost["total_cost"] >= 0


def test_calculate_round_trip_cost(cost_calculator):
    """Test round trip cost calculation"""
    bond = create_test_bond()
    quantity = 100
    
    result = cost_calculator.calculate_round_trip_cost(bond, quantity)
    
    assert "buy_cost" in result
    assert "sell_cost" in result
    assert "round_trip_cost_pct" in result
    assert result["buy_cost"] >= 0
    assert result["sell_cost"] >= 0
