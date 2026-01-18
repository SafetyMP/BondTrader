"""
Unit tests for backtesting module
"""

import os
import sys

import pytest

pytestmark = pytest.mark.unit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fixtures.bond_factory import create_multiple_bonds

from bondtrader.analytics.backtesting import BacktestEngine
from bondtrader.core.bond_valuation import BondValuator


@pytest.fixture
def backtest_engine():
    """Create backtest engine instance"""
    return BacktestEngine()


@pytest.fixture
def historical_bonds():
    """Create historical bond data for backtesting"""
    # Create bonds for 3 time periods
    bonds_period1 = create_multiple_bonds(count=5)
    bonds_period2 = create_multiple_bonds(count=5)
    bonds_period3 = create_multiple_bonds(count=5)

    # Modify prices slightly for different periods
    for bond in bonds_period2:
        bond.current_price *= 1.01
    for bond in bonds_period3:
        bond.current_price *= 0.99

    return [bonds_period1, bonds_period2, bonds_period3]


def test_backtest_engine_initialization():
    """Test backtest engine initialization"""
    engine = BacktestEngine()
    assert engine.valuator is not None


def test_backtest_engine_with_valuator():
    """Test backtest engine with custom valuator"""
    valuator = BondValuator(risk_free_rate=0.04)
    engine = BacktestEngine(valuator=valuator)
    assert engine.valuator.risk_free_rate == 0.04


def test_backtest_arbitrage_strategy(backtest_engine, historical_bonds):
    """Test backtesting arbitrage strategy"""
    result = backtest_engine.backtest_arbitrage_strategy(historical_bonds, initial_capital=1000000, transaction_costs=True)

    assert "final_capital" in result
    assert "total_return" in result
    assert "total_return_pct" in result
    assert "trades" in result
    assert "portfolio_values" in result
    assert result["final_capital"] >= 0


def test_backtest_arbitrage_strategy_no_transaction_costs(backtest_engine, historical_bonds):
    """Test backtesting without transaction costs"""
    result = backtest_engine.backtest_arbitrage_strategy(historical_bonds, initial_capital=1000000, transaction_costs=False)

    assert "final_capital" in result
    assert result["final_capital"] >= 0


def test_backtest_arbitrage_strategy_with_different_capital(backtest_engine, historical_bonds):
    """Test backtesting with different initial capital"""
    result = backtest_engine.backtest_arbitrage_strategy(historical_bonds, initial_capital=500000)

    assert result["final_capital"] >= 0
    assert result["initial_capital"] == 500000


def test_backtest_arbitrage_strategy_single_period(backtest_engine):
    """Test backtesting with single period"""
    bonds = create_multiple_bonds(count=5)
    result = backtest_engine.backtest_arbitrage_strategy([bonds])

    assert result["final_capital"] >= 0
    assert isinstance(result["trades"], list)
