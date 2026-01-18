"""
Unit tests for portfolio optimization module
"""

import os
import sys

import pytest

pytestmark = pytest.mark.unit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bondtrader.analytics.portfolio_optimization import PortfolioOptimizer
from bondtrader.core.bond_valuation import BondValuator
from fixtures.bond_factory import create_multiple_bonds


@pytest.fixture
def optimizer():
    """Create portfolio optimizer instance"""
    return PortfolioOptimizer()


@pytest.fixture
def portfolio_bonds():
    """Create bonds for portfolio"""
    return create_multiple_bonds(count=5)


def test_optimizer_initialization():
    """Test optimizer initialization"""
    opt = PortfolioOptimizer()
    assert opt.valuator is not None


def test_optimizer_with_valuator():
    """Test optimizer initialization with custom valuator"""
    valuator = BondValuator(risk_free_rate=0.04)
    opt = PortfolioOptimizer(valuator=valuator)
    assert opt.valuator.risk_free_rate == 0.04


def test_calculate_returns_and_covariance_historical(optimizer, portfolio_bonds):
    """Test returns and covariance calculation using historical method"""
    returns, covariance = optimizer.calculate_returns_and_covariance(
        portfolio_bonds, lookback_periods=252, method="historical"
    )
    
    assert len(returns) == len(portfolio_bonds)
    assert covariance.shape == (len(portfolio_bonds), len(portfolio_bonds))
    assert all(isinstance(r, float) for r in returns)


def test_calculate_returns_and_covariance_implied(optimizer, portfolio_bonds):
    """Test returns and covariance calculation using implied method"""
    returns, covariance = optimizer.calculate_returns_and_covariance(
        portfolio_bonds, method="implied"
    )
    
    assert len(returns) == len(portfolio_bonds)
    assert covariance.shape == (len(portfolio_bonds), len(portfolio_bonds))


def test_markowitz_optimization(optimizer, portfolio_bonds):
    """Test Markowitz optimization"""
    result = optimizer.markowitz_optimization(portfolio_bonds, risk_aversion=1.0)
    
    assert "weights" in result
    assert "portfolio_return" in result
    assert "portfolio_volatility" in result
    assert "sharpe_ratio" in result
    assert len(result["weights"]) == len(portfolio_bonds)
    # Weights should sum to approximately 1
    assert abs(sum(result["weights"]) - 1.0) < 1e-6


def test_markowitz_optimization_with_target_return(optimizer, portfolio_bonds):
    """Test Markowitz optimization with target return"""
    result = optimizer.markowitz_optimization(
        portfolio_bonds, target_return=0.05, risk_aversion=1.0
    )
    
    assert "weights" in result
    assert abs(sum(result["weights"]) - 1.0) < 1e-6


def test_markowitz_optimization_weights_valid(optimizer, portfolio_bonds):
    """Test that optimized weights are valid (between 0 and 1)"""
    result = optimizer.markowitz_optimization(portfolio_bonds)
    
    for weight in result["weights"]:
        assert 0 <= weight <= 1


def test_risk_parity_optimization(optimizer, portfolio_bonds):
    """Test risk parity optimization"""
    result = optimizer.risk_parity_optimization(portfolio_bonds)
    
    assert "weights" in result
    assert len(result["weights"]) == len(portfolio_bonds)
    assert abs(sum(result["weights"]) - 1.0) < 1e-6


def test_efficient_frontier(optimizer, portfolio_bonds):
    """Test efficient frontier calculation"""
    result = optimizer.efficient_frontier(portfolio_bonds, num_points=5)
    
    assert "returns" in result
    assert "volatilities" in result
    assert "weights" in result
    assert "max_sharpe_portfolio" in result
    assert len(result["returns"]) == 5


def test_black_litterman_optimization(optimizer, portfolio_bonds):
    """Test Black-Litterman optimization"""
    result = optimizer.black_litterman_optimization(portfolio_bonds)
    
    assert "weights" in result
    assert "bl_returns" in result
    assert "equilibrium_returns" in result
    assert len(result["weights"]) == len(portfolio_bonds)
