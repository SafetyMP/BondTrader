"""
Unit tests for risk management module
"""

import os
import sys

import pytest

pytestmark = pytest.mark.unit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import sys

# Add parent directories to path for fixture imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fixtures.bond_factory import create_multiple_bonds, create_test_bond

from bondtrader.core.bond_models import Bond, BondType
from bondtrader.risk.risk_management import RiskManager


@pytest.fixture
def risk_manager():
    """Create risk manager instance"""
    return RiskManager()


@pytest.fixture
def portfolio_bonds():
    """Create bonds for portfolio"""
    return create_multiple_bonds(count=5)


def test_risk_manager_initialization():
    """Test risk manager initialization"""
    manager = RiskManager()
    assert manager.valuator is not None


def test_calculate_var_historical(risk_manager, portfolio_bonds):
    """Test VaR calculation using historical method"""
    result = risk_manager.calculate_var(
        portfolio_bonds,
        confidence_level=0.95,
        time_horizon=1,
        method="historical",
    )

    assert "var_value" in result
    assert "var_percentage" in result
    assert result["confidence_level"] == 0.95
    assert result["method"] == "historical"
    assert result["var_value"] >= 0


def test_calculate_var_parametric(risk_manager, portfolio_bonds):
    """Test VaR calculation using parametric method"""
    result = risk_manager.calculate_var(
        portfolio_bonds,
        confidence_level=0.95,
        time_horizon=1,
        method="parametric",
    )

    assert "var_value" in result
    assert "var_percentage" in result
    assert result["method"] == "parametric"


def test_calculate_var_monte_carlo(risk_manager, portfolio_bonds):
    """Test VaR calculation using Monte Carlo method"""
    result = risk_manager.calculate_var(
        portfolio_bonds,
        confidence_level=0.95,
        time_horizon=1,
        method="monte_carlo",
    )

    assert "var_value" in result
    assert "var_percentage" in result
    assert result["method"] == "monte_carlo"


def test_calculate_var_with_weights(risk_manager, portfolio_bonds):
    """Test VaR calculation with custom weights"""
    weights = [0.3, 0.2, 0.2, 0.15, 0.15]

    result = risk_manager.calculate_var(
        portfolio_bonds,
        weights=weights,
        confidence_level=0.95,
        method="historical",
    )

    assert result["var_value"] >= 0


def test_calculate_var_invalid_method(risk_manager, portfolio_bonds):
    """Test VaR with invalid method"""
    with pytest.raises(ValueError, match="Unknown VaR method"):
        risk_manager.calculate_var(portfolio_bonds, method="invalid")


def test_calculate_credit_risk(risk_manager):
    """Test credit risk calculation"""
    bond = create_test_bond(credit_rating="BBB")

    risk = risk_manager.calculate_credit_risk(bond)

    assert "default_probability" in risk
    assert "recovery_rate" in risk
    assert "expected_loss" in risk
    assert "credit_spread" in risk
    assert risk["default_probability"] >= 0
    assert risk["recovery_rate"] > 0


def test_calculate_credit_risk_different_ratings(risk_manager):
    """Test credit risk for different ratings"""
    ratings = ["AAA", "AA", "BBB", "BB", "B"]

    for rating in ratings:
        bond = create_test_bond(credit_rating=rating)
        risk = risk_manager.calculate_credit_risk(bond)

        assert risk["default_probability"] >= 0
        assert risk["recovery_rate"] > 0


def test_calculate_interest_rate_sensitivity(risk_manager):
    """Test interest rate sensitivity calculation"""
    bond = create_test_bond()

    sensitivity = risk_manager.calculate_interest_rate_sensitivity(bond, rate_change=0.01)

    assert "duration" in sensitivity
    assert "convexity" in sensitivity
    assert "price_change_duration_pct" in sensitivity
    assert "price_change_full_pct" in sensitivity
    assert "new_price_duration" in sensitivity
    assert "new_price_full" in sensitivity


def test_calculate_var_different_confidence_levels(risk_manager, portfolio_bonds):
    """Test VaR calculation with different confidence levels"""
    for confidence in [0.90, 0.95, 0.99]:
        result = risk_manager.calculate_var(
            portfolio_bonds,
            confidence_level=confidence,
            method="historical",
        )

        assert result["confidence_level"] == confidence
        assert result["var_value"] >= 0


def test_get_default_probability(risk_manager):
    """Test default probability lookup"""
    prob_aaa = risk_manager._get_default_probability("AAA")
    prob_bbb = risk_manager._get_default_probability("BBB")
    prob_b = risk_manager._get_default_probability("B")

    assert prob_aaa < prob_bbb < prob_b
    assert prob_aaa >= 0


def test_get_recovery_rate(risk_manager):
    """Test recovery rate lookup"""
    rate_aaa = risk_manager._get_recovery_rate("AAA")
    rate_bbb = risk_manager._get_recovery_rate("BBB")

    assert rate_aaa > rate_bbb
    assert 0 < rate_aaa <= 1
    assert 0 < rate_bbb <= 1
