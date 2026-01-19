"""
Unit tests for enhanced credit risk module
"""

import os
import sys

import pytest

pytestmark = pytest.mark.unit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fixtures.bond_factory import create_multiple_bonds, create_test_bond

from bondtrader.core.bond_valuation import BondValuator
from bondtrader.risk.credit_risk_enhanced import CreditRiskEnhanced


@pytest.fixture
def credit_risk_enhanced():
    """Create enhanced credit risk analyzer instance"""
    return CreditRiskEnhanced()


def test_credit_risk_enhanced_initialization():
    """Test enhanced credit risk initialization"""
    analyzer = CreditRiskEnhanced()
    assert analyzer.valuator is not None
    assert analyzer.migration_matrix is not None


def test_credit_risk_enhanced_with_valuator():
    """Test enhanced credit risk with custom valuator"""
    valuator = BondValuator(risk_free_rate=0.04)
    analyzer = CreditRiskEnhanced(valuator=valuator)
    assert analyzer.valuator.risk_free_rate == 0.04


def test_merton_structural_model(credit_risk_enhanced):
    """Test Merton structural model"""
    bond = create_test_bond(credit_rating="BBB")

    result = credit_risk_enhanced.merton_structural_model(bond)

    assert "default_probability" in result
    assert "distance_to_default" in result
    assert "recovery_rate" in result
    assert result["model"] == "Merton"
    assert 0 <= result["default_probability"] <= 1


def test_credit_migration_analysis(credit_risk_enhanced):
    """Test credit migration analysis"""
    bond = create_test_bond(credit_rating="BBB")

    result = credit_risk_enhanced.credit_migration_analysis(
        bond, time_horizon=1.0, num_scenarios=100
    )

    assert "value_distribution" in result
    assert "mean_value" in result
    assert "migration_probabilities" in result
    assert "current_rating" in result


def test_get_recovery_rate_enhanced(credit_risk_enhanced):
    """Test recovery rate lookup"""
    rate_aaa = credit_risk_enhanced.risk_manager._get_recovery_rate("AAA")
    rate_bbb = credit_risk_enhanced.risk_manager._get_recovery_rate("BBB")

    assert rate_aaa > rate_bbb
    assert 0 < rate_aaa <= 1
