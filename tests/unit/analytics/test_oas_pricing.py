"""
Unit tests for OAS pricing module
"""

import os
import sys

import pytest

pytestmark = pytest.mark.unit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bondtrader.analytics.oas_pricing import OASPricer
from fixtures.bond_factory import create_test_bond, create_multiple_bonds


@pytest.fixture
def oas_pricer():
    """Create OAS pricer instance"""
    return OASPricer()


def test_oas_pricer_initialization():
    """Test OAS pricer initialization"""
    pricer = OASPricer()
    assert pricer.valuator is not None


def test_calculate_oas(oas_pricer):
    """Test OAS calculation"""
    bond = create_test_bond()
    
    result = oas_pricer.calculate_oas(bond)
    
    assert "oas" in result
    assert "oas_bps" in result
    assert isinstance(result["oas"], float)


def test_calculate_oas_spread_bps(oas_pricer):
    """Test OAS in basis points"""
    bond = create_test_bond()
    
    result = oas_pricer.calculate_oas(bond)
    
    assert result["oas_bps"] == result["oas"] * 10000


def test_calculate_oas_callable_bond(oas_pricer):
    """Test OAS calculation for callable bond"""
    bond = create_test_bond()
    bond.callable = True
    
    result = oas_pricer.calculate_oas(bond, volatility=0.15)
    
    assert "oas" in result
    assert "option_value" in result or "error" not in result
