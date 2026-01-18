"""
Smoke tests for critical functionality
These tests must pass before any deployment
Should run in <30 seconds total
"""

import os
import sys
from datetime import datetime, timedelta

import pytest

# Add parent directories to path for fixture imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fixtures.bond_factory import create_test_bond

from bondtrader.config import get_config
from bondtrader.core.arbitrage_detector import ArbitrageDetector
from bondtrader.core.bond_models import Bond, BondType
from bondtrader.core.bond_valuation import BondValuator


@pytest.mark.smoke
def test_bond_valuation_works():
    """Verify bond valuation doesn't crash and returns positive value"""
    valuator = BondValuator()
    bond = create_test_bond()

    fair_value = valuator.calculate_fair_value(bond)

    assert fair_value > 0
    assert isinstance(fair_value, float)


@pytest.mark.smoke
def test_arbitrage_detection_works():
    """Verify arbitrage detection doesn't crash"""
    detector = ArbitrageDetector()
    bonds = [create_test_bond(bond_id=f"BOND-{i}") for i in range(3)]

    opportunities = detector.find_arbitrage_opportunities(bonds)

    assert isinstance(opportunities, list)


@pytest.mark.smoke
def test_bond_creation_works():
    """Verify bond creation doesn't crash"""
    bond = create_test_bond()

    assert bond.bond_id == "TEST-001"
    assert bond.face_value == 1000
    assert bond.coupon_rate == 5.0


@pytest.mark.smoke
def test_config_initialization_works():
    """Verify config initialization doesn't crash"""
    config = get_config()

    assert config.default_risk_free_rate > 0
    assert config.ml_model_type is not None


@pytest.mark.smoke
def test_yield_calculation_works():
    """Verify YTM calculation doesn't crash"""
    valuator = BondValuator()
    bond = create_test_bond()

    ytm = valuator.calculate_yield_to_maturity(bond)

    assert isinstance(ytm, float)
    assert ytm > 0


@pytest.mark.smoke
def test_duration_calculation_works():
    """Verify duration calculation doesn't crash"""
    valuator = BondValuator()
    bond = create_test_bond()

    ytm = valuator.calculate_yield_to_maturity(bond)
    duration = valuator.calculate_duration(bond, ytm)

    assert isinstance(duration, float)
    assert duration > 0


@pytest.mark.smoke
def test_convexity_calculation_works():
    """Verify convexity calculation doesn't crash"""
    valuator = BondValuator()
    bond = create_test_bond()

    ytm = valuator.calculate_yield_to_maturity(bond)
    convexity = valuator.calculate_convexity(bond, ytm)

    assert isinstance(convexity, float)


@pytest.mark.smoke
def test_bond_characteristics_works():
    """Verify bond characteristics extraction doesn't crash"""
    bond = create_test_bond()

    characteristics = bond.get_bond_characteristics()

    assert isinstance(characteristics, dict)
    assert "coupon_rate" in characteristics
    assert "time_to_maturity" in characteristics
