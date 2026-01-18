"""
Unit tests for bond valuation module
"""

import os
import sys
from datetime import datetime, timedelta

import pytest

pytestmark = pytest.mark.unit

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bondtrader.core.bond_models import Bond, BondType
from bondtrader.core.bond_valuation import BondValuator


@pytest.fixture
def sample_bond():
    """Create a sample bond for testing"""
    return Bond(
        bond_id="TEST-001",
        bond_type=BondType.CORPORATE,
        face_value=1000,
        coupon_rate=5.0,
        maturity_date=datetime.now() + timedelta(days=1825),  # 5 years
        issue_date=datetime.now() - timedelta(days=365),
        current_price=950,
        credit_rating="BBB",
        issuer="Test Corp",
        frequency=2,
    )


@pytest.fixture
def valuator():
    """Create a bond valuator for testing"""
    return BondValuator(risk_free_rate=0.03)


def test_fair_value_calculation(sample_bond, valuator):
    """Test fair value calculation"""
    fair_value = valuator.calculate_fair_value(sample_bond)
    assert fair_value > 0
    assert isinstance(fair_value, float)


def test_ytm_calculation(sample_bond, valuator):
    """Test yield to maturity calculation"""
    ytm = valuator.calculate_yield_to_maturity(sample_bond)
    assert ytm > 0
    assert isinstance(ytm, float)


def test_duration_calculation(sample_bond, valuator):
    """Test duration calculation"""
    duration = valuator.calculate_duration(sample_bond)
    assert duration > 0
    assert isinstance(duration, float)


def test_convexity_calculation(sample_bond, valuator):
    """Test convexity calculation"""
    convexity = valuator.calculate_convexity(sample_bond)
    assert convexity > 0
    assert isinstance(convexity, float)


def test_price_mismatch(sample_bond, valuator):
    """Test price mismatch calculation"""
    mismatch = valuator.calculate_price_mismatch(sample_bond)
    assert "fair_value" in mismatch
    assert "market_price" in mismatch
    assert "mismatch_percentage" in mismatch
    assert mismatch["market_price"] == sample_bond.current_price


def test_zero_coupon_bond():
    """Test zero coupon bond valuation"""
    zero_coupon = Bond(
        bond_id="ZC-001",
        bond_type=BondType.ZERO_COUPON,
        face_value=1000,
        coupon_rate=0.0,
        maturity_date=datetime.now() + timedelta(days=1825),
        issue_date=datetime.now() - timedelta(days=365),
        current_price=800,
        credit_rating="AAA",
    )

    valuator = BondValuator()
    fair_value = valuator.calculate_fair_value(zero_coupon)
    ytm = valuator.calculate_yield_to_maturity(zero_coupon)

    assert fair_value > 0
    assert ytm > 0


def test_matured_bond(sample_bond, valuator):
    """Test bond that has matured"""
    matured_bond = Bond(
        bond_id="MAT-001",
        bond_type=BondType.CORPORATE,
        face_value=1000,
        coupon_rate=5.0,
        maturity_date=datetime.now() - timedelta(days=1),
        issue_date=datetime.now() - timedelta(days=3650),
        current_price=1000,
        credit_rating="BBB",
    )

    fair_value = valuator.calculate_fair_value(matured_bond)
    assert fair_value == matured_bond.face_value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
