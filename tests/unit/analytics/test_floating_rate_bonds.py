"""
Unit tests for floating rate bonds module
"""

import os
import sys

import pytest

pytestmark = pytest.mark.unit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fixtures.bond_factory import create_test_bond

from bondtrader.analytics.floating_rate_bonds import FloatingRateBondPricer
from bondtrader.core.bond_models import BondType


@pytest.fixture
def floating_rate_pricer():
    """Create floating rate bond pricer instance"""
    return FloatingRateBondPricer()


def test_floating_rate_pricer_initialization():
    """Test floating rate pricer initialization"""
    pricer = FloatingRateBondPricer()
    assert pricer.valuator is not None


def test_price_floating_rate_bond(floating_rate_pricer):
    """Test pricing floating rate bond"""
    from datetime import datetime, timedelta

    bond = create_test_bond()
    next_reset = datetime.now() + timedelta(days=90)

    result = floating_rate_pricer.price_floating_rate_bond(
        bond, next_reset_date=next_reset, spread=0.01
    )

    # Result should contain pricing information (clean_price, dirty_price, or price)
    assert (
        "clean_price" in result or "dirty_price" in result or "price" in result or "value" in result
    )
    price = (
        result.get("price")
        or result.get("clean_price")
        or result.get("dirty_price")
        or result.get("value", 0)
    )
    assert price > 0


def test_calculate_floating_coupon(floating_rate_pricer):
    """Test floating coupon calculation"""
    from datetime import datetime, timedelta

    bond = create_test_bond()
    reset_date = datetime.now() + timedelta(days=30)

    result = floating_rate_pricer.calculate_floating_coupon(
        bond, reset_date, reference_rate=0.03, spread=0.01
    )

    assert "coupon_rate" in result
    assert "coupon_payment" in result
    assert result["coupon_rate"] > 0
