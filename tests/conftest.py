"""
Pytest configuration and shared fixtures
"""

import pytest
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bondtrader.core.bond_models import Bond, BondType
from bondtrader.core.bond_valuation import BondValuator
from bondtrader.core.arbitrage_detector import ArbitrageDetector


@pytest.fixture(scope="session")
def sample_bond():
    """Create a sample bond for testing (shared across tests)"""
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
        frequency=2
    )


@pytest.fixture(scope="session")
def valuator():
    """Create a bond valuator for testing (shared across tests)"""
    return BondValuator(risk_free_rate=0.03)


@pytest.fixture
def sample_bonds():
    """Create multiple sample bonds for testing"""
    now = datetime.now()
    return [
        Bond(
            bond_id=f"BOND-{i:03d}",
            bond_type=BondType.CORPORATE,
            face_value=1000,
            coupon_rate=4.0 + (i % 3),
            maturity_date=now + timedelta(days=365 * (2 + i % 5)),
            issue_date=now - timedelta(days=365),
            current_price=950 + (i * 10),
            credit_rating=["AAA", "AA", "A", "BBB", "BB"][i % 5],
            issuer=f"Test Corp {i}",
            frequency=2
        )
        for i in range(5)
    ]
