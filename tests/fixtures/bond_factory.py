"""
Factory functions for creating test bonds
Reduces duplication and provides reusable test data
"""

from datetime import datetime, timedelta
from typing import List, Optional

from bondtrader.core.bond_models import Bond, BondType


def create_test_bond(**overrides) -> Bond:
    """
    Factory to create test bonds with sensible defaults
    
    Args:
        **overrides: Any bond parameters to override defaults
        
    Returns:
        Bond instance with specified or default parameters
        
    Example:
        >>> bond = create_test_bond(bond_id="CUSTOM-001", coupon_rate=6.0)
        >>> assert bond.coupon_rate == 6.0
    """
    defaults = {
        "bond_id": "TEST-001",
        "bond_type": BondType.CORPORATE,
        "face_value": 1000,
        "coupon_rate": 5.0,
        "maturity_date": datetime.now() + timedelta(days=1825),  # 5 years
        "issue_date": datetime.now() - timedelta(days=365),
        "current_price": 950,
        "credit_rating": "BBB",
        "issuer": "Test Corp",
        "frequency": 2,
    }
    defaults.update(overrides)
    return Bond(**defaults)


def create_multiple_bonds(count: int = 5, bond_type: Optional[BondType] = None) -> List[Bond]:
    """
    Create multiple test bonds with varied characteristics
    
    Args:
        count: Number of bonds to create
        bond_type: Optional bond type (if None, varies by index)
        
    Returns:
        List of Bond instances
    """
    now = datetime.now()
    ratings = ["AAA", "AA", "A", "BBB", "BB"]
    bond_types = [
        BondType.TREASURY,
        BondType.CORPORATE,
        BondType.FIXED_RATE,
        BondType.HIGH_YIELD,
        BondType.ZERO_COUPON,
    ]

    bonds = []
    for i in range(count):
        bonds.append(
            create_test_bond(
                bond_id=f"BOND-{i:03d}",
                bond_type=bond_type if bond_type else bond_types[i % len(bond_types)],
                coupon_rate=4.0 + (i % 3),
                maturity_date=now + timedelta(days=365 * (2 + i % 5)),
                issue_date=now - timedelta(days=365),
                current_price=950 + (i * 10),
                credit_rating=ratings[i % len(ratings)],
                issuer=f"Test Corp {i}",
            )
        )

    return bonds


def create_treasury_bond(**overrides) -> Bond:
    """Create a treasury bond with defaults"""
    return create_test_bond(
        bond_type=BondType.TREASURY,
        credit_rating="AAA",
        issuer="US Treasury",
        coupon_rate=3.5,
        **overrides,
    )


def create_high_yield_bond(**overrides) -> Bond:
    """Create a high yield bond with defaults"""
    return create_test_bond(
        bond_type=BondType.HIGH_YIELD,
        credit_rating="BB",
        coupon_rate=8.0,
        current_price=900,
        **overrides,
    )


def create_zero_coupon_bond(**overrides) -> Bond:
    """Create a zero coupon bond with defaults"""
    return create_test_bond(
        bond_type=BondType.ZERO_COUPON,
        coupon_rate=0.0,
        **overrides,
    )
