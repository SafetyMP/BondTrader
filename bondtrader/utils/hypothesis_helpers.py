"""
Hypothesis Testing Helpers
Property-based testing utilities for bond trading system
"""

from datetime import datetime, timedelta
from typing import List

# Optional Hypothesis for property-based testing
try:
    from hypothesis import given
    from hypothesis import strategies as st
    from hypothesis.extra.numpy import arrays

    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False

import numpy as np

from bondtrader.core.bond_models import Bond, BondType

if HAS_HYPOTHESIS:

    # Strategy generators for bond testing
    def bond_strategy():
        """Generate Bond objects for property-based testing"""
        return st.builds(
            Bond,
            bond_id=st.text(min_size=1, max_size=20),
            bond_type=st.sampled_from(list(BondType)),
            face_value=st.floats(min_value=100, max_value=1000000),
            coupon_rate=st.floats(min_value=0, max_value=20),
            maturity_date=st.datetimes(
                min_value=datetime.now(), max_value=datetime.now() + timedelta(days=365 * 30)
            ),
            issue_date=st.datetimes(
                min_value=datetime.now() - timedelta(days=365 * 10), max_value=datetime.now()
            ),
            current_price=st.floats(min_value=50, max_value=2000),
            credit_rating=st.sampled_from(["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]),
            issuer=st.text(min_size=1, max_size=50),
            frequency=st.integers(min_value=1, max_value=12),
            callable=st.booleans(),
            convertible=st.booleans(),
        )

    def bond_list_strategy(min_size: int = 1, max_size: int = 100):
        """Generate lists of bonds for property-based testing"""
        return st.lists(bond_strategy(), min_size=min_size, max_size=max_size)

    def float_strategy(min_value: float = 0.0, max_value: float = 1.0):
        """Generate float values for testing"""
        return st.floats(
            min_value=min_value, max_value=max_value, allow_nan=False, allow_infinity=False
        )

    def positive_float_strategy():
        """Generate positive float values"""
        return st.floats(min_value=0.001, max_value=1000.0, allow_nan=False, allow_infinity=False)

    def yield_strategy():
        """Generate yield values (0-20%)"""
        return st.floats(min_value=0.0, max_value=0.20, allow_nan=False, allow_infinity=False)

    def weights_strategy(n: int):
        """Generate portfolio weights that sum to 1"""
        return arrays(np.float64, (n,), elements=st.floats(min_value=0, max_value=1)).filter(
            lambda w: abs(np.sum(w) - 1.0) < 0.01  # Allow small floating point errors
        )

    def covariance_matrix_strategy(n: int):
        """Generate valid covariance matrix"""
        # Generate symmetric positive semi-definite matrix
        return st.builds(
            lambda x: np.dot(x, x.T),
            arrays(
                np.float64, (n, n), elements=st.floats(min_value=-1, max_value=1, allow_nan=False)
            ),
        )

else:
    # Fallback strategies if Hypothesis not available
    def bond_strategy():
        """Fallback - Hypothesis not available"""
        raise ImportError("Hypothesis not installed. Install with: pip install hypothesis")

    def bond_list_strategy(min_size: int = 1, max_size: int = 100):
        """Fallback - Hypothesis not available"""
        raise ImportError("Hypothesis not installed. Install with: pip install hypothesis")

    def float_strategy(min_value: float = 0.0, max_value: float = 1.0):
        """Fallback - Hypothesis not available"""
        raise ImportError("Hypothesis not installed. Install with: pip install hypothesis")

    def positive_float_strategy():
        """Fallback - Hypothesis not available"""
        raise ImportError("Hypothesis not installed. Install with: pip install hypothesis")

    def yield_strategy():
        """Fallback - Hypothesis not available"""
        raise ImportError("Hypothesis not installed. Install with: pip install hypothesis")

    def weights_strategy(n: int):
        """Fallback - Hypothesis not available"""
        raise ImportError("Hypothesis not installed. Install with: pip install hypothesis")

    def covariance_matrix_strategy(n: int):
        """Fallback - Hypothesis not available"""
        raise ImportError("Hypothesis not installed. Install with: pip install hypothesis")


def test_bond_properties(bond: Bond) -> bool:
    """
    Test basic properties that all bonds should satisfy

    This function can be used with property-based testing to validate
    that bond calculations maintain certain invariants

    Args:
        bond: Bond object to test

    Returns:
        True if all properties satisfied
    """
    # Property 1: Price must be positive
    assert bond.current_price > 0, "Price must be positive"

    # Property 2: Face value must be positive
    assert bond.face_value > 0, "Face value must be positive"

    # Property 3: Maturity must be after issue date
    assert bond.maturity_date > bond.issue_date, "Maturity must be after issue date"

    # Property 4: Time to maturity must be non-negative
    assert bond.time_to_maturity >= 0, "Time to maturity must be non-negative"

    # Property 5: Frequency must be positive
    assert bond.frequency > 0, "Frequency must be positive"

    # Property 6: Zero coupon bonds must have coupon_rate=0
    if bond.bond_type == BondType.ZERO_COUPON:
        assert bond.coupon_rate == 0, "Zero coupon bonds must have coupon_rate=0"

    return True


if HAS_HYPOTHESIS:

    @given(bond_strategy())
    def test_bond_properties_hypothesis(bond: Bond):
        """Property-based test for bond properties"""
        test_bond_properties(bond)
