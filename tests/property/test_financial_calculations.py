"""
Property-Based Testing for Financial Calculations
Uses Hypothesis to test calculation properties

CRITICAL: Catches edge cases and ensures calculation correctness
"""

from datetime import datetime, timedelta

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bondtrader.core.bond_models import Bond, BondType
from bondtrader.core.bond_valuation import BondValuator


class TestBondValuationProperties:
    """Property-based tests for bond valuation calculations"""

    @given(
        face_value=st.floats(min_value=100, max_value=1e6),
        coupon_rate=st.floats(min_value=0, max_value=0.2),
        years_to_maturity=st.floats(min_value=0.1, max_value=30),
        current_price=st.floats(min_value=50, max_value=2000),
    )
    @settings(max_examples=100, deadline=5000)
    def test_ytm_properties(self, face_value, coupon_rate, years_to_maturity, current_price):
        """
        Property: YTM should always be positive and finite.
        """
        bond = Bond(
            bond_id="TEST",
            bond_type=BondType.CORPORATE,
            face_value=face_value,
            coupon_rate=coupon_rate * 100,  # Convert to percentage
            maturity_date=datetime.now() + timedelta(days=int(years_to_maturity * 365)),
            issue_date=datetime.now() - timedelta(days=365),
            current_price=current_price,
        )

        valuator = BondValuator()
        ytm = valuator.calculate_yield_to_maturity(bond)

        # Property 1: YTM should be positive
        assert ytm >= 0, f"YTM should be non-negative, got {ytm}"

        # Property 2: YTM should be finite
        assert abs(ytm) < 1e10, f"YTM should be finite, got {ytm}"

        # Property 3: YTM should be reasonable (less than 100%)
        assert ytm < 1.0, f"YTM should be less than 100%, got {ytm}"

    @given(
        face_value=st.floats(min_value=100, max_value=1e6),
        coupon_rate=st.floats(min_value=0, max_value=0.2),
        years_to_maturity=st.floats(min_value=0.1, max_value=30),
        risk_free_rate=st.floats(min_value=0, max_value=0.1),
    )
    @settings(max_examples=100, deadline=5000)
    def test_fair_value_properties(self, face_value, coupon_rate, years_to_maturity, risk_free_rate):
        """
        Property: Fair value should be positive and reasonable relative to face value.
        """
        bond = Bond(
            bond_id="TEST",
            bond_type=BondType.CORPORATE,
            face_value=face_value,
            coupon_rate=coupon_rate * 100,
            maturity_date=datetime.now() + timedelta(days=int(years_to_maturity * 365)),
            issue_date=datetime.now() - timedelta(days=365),
            current_price=face_value * 0.9,  # Reasonable price
        )

        valuator = BondValuator(risk_free_rate=risk_free_rate)
        fair_value = valuator.calculate_fair_value(bond)

        # Property 1: Fair value should be positive
        assert fair_value > 0, f"Fair value should be positive, got {fair_value}"

        # Property 2: Fair value should be finite
        assert abs(fair_value) < 1e12, f"Fair value should be finite, got {fair_value}"

        # Property 3: Fair value should be reasonable (between 10% and 500% of face value)
        ratio = fair_value / face_value
        assert 0.1 < ratio < 5.0, f"Fair value ratio should be reasonable, got {ratio}"

    @given(
        face_value=st.floats(min_value=100, max_value=1e6),
        coupon_rate=st.floats(min_value=0, max_value=0.2),
        years_to_maturity=st.floats(min_value=0.1, max_value=30),
    )
    @settings(max_examples=50, deadline=5000)
    def test_duration_properties(self, face_value, coupon_rate, years_to_maturity):
        """
        Property: Duration should be positive and less than time to maturity.
        """
        bond = Bond(
            bond_id="TEST",
            bond_type=BondType.CORPORATE,
            face_value=face_value,
            coupon_rate=coupon_rate * 100,
            maturity_date=datetime.now() + timedelta(days=int(years_to_maturity * 365)),
            issue_date=datetime.now() - timedelta(days=365),
            current_price=face_value * 0.95,
        )

        valuator = BondValuator()
        ytm = valuator.calculate_yield_to_maturity(bond)
        duration = valuator.calculate_duration(bond, ytm)

        # Property 1: Duration should be positive
        assert duration >= 0, f"Duration should be non-negative, got {duration}"

        # Property 2: Duration should be less than or equal to time to maturity
        assert duration <= years_to_maturity * 1.1, f"Duration should be <= time to maturity, got {duration}"

    @given(
        face_value1=st.floats(min_value=100, max_value=1e6),
        face_value2=st.floats(min_value=100, max_value=1e6),
        coupon_rate=st.floats(min_value=0, max_value=0.2),
    )
    @settings(max_examples=50, deadline=5000)
    def test_fair_value_monotonicity(self, face_value1, face_value2, coupon_rate):
        """
        Property: Fair value should increase with face value (monotonicity).
        """
        if face_value1 >= face_value2:
            return  # Skip if not ordered

        bond1 = Bond(
            bond_id="TEST1",
            bond_type=BondType.CORPORATE,
            face_value=face_value1,
            coupon_rate=coupon_rate * 100,
            maturity_date=datetime.now() + timedelta(days=1825),  # 5 years
            issue_date=datetime.now() - timedelta(days=365),
            current_price=face_value1 * 0.95,
        )

        bond2 = Bond(
            bond_id="TEST2",
            bond_type=BondType.CORPORATE,
            face_value=face_value2,
            coupon_rate=coupon_rate * 100,
            maturity_date=datetime.now() + timedelta(days=1825),  # 5 years
            issue_date=datetime.now() - timedelta(days=365),
            current_price=face_value2 * 0.95,
        )

        valuator = BondValuator()
        fair_value1 = valuator.calculate_fair_value(bond1)
        fair_value2 = valuator.calculate_fair_value(bond2)

        # Property: Fair value should increase with face value
        assert fair_value2 >= fair_value1, f"Fair value should be monotonic: {fair_value1} vs {fair_value2}"
