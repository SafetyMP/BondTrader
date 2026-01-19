"""
Tests for numba helpers
"""

import numpy as np
import pytest

from bondtrader.utils.numba_helpers import (
    binomial_tree_backward_step,
    binomial_tree_discount,
    enable_numba_jit,
    monte_carlo_price_simulation,
    vectorized_coupon_derivative,
    vectorized_coupon_pv,
)


@pytest.mark.unit
class TestNumbaHelpers:
    """Test numba helper functions"""

    def test_monte_carlo_price_simulation(self):
        """Test Monte Carlo price simulation"""
        result = monte_carlo_price_simulation(
            current_price=1000.0,
            duration=5.0,
            convexity=25.0,
            yield_change=0.01,
            face_value=1000.0,
        )
        assert isinstance(result, float)
        assert result > 0

    def test_binomial_tree_discount(self):
        """Test binomial tree discount"""
        result = binomial_tree_discount(value=1000.0, rate=0.03, dt=0.25)
        assert isinstance(result, float)
        assert result < 1000.0  # Discounted value should be less

    def test_binomial_tree_backward_step(self):
        """Test binomial tree backward step"""
        result = binomial_tree_backward_step(
            up_value=1050.0,
            down_value=950.0,
            coupon=25.0,
            rate=0.03,
            dt=0.25,
            prob_up=0.5,
        )
        assert isinstance(result, float)
        assert result > 0

    def test_vectorized_coupon_pv(self):
        """Test vectorized coupon present value"""
        result = vectorized_coupon_pv(coupon_payment=25.0, periods=10, ytm=0.05, frequency=2.0)
        assert isinstance(result, float)
        assert result > 0

    def test_vectorized_coupon_derivative(self):
        """Test vectorized coupon derivative"""
        result = vectorized_coupon_derivative(
            coupon_payment=25.0, periods=10, ytm=0.05, frequency=2.0
        )
        assert isinstance(result, float)

    def test_enable_numba_jit_decorator(self):
        """Test numba JIT decorator"""

        @enable_numba_jit
        def test_func(x):
            return x * 2

        result = test_func(21)
        assert result == 42
