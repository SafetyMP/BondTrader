"""
Numba JIT Helper Functions
JIT-compiled numerical functions for performance-critical computations
"""

import numpy as np

# Optional Numba for JIT compilation
try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Fallback decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


@jit(nopython=True, cache=True)
def monte_carlo_price_simulation(
    current_price: float,
    duration: float,
    convexity: float,
    yield_change: float,
    face_value: float,
) -> float:
    """
    JIT-compiled helper for Monte Carlo price simulation
    
    Calculates new bond price given yield change using duration and convexity
    
    Args:
        current_price: Current bond price
        duration: Bond duration
        convexity: Bond convexity
        yield_change: Yield change (as decimal)
        face_value: Face value
        
    Returns:
        New bond price
    """
    # Price change using duration and convexity
    price_change_pct = -duration * yield_change + 0.5 * convexity * (yield_change**2)
    new_price = current_price * (1 + price_change_pct)
    return new_price


@jit(nopython=True, cache=True)
def binomial_tree_discount(
    value: float,
    rate: float,
    dt: float,
) -> float:
    """
    JIT-compiled discount factor calculation
    
    Args:
        value: Future value
        rate: Discount rate
        dt: Time step
        
    Returns:
        Present value
    """
    discount_factor = np.exp(-rate * dt)
    return value * discount_factor


@jit(nopython=True, cache=True)
def binomial_tree_backward_step(
    up_value: float,
    down_value: float,
    coupon: float,
    rate: float,
    dt: float,
    prob_up: float = 0.5,
) -> float:
    """
    JIT-compiled binomial tree backward induction step
    
    Args:
        up_value: Value if rates go up
        down_value: Value if rates go down
        coupon: Coupon payment
        rate: Current discount rate
        dt: Time step
        prob_up: Probability of up move (default 0.5)
        
    Returns:
        Discounted expected value
    """
    prob_down = 1.0 - prob_up
    expected_value = prob_up * up_value + prob_down * down_value
    total_value = expected_value + coupon
    discount_factor = np.exp(-rate * dt)
    return total_value * discount_factor


@jit(nopython=True, cache=True)
def vectorized_coupon_pv(
    coupon_payment: float,
    periods: int,
    ytm: float,
    frequency: float,
) -> float:
    """
    JIT-compiled present value of coupon payments (vectorized)
    
    Args:
        coupon_payment: Coupon payment per period
        periods: Number of periods
        ytm: Yield to maturity
        frequency: Payment frequency
        
    Returns:
        Present value of all coupon payments
    """
    if periods == 0:
        return 0.0
    
    total = 0.0
    rate = 1.0 + ytm / frequency
    
    for i in range(1, periods + 1):
        discount = rate ** i
        total += coupon_payment / discount
    
    return total


@jit(nopython=True, cache=True)
def vectorized_coupon_derivative(
    coupon_payment: float,
    periods: int,
    ytm: float,
    frequency: float,
) -> float:
    """
    JIT-compiled derivative of coupon PV for Newton-Raphson
    
    Args:
        coupon_payment: Coupon payment per period
        periods: Number of periods
        ytm: Yield to maturity
        frequency: Payment frequency
        
    Returns:
        Derivative of coupon present value
    """
    if periods == 0:
        return 0.0
    
    total = 0.0
    rate = 1.0 + ytm / frequency
    
    for i in range(1, periods + 1):
        derivative_term = i / (frequency * (rate ** (i + 1)))
        total += coupon_payment * derivative_term
    
    return -total


def enable_numba_jit(func):
    """
    Decorator to conditionally enable Numba JIT compilation
    
    Usage:
        @enable_numba_jit
        def my_function(...):
            ...
    
    If Numba is not available, function runs normally
    """
    if HAS_NUMBA:
        return jit(nopython=True, cache=True)(func)
    else:
        return func
