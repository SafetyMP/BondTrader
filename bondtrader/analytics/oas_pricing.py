"""
Option-Adjusted Spread (OAS) Pricing Module
Implements binomial tree pricing for bonds with embedded options
Industry-standard approach used by Goldman Sachs, JPMorgan, etc.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import brentq

# Optional Numba for JIT compilation
try:
    from numba import jit

    from bondtrader.utils.numba_helpers import HAS_NUMBA as HAS_NUMBA_HELPERS
    from bondtrader.utils.numba_helpers import (
        binomial_tree_backward_step,
        binomial_tree_discount,
    )

    HAS_NUMBA = HAS_NUMBA_HELPERS
except ImportError:
    HAS_NUMBA = False

from bondtrader.core.bond_models import Bond, BondType
from bondtrader.core.bond_valuation import BondValuator
from bondtrader.utils.utils import logger


class OASPricer:
    """
    Option-Adjusted Spread (OAS) pricing engine
    Uses binomial tree model for bonds with embedded options (callable, putable)
    """

    def __init__(self, valuator: BondValuator = None, num_steps: int = 100):
        """
        Initialize OAS pricer

        Args:
            valuator: Bond valuator instance
            num_steps: Number of steps in binomial tree
        """
        self.valuator = valuator if valuator else BondValuator()
        self.num_steps = num_steps

    def calculate_oas(
        self,
        bond: Bond,
        market_price: Optional[float] = None,
        volatility: float = 0.15,
        call_price: Optional[float] = None,
        put_price: Optional[float] = None,
    ) -> Dict:
        """
        Calculate Option-Adjusted Spread (OAS)

        OAS is the spread added to the risk-free rate curve that makes the
        theoretical value equal to the market price when accounting for embedded options

        Args:
            bond: Bond object (must have callable or putable set)
            market_price: Market price (uses bond.current_price if None)
            volatility: Interest rate volatility (as decimal, e.g., 0.15 for 15%)
            call_price: Call price if callable (uses face value if None)
            put_price: Put price if putable (uses face value if None)

        Returns:
            Dictionary with OAS and details
        """
        if not (bond.callable or bond.convertible):
            # For non-callable bonds, OAS equals Z-spread
            return self._calculate_z_spread(bond, market_price)

        price = market_price if market_price is not None else bond.current_price
        time_to_maturity = bond.time_to_maturity

        if time_to_maturity <= 0:
            return {"oas": 0, "error": "Bond has matured"}

        # Set default exercise prices
        call_exercise_price = call_price if call_price is not None else bond.face_value
        put_exercise_price = put_price if put_price is not None else bond.face_value

        # Build binomial tree and calculate OAS
        try:
            oas = self._solve_oas(bond, price, volatility, call_exercise_price, put_exercise_price)

            # Calculate option value
            option_free_value = self.valuator.calculate_fair_value(bond)
            option_adjusted_value = self._binomial_price(bond, oas, volatility, call_exercise_price, put_exercise_price)
            option_value = option_free_value - option_adjusted_value

            return {
                "oas": oas,
                "oas_bps": oas * 10000,  # Basis points
                "market_price": price,
                "option_free_value": option_free_value,
                "option_adjusted_value": option_adjusted_value,
                "option_value": option_value,
                "volatility": volatility,
                "callable": bond.callable,
                "putable": bond.convertible,  # Using convertible flag for putable
            }
        except Exception as e:
            logger.error(f"Error calculating OAS: {e}")
            return {"oas": 0, "error": str(e)}

    def _solve_oas(self, bond: Bond, target_price: float, volatility: float, call_price: float, put_price: float) -> float:
        """Solve for OAS that makes binomial price equal to market price"""

        def price_difference(oas):
            calculated_price = self._binomial_price(bond, oas, volatility, call_price, put_price)
            return calculated_price - target_price

        # Use Brent's method for robust root finding
        try:
            oas = brentq(price_difference, -0.10, 0.20, maxiter=100)
            return oas
        except ValueError:
            # Fallback to bisection if Brent's method fails
            return self._bisect_oas(price_difference, -0.10, 0.20)

    def _bisect_oas(self, func, low: float, high: float, tol: float = 1e-6) -> float:
        """Bisection method for finding OAS"""
        for _ in range(50):
            mid = (low + high) / 2
            if abs(func(mid)) < tol:
                return mid
            if func(mid) * func(low) < 0:
                high = mid
            else:
                low = mid
        return (low + high) / 2

    def _binomial_price(self, bond: Bond, oas: float, volatility: float, call_price: float, put_price: float) -> float:
        """
        Calculate bond price using binomial tree with OAS

        Args:
            bond: Bond object
            oas: Option-adjusted spread
            volatility: Interest rate volatility
            call_price: Call exercise price
            put_price: Put exercise price

        Returns:
            Bond price from binomial tree
        """
        time_to_maturity = bond.time_to_maturity
        if time_to_maturity <= 0:
            return bond.face_value

        dt = time_to_maturity / self.num_steps
        rf_rate = self.valuator.risk_free_rate

        # Build interest rate tree
        # Simple model: lognormal short rate process
        rates = self._build_rate_tree(rf_rate, volatility, dt)

        # Calculate bond values backward through tree
        # Start at maturity
        values = np.full(self.num_steps + 1, bond.face_value)

        # Work backwards through tree
        for step in range(self.num_steps - 1, -1, -1):
            coupon_payment = (bond.coupon_rate / 100) * bond.face_value * dt

            new_values = np.zeros(step + 1)

            for node in range(step + 1):
                # Discounted expected value (risk-neutral pricing)
                if step < self.num_steps - 1:
                    # Expected value from next period
                    up_value = values[node + 1]
                    down_value = values[node]

                    # Risk-neutral probability (simplified: 0.5 each)
                    prob_up = 0.5
                    prob_down = 0.5

                    expected_value = prob_up * up_value + prob_down * down_value
                else:
                    expected_value = bond.face_value

                # Discount using current rate + OAS
                current_rate = rates[step][node] if step < len(rates) else rf_rate
                discount_rate = current_rate + oas
                discount_factor = np.exp(-discount_rate * dt)

                bond_value = (expected_value + coupon_payment) * discount_factor

                # Apply option features
                if bond.callable:
                    # Call option: issuer can call at call_price
                    bond_value = min(bond_value, call_price)

                if bond.convertible:  # Using as putable flag
                    # Put option: holder can put at put_price
                    bond_value = max(bond_value, put_price)

                new_values[node] = bond_value

            values = new_values

        return values[0] if len(values) > 0 else bond.face_value

    def _build_rate_tree(self, initial_rate: float, volatility: float, dt: float) -> List[List[float]]:
        """Build binomial interest rate tree"""
        rates = []

        # Simple model: rates evolve as geometric random walk
        for step in range(self.num_steps):
            step_rates = []
            for node in range(step + 1):
                # Up and down moves
                up_factor = np.exp(volatility * np.sqrt(dt))
                down_factor = 1 / up_factor

                # Number of up moves
                num_ups = node
                num_downs = step - node

                rate = initial_rate * (up_factor**num_ups) * (down_factor**num_downs)
                step_rates.append(rate)

            rates.append(step_rates)

        return rates

    def _calculate_z_spread(self, bond: Bond, market_price: Optional[float] = None) -> Dict:
        """Calculate Z-spread for non-callable bonds (OAS = Z-spread)"""
        from bondtrader.analytics.advanced_analytics import AdvancedAnalytics

        analytics = AdvancedAnalytics(self.valuator)
        z_spread_result = analytics.calculate_z_spread(bond, market_price)

        return {
            "oas": z_spread_result.get("z_spread", 0),
            "oas_bps": z_spread_result.get("z_spread_bps", 0),
            "market_price": market_price if market_price else bond.current_price,
            "option_free_value": z_spread_result.get("market_price", bond.current_price),
            "option_adjusted_value": z_spread_result.get("market_price", bond.current_price),
            "option_value": 0,
            "note": "Non-callable bond: OAS equals Z-spread",
        }

    def calculate_option_value(self, bond: Bond, volatility: float = 0.15, call_price: Optional[float] = None) -> Dict:
        """
        Calculate the value of embedded options

        Args:
            bond: Bond object
            volatility: Interest rate volatility
            call_price: Call exercise price

        Returns:
            Dictionary with option values
        """
        if not bond.callable:
            return {"call_option_value": 0, "note": "Bond is not callable"}

        # Price without option (straight bond)
        option_free_value = self.valuator.calculate_fair_value(bond)

        # Price with option (using OAS)
        oas_result = self.calculate_oas(bond, bond.current_price, volatility, call_price)

        option_adjusted_value = oas_result.get("option_adjusted_value", option_free_value)
        option_value = option_free_value - option_adjusted_value

        return {
            "call_option_value": option_value,
            "option_free_value": option_free_value,
            "option_adjusted_value": option_adjusted_value,
            "oas": oas_result.get("oas", 0),
            "oas_bps": oas_result.get("oas_bps", 0),
        }

    def price_callable_bond(self, bond: Bond, volatility: float = 0.15, call_schedule: Optional[List[Dict]] = None) -> Dict:
        """
        Price callable bond with call schedule

        Args:
            bond: Callable bond
            volatility: Interest rate volatility
            call_schedule: List of call dates and prices

        Returns:
            Pricing results
        """
        if not bond.callable:
            return {"error": "Bond is not callable", "value": self.valuator.calculate_fair_value(bond)}

        # Simplified: use single call price
        # In production, would handle full call schedule
        call_price = bond.face_value  # Default to par

        if call_schedule:
            # Find first callable date
            first_call = min(call_schedule, key=lambda x: x.get("date", datetime.max))
            call_price = first_call.get("price", bond.face_value)

        oas_result = self.calculate_oas(bond, bond.current_price, volatility, call_price)

        return {
            "theoretical_value": oas_result.get("option_adjusted_value"),
            "market_price": bond.current_price,
            "oas": oas_result.get("oas"),
            "oas_bps": oas_result.get("oas_bps"),
            "option_value": oas_result.get("option_value"),
            "call_price": call_price,
        }
