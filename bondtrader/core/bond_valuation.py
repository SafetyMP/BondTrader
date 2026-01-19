"""
Bond Valuation Algorithms
Implements various valuation methods for different bond types
"""

from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any, Dict, List, Optional

import numpy as np

from bondtrader.core.bond_models import Bond, BondType
from bondtrader.utils.utils import logger


class BondValuator:
    """Core bond valuation engine with calculation caching for performance"""

    def __init__(self, risk_free_rate: float = 0.03, enable_caching: bool = True):
        """
        Initialize valuator with risk-free rate (default 3%)

        Args:
            risk_free_rate: Annual risk-free rate as decimal (e.g., 0.03 for 3%)
            enable_caching: Enable caching for expensive calculations (default: True)
        """
        self.risk_free_rate = risk_free_rate
        self.enable_caching = enable_caching
        # Cache for calculations - using dict for manual cache management
        self._calculation_cache = {} if enable_caching else None
        # Cache size limit to prevent memory issues
        self._cache_size_limit = 10000

    def calculate_yield_to_maturity(
        self,
        bond: Bond,
        market_price: Optional[float] = None,
        tolerance: float = 1e-6,
        max_iterations: int = 100,
    ) -> float:
        """
        Calculate Yield to Maturity using Newton-Raphson method
        Caches results for performance optimization

        Args:
            bond: Bond object to calculate YTM for
            market_price: Current market price (uses bond.current_price if None)
            tolerance: Convergence tolerance for Newton-Raphson iteration
            max_iterations: Maximum iterations for Newton-Raphson algorithm

        Returns:
            YTM as decimal (e.g., 0.05 for 5%)

        Raises:
            ValueError: If bond has invalid maturity date or negative values
            TypeError: If bond is not a Bond instance
        """
        if not isinstance(bond, Bond):
            raise TypeError(f"Expected Bond instance, got {type(bond)}")

        price = market_price if market_price is not None else bond.current_price

        if price <= 0:
            raise ValueError(f"Market price must be positive, got {price}")

        # Check cache if enabled
        if self.enable_caching and self._calculation_cache is not None:
            cache_key = (bond.bond_id, price, "ytm")
            if cache_key in self._calculation_cache:
                return self._calculation_cache[cache_key]

        time_to_maturity = bond.time_to_maturity

        if time_to_maturity <= 0:
            return 0.0

        if bond.bond_type == BondType.ZERO_COUPON:
            # Zero coupon: Price = Face / (1 + ytm)^T
            ytm = (bond.face_value / price) ** (1 / time_to_maturity) - 1
            result = max(0, ytm)
            # Cache result
            if self.enable_caching and self._calculation_cache is not None:
                self._update_cache(cache_key, result)
            return result

        # Initial guess for YTM
        ytm = self.risk_free_rate + bond.coupon_rate / 100

        # Newton-Raphson iteration - optimized with vectorized calculations
        periods = int(time_to_maturity * bond.frequency)
        coupon_payment = (bond.coupon_rate / 100) * bond.face_value / bond.frequency
        freq_ytm = bond.frequency

        for _ in range(max_iterations):
            # Vectorized calculation for present value of cash flows (optimized)
            if periods > 0:
                periods_array = np.arange(1, periods + 1)
                discount_factors = (1 + ytm / freq_ytm) ** periods_array
                pv_coupons = np.sum(coupon_payment / discount_factors)
                pv_face = bond.face_value / ((1 + ytm / freq_ytm) ** periods)
            else:
                pv_coupons = 0
                pv_face = bond.face_value

            calculated_price = pv_coupons + pv_face

            # Vectorized derivative calculation (optimized)
            if periods > 0:
                derivative_array = periods_array / (freq_ytm * ((1 + ytm / freq_ytm) ** (periods_array + 1)))
                derivative = -np.sum(coupon_payment * derivative_array)
                derivative -= (periods * bond.face_value) / (freq_ytm * ((1 + ytm / freq_ytm) ** (periods + 1)))
            else:
                derivative = -1e-10

            # Update YTM
            price_diff = calculated_price - price
            if abs(price_diff) < tolerance:
                break

            if abs(derivative) < 1e-10:
                break

            ytm -= price_diff / derivative

            # Ensure YTM is positive
            ytm = max(0.001, ytm)

        # Cache result
        if self.enable_caching and self._calculation_cache is not None:
            self._update_cache((bond.bond_id, price, "ytm"), ytm)

        return ytm

    def calculate_fair_value(
        self,
        bond: Bond,
        required_yield: Optional[float] = None,
        risk_free_rate: Optional[float] = None,
    ) -> float:
        """
        Calculate theoretical fair value of a bond with comprehensive validation.

        CRITICAL: All inputs and outputs are validated to prevent incorrect valuations.

        Args:
            bond: Bond object
            required_yield: Required yield (if None, uses risk-free + spread)
            risk_free_rate: Risk-free rate override

        Returns:
            Fair value of the bond

        Raises:
            ValueError: If bond data is invalid or calculation produces invalid result
            CalculationError: If calculation fails or produces suspicious values
        """
        from bondtrader.core.exceptions import CalculationError, InvalidBondError

        # CRITICAL: Comprehensive input validation
        if bond.face_value <= 0:
            raise InvalidBondError(f"Face value must be positive, got {bond.face_value}")
        if bond.current_price <= 0:
            raise InvalidBondError(f"Current price must be positive, got {bond.current_price}")
        if bond.coupon_rate < 0 or bond.coupon_rate > 100:
            raise InvalidBondError(f"Coupon rate must be between 0 and 100, got {bond.coupon_rate}")
        if bond.frequency <= 0:
            raise InvalidBondError(f"Frequency must be positive, got {bond.frequency}")

        # Validate risk-free rate if provided
        if risk_free_rate is not None:
            if risk_free_rate < 0 or risk_free_rate > 1:
                raise InvalidBondError(f"Risk-free rate must be between 0 and 1, got {risk_free_rate}")
        if required_yield is not None:
            if required_yield < 0 or required_yield > 1:
                raise InvalidBondError(f"Required yield must be between 0 and 1, got {required_yield}")

        rf_rate = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
        time_to_maturity = bond.time_to_maturity

        if time_to_maturity <= 0:
            # Bond is at or past maturity - return face value
            return bond.face_value

        if bond.bond_type == BondType.ZERO_COUPON:
            # Zero coupon bond pricing
            required_ytm = required_yield if required_yield is not None else rf_rate + 0.01
            fair_value = bond.face_value / ((1 + required_ytm) ** time_to_maturity)
            return fair_value

        if bond.bond_type == BondType.FLOATING_RATE:
            # Floating rate bonds trade near par
            # Use floating rate pricer if available
            try:
                from bondtrader.analytics.floating_rate_bonds import FloatingRateBondPricer

                fr_pricer = FloatingRateBondPricer(self)
                next_reset = bond.maturity_date - timedelta(days=182)  # Approximate
                fr_result = fr_pricer.price_floating_rate_bond(bond, next_reset, use_multi_curve=False)
                return fr_result.get("clean_price", bond.face_value)
            except ImportError:
                # Fallback: floating rate bonds trade near par
                return bond.face_value

        # Calculate required yield with credit spread
        if required_yield is None:
            # Credit spread based on rating
            spread = self._get_credit_spread(bond.credit_rating)
            required_ytm = rf_rate + spread
        else:
            required_ytm = required_yield

        # Calculate present value of coupon payments (vectorized)
        periods = int(time_to_maturity * bond.frequency)
        coupon_payment = (bond.coupon_rate / 100) * bond.face_value / bond.frequency

        if periods > 0:
            # Vectorized calculation for better performance
            periods_array = np.arange(1, periods + 1)
            discount_factors = (1 + required_ytm / bond.frequency) ** periods_array
            pv_coupons = np.sum(coupon_payment / discount_factors)
        else:
            pv_coupons = 0

        # Present value of face value
        pv_face = bond.face_value / ((1 + required_ytm / bond.frequency) ** periods) if periods > 0 else bond.face_value

        fair_value = pv_coupons + pv_face

        # CRITICAL: Output validation and anomaly detection
        if fair_value <= 0:
            raise CalculationError(f"Invalid fair value calculated: {fair_value}. Must be positive.")
        if not np.isfinite(fair_value):
            raise CalculationError(f"Fair value is not finite: {fair_value}")

        # Anomaly detection: Flag suspicious deviations (>50% from market price)
        deviation_pct = abs(fair_value - bond.current_price) / bond.current_price if bond.current_price > 0 else 0
        if deviation_pct > 0.5:  # More than 50% deviation
            logger.warning(
                f"Large deviation detected for bond {bond.bond_id}: "
                f"fair_value={fair_value:.2f}, market_price={bond.current_price:.2f}, "
                f"deviation={deviation_pct*100:.1f}%. This may require manual review."
            )
            # In production: trigger alert for manual review

        # Sanity check: Fair value should be reasonable relative to face value
        # (typically between 50% and 200% of face value for most bonds)
        if fair_value < bond.face_value * 0.1 or fair_value > bond.face_value * 5.0:
            logger.warning(
                f"Unusual fair value for bond {bond.bond_id}: "
                f"fair_value={fair_value:.2f}, face_value={bond.face_value:.2f}, "
                f"ratio={fair_value/bond.face_value:.2f}x"
            )

        return fair_value

    # Class-level constant for credit spreads (optimized - no dict lookup overhead)
    _CREDIT_SPREADS = {
        "AAA": 0.003,
        "AA+": 0.005,
        "AA": 0.007,
        "AA-": 0.010,
        "A+": 0.015,
        "A": 0.020,
        "A-": 0.025,
        "BBB+": 0.030,
        "BBB": 0.040,
        "BBB-": 0.050,
        "BB+": 0.060,
        "BB": 0.080,
        "BB-": 0.100,
        "B+": 0.120,
        "B": 0.150,
        "B-": 0.180,
        "CCC+": 0.220,
        "CCC": 0.280,
        "CCC-": 0.350,
        "D": 0.500,
        "NR": 0.050,
    }

    def _get_credit_spread(self, rating: str) -> float:
        """Get credit spread based on rating (in decimal form) - optimized with dict lookup"""
        return self._CREDIT_SPREADS.get(rating.upper(), 0.040)  # Default to BBB spread

    def calculate_duration(self, bond: Bond, ytm: Optional[float] = None) -> float:
        """Calculate Macaulay Duration with caching"""
        if ytm is None:
            ytm = self.calculate_yield_to_maturity(bond)

        # Check cache if enabled
        if self.enable_caching and self._calculation_cache is not None:
            cache_key = (bond.bond_id, ytm, "duration")
            if cache_key in self._calculation_cache:
                return self._calculation_cache[cache_key]

        time_to_maturity = bond.time_to_maturity
        if time_to_maturity <= 0:
            return 0

        if bond.bond_type == BondType.ZERO_COUPON:
            result = time_to_maturity
            # Cache result
            if self.enable_caching and self._calculation_cache is not None:
                self._update_cache(cache_key, result)
            return result

        periods = int(time_to_maturity * bond.frequency)
        coupon_payment = (bond.coupon_rate / 100) * bond.face_value / bond.frequency

        if periods > 0:
            # Vectorized calculation
            periods_array = np.arange(1, periods + 1)
            discount_factors = (1 + ytm / bond.frequency) ** periods_array
            pv_coupons = coupon_payment / discount_factors
            weighted_sum = np.sum((periods_array / bond.frequency) * pv_coupons)
            pv_sum = np.sum(pv_coupons)
        else:
            weighted_sum = 0
            pv_sum = 0

        # Add face value
        pv_face = bond.face_value / ((1 + ytm / bond.frequency) ** periods)
        weighted_sum += time_to_maturity * pv_face
        pv_sum += pv_face

        duration = weighted_sum / pv_sum if pv_sum > 0 else 0

        # Cache result
        if self.enable_caching and self._calculation_cache is not None:
            self._update_cache((bond.bond_id, ytm, "duration"), duration)

        return duration

    def calculate_convexity(self, bond: Bond, ytm: Optional[float] = None) -> float:
        """Calculate convexity with caching"""
        if ytm is None:
            ytm = self.calculate_yield_to_maturity(bond)

        # Check cache if enabled
        if self.enable_caching and self._calculation_cache is not None:
            cache_key = (bond.bond_id, ytm, "convexity")
            if cache_key in self._calculation_cache:
                return self._calculation_cache[cache_key]

        time_to_maturity = bond.time_to_maturity
        if time_to_maturity <= 0:
            return 0

        periods = int(time_to_maturity * bond.frequency)
        coupon_payment = (bond.coupon_rate / 100) * bond.face_value / bond.frequency

        if periods > 0:
            # Vectorized calculation
            periods_array = np.arange(1, periods + 1)
            discount_factors = (1 + ytm / bond.frequency) ** periods_array
            pv_coupons = coupon_payment / discount_factors
            convexity_factor = 1 / ((1 + ytm / bond.frequency) ** 2)
            convexity_sum = np.sum(periods_array * (periods_array + 1) * pv_coupons * convexity_factor)
            pv_sum = np.sum(pv_coupons)
        else:
            convexity_sum = 0
            pv_sum = 0

        # Add face value
        pv_face = bond.face_value / ((1 + ytm / bond.frequency) ** periods)
        convexity_sum += (periods * (periods + 1)) * pv_face / ((1 + ytm / bond.frequency) ** 2)
        pv_sum += pv_face

        convexity = convexity_sum / (pv_sum * (bond.frequency**2)) if pv_sum > 0 else 0

        # Cache result
        if self.enable_caching and self._calculation_cache is not None:
            self._update_cache((bond.bond_id, ytm, "convexity"), convexity)

        return convexity

    def calculate_price_mismatch(self, bond: Bond) -> Dict[str, Any]:
        """
        Calculate mismatch between market price and fair value

        Args:
            bond: Bond object to analyze

        Returns:
            Dictionary with mismatch percentage and details containing:
            - fair_value: Calculated fair value
            - market_price: Current market price
            - mismatch_absolute: Absolute difference
            - mismatch_percentage: Percentage difference
            - overvalued: Boolean indicating if overvalued
            - undervalued: Boolean indicating if undervalued

        Raises:
            ValueError: If bond has invalid price or maturity
            TypeError: If bond is not a Bond instance
        """
        if not isinstance(bond, Bond):
            raise TypeError(f"Expected Bond instance, got {type(bond)}")

        try:
            fair_value = self.calculate_fair_value(bond)
            market_price = bond.current_price

            if fair_value <= 0:
                raise ValueError(f"Invalid fair value calculated: {fair_value}")
            if market_price <= 0:
                raise ValueError(f"Invalid market price: {market_price}")

            mismatch = market_price - fair_value
            mismatch_pct = (mismatch / fair_value) * 100

            return {
                "fair_value": fair_value,
                "market_price": market_price,
                "mismatch_absolute": mismatch,
                "mismatch_percentage": mismatch_pct,
                "overvalued": mismatch_pct > 0,
                "undervalued": mismatch_pct < 0,
            }
        except (ValueError, TypeError) as e:
            from bondtrader.utils.utils import logger

            logger.error(f"Error calculating price mismatch for bond {bond.bond_id}: {e}")
            raise

    def _update_cache(self, key: tuple, value: float) -> None:
        """Update calculation cache with size limit management"""
        if not self.enable_caching or self._calculation_cache is None:
            return

        # Clear cache if it's too large (simple FIFO eviction)
        if len(self._calculation_cache) >= self._cache_size_limit:
            # Remove oldest 20% of entries (simple approach)
            keys_to_remove = list(self._calculation_cache.keys())[: self._cache_size_limit // 5]
            for k in keys_to_remove:
                del self._calculation_cache[k]

        self._calculation_cache[key] = value

    def clear_cache(self) -> None:
        """Clear calculation cache"""
        if self._calculation_cache is not None:
            self._calculation_cache.clear()

    def batch_calculate_ytm(self, bonds: List[Bond], market_prices: Optional[List[float]] = None) -> np.ndarray:
        """
        Batch calculate YTM for multiple bonds (vectorized where possible)

        Args:
            bonds: List of bonds
            market_prices: Optional list of market prices (uses bond.current_price if None)

        Returns:
            Array of YTM values
        """
        if market_prices is None:
            market_prices = [b.current_price for b in bonds]

        ytms = np.array([self.calculate_yield_to_maturity(bond, price) for bond, price in zip(bonds, market_prices)])
        return ytms

    def batch_calculate_duration(self, bonds: List[Bond], ytms: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Batch calculate duration for multiple bonds

        Args:
            bonds: List of bonds
            ytms: Optional array of YTM values (calculated if None)

        Returns:
            Array of duration values
        """
        if ytms is None:
            ytms = self.batch_calculate_ytm(bonds)

        durations = np.array([self.calculate_duration(bond, ytm) for bond, ytm in zip(bonds, ytms)])
        return durations
