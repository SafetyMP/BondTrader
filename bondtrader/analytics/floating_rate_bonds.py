"""
Floating Rate Bond Pricing Module
Industry-standard pricing for floating rate bonds (LIBOR/SOFR-based)
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np

from bondtrader.analytics.multi_curve import MultiCurveFramework
from bondtrader.core.bond_models import Bond, BondType
from bondtrader.core.bond_valuation import BondValuator
from bondtrader.utils.utils import logger


class FloatingRateBondPricer:
    """
    Pricing engine for floating rate bonds
    Handles LIBOR/SOFR-based floating coupons with reset mechanisms
    """

    def __init__(self, valuator: BondValuator = None, multi_curve: MultiCurveFramework = None):
        """
        Initialize floating rate bond pricer

        Args:
            valuator: Optional BondValuator instance. If None, gets from container.
            multi_curve: Optional MultiCurveFramework instance.
        """
        if valuator is None:
            from bondtrader.core.container import get_container

            self.valuator = get_container().get_valuator()
        else:
            self.valuator = valuator
        self.multi_curve = multi_curve if multi_curve else MultiCurveFramework(self.valuator)
        if not self.multi_curve.ois_curve:
            self.multi_curve.initialize_default_curves()

    def calculate_floating_coupon(
        self, bond: Bond, reset_date: datetime, reference_rate: Optional[float] = None, spread: float = 0.0
    ) -> Dict:
        """
        Calculate floating rate coupon payment

        Args:
            bond: Bond object (should be FLOATING_RATE type)
            reset_date: Date of last reset
            reference_rate: Reference rate (LIBOR/SOFR) - if None, uses curve
            spread: Spread over reference rate (in decimal, e.g., 0.002 for 20bp)

        Returns:
            Dictionary with coupon details
        """
        if bond.bond_type != BondType.FLOATING_RATE:
            logger.warning(f"Bond {bond.bond_id} is not floating rate type")

        # Get reference rate from curve if not provided
        if reference_rate is None:
            time_to_reset = (reset_date - datetime.now()).days / 365.25
            if time_to_reset > 0:
                reference_rate = float(self.multi_curve.libor_curve["interpolation"](time_to_reset))
            else:
                # Use short-term rate
                reference_rate = self.valuator.risk_free_rate + 0.002

        # Calculate coupon rate
        coupon_rate = reference_rate + spread

        # Coupon payment
        coupon_payment = coupon_rate * bond.face_value / bond.frequency

        return {
            "reference_rate": reference_rate,
            "reference_rate_pct": reference_rate * 100,
            "spread": spread,
            "spread_bps": spread * 10000,
            "coupon_rate": coupon_rate,
            "coupon_rate_pct": coupon_rate * 100,
            "coupon_payment": coupon_payment,
            "reset_date": reset_date,
            "frequency": bond.frequency,
        }

    def price_floating_rate_bond(
        self,
        bond: Bond,
        next_reset_date: datetime,
        current_reference_rate: Optional[float] = None,
        spread: float = 0.0,
        use_multi_curve: bool = True,
    ) -> Dict:
        """
        Price a floating rate bond

        Floating rate bonds are priced near par because coupons reset to market rates

        Args:
            bond: Floating rate bond
            next_reset_date: Next coupon reset date
            current_reference_rate: Current reference rate
            spread: Spread over reference rate
            use_multi_curve: Use multi-curve framework

        Returns:
            Pricing results
        """
        time_to_maturity = bond.time_to_maturity
        time_to_next_reset = (next_reset_date - datetime.now()).days / 365.25

        if time_to_maturity <= 0:
            return {"value": bond.face_value, "note": "Bond has matured"}

        # Get current reference rate
        if current_reference_rate is None:
            if use_multi_curve and self.multi_curve.libor_curve:
                current_reference_rate = float(self.multi_curve.libor_curve["interpolation"](time_to_next_reset))
            else:
                current_reference_rate = self.valuator.risk_free_rate + 0.002

        # Calculate next coupon
        next_coupon = self.calculate_floating_coupon(bond, next_reset_date, current_reference_rate, spread)

        # Floating rate bonds trade near par because:
        # 1. Next coupon is known (discounted)
        # 2. Future coupons reset to market rates (PV ≈ par)
        # 3. Principal repaid at par

        if use_multi_curve:
            # Discount next coupon using OIS
            discount_factor = self.multi_curve.get_discount_factor(time_to_next_reset, curve_type="OIS")
            pv_next_coupon = next_coupon["coupon_payment"] * discount_factor

            # Remaining value (par + accrued)
            # After next reset, bond will trade near par
            discount_to_maturity = self.multi_curve.get_discount_factor(time_to_maturity, curve_type="OIS")
            pv_principal = bond.face_value * discount_to_maturity

            # Accrued interest (from last reset to now)
            days_since_reset = (datetime.now() - next_reset_date + timedelta(days=365 / bond.frequency)).days
            days_in_period = 365 / bond.frequency
            accrued_interest = (days_since_reset / days_in_period) * next_coupon["coupon_payment"]

            total_value = pv_next_coupon + pv_principal + accrued_interest
        else:
            # Simplified: floating rate bonds trade near par
            # More sophisticated: discount next coupon, rest ≈ par
            discount_factor = np.exp(-self.valuator.risk_free_rate * time_to_next_reset)
            pv_next_coupon = next_coupon["coupon_payment"] * discount_factor
            pv_principal = bond.face_value * np.exp(-self.valuator.risk_free_rate * time_to_maturity)
            total_value = pv_next_coupon + pv_principal

        # Clean price (without accrued interest)
        clean_price = total_value - accrued_interest if "accrued_interest" in locals() else total_value

        return {
            "dirty_price": total_value,
            "clean_price": clean_price,
            "accrued_interest": accrued_interest if "accrued_interest" in locals() else 0,
            "next_coupon_rate": next_coupon["coupon_rate_pct"],
            "reference_rate": current_reference_rate * 100,
            "spread_bps": spread * 10000,
            "time_to_next_reset": time_to_next_reset,
            "time_to_maturity": time_to_maturity,
            "par_value": bond.face_value,
            "price_to_par": clean_price / bond.face_value if bond.face_value > 0 else 1.0,
        }

    def calculate_discount_margin(self, bond: Bond, market_price: float, next_reset_date: datetime) -> Dict:
        """
        Calculate Discount Margin (DM) for floating rate bond

        DM is the spread that makes PV = market price
        Similar to YTM for fixed rate bonds

        Args:
            bond: Floating rate bond
            market_price: Current market price
            next_reset_date: Next reset date

        Returns:
            Discount margin analysis
        """
        from scipy.optimize import brentq

        def price_with_dm(dm):
            result = self.price_floating_rate_bond(bond, next_reset_date, spread=dm, use_multi_curve=True)
            return result["clean_price"] - market_price

        try:
            dm = brentq(price_with_dm, -0.05, 0.10, maxiter=100)
            return {
                "discount_margin": dm,
                "discount_margin_bps": dm * 10000,
                "market_price": market_price,
                "par_value": bond.face_value,
                "spread_to_par": ((market_price - bond.face_value) / bond.face_value) * 100,
            }
        except ValueError:
            return {"discount_margin": 0, "error": "Could not solve for discount margin"}
