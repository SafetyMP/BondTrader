"""
Multi-Curve Framework
Industry-standard post-2008 approach: separate discounting and forwarding curves
Used by all major financial institutions
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d

from bondtrader.core.bond_models import Bond
from bondtrader.core.bond_valuation import BondValuator
from bondtrader.utils.utils import logger


class MultiCurveFramework:
    """
    Multi-curve framework for modern bond pricing
    Separates discounting curve (OIS) from forwarding curve (LIBOR/SOFR)
    Essential for post-2008 valuation accuracy
    """

    def __init__(self, valuator: BondValuator = None):
        """
        Initialize multi-curve framework

        Args:
            valuator: Optional BondValuator instance. If None, gets from container.
        """
        if valuator is None:
            from bondtrader.core.container import get_container

            self.valuator = get_container().get_valuator()
        else:
            self.valuator = valuator

        # Default curves (in production, would fetch from market data)
        self.ois_curve = None  # Overnight Index Swap curve (discounting)
        self.libor_curve = None  # LIBOR/SOFR curve (forwarding)
        self.treasury_curve = None  # Treasury curve (benchmark)

    def build_ois_curve(
        self, maturities: List[float], rates: List[float], interpolation_method: str = "cubic"
    ) -> Dict:
        """
        Build OIS (Overnight Index Swap) discounting curve

        OIS rates are used for discounting in post-2008 framework
        Typically lower than LIBOR due to lower credit risk

        Args:
            maturities: List of maturities in years
            rates: List of OIS rates (as decimal)
            interpolation_method: 'linear', 'cubic', 'log_linear'

        Returns:
            Dictionary with curve data and interpolation function
        """
        maturities = np.array(maturities)
        rates = np.array(rates)

        # Create interpolation function
        if interpolation_method == "linear":
            interp_func = interp1d(
                maturities, rates, kind="linear", fill_value="extrapolate", bounds_error=False
            )
        elif interpolation_method == "cubic":
            interp_func = interp1d(
                maturities, rates, kind="cubic", fill_value="extrapolate", bounds_error=False
            )
        elif interpolation_method == "log_linear":
            # Log-linear interpolation on rates
            log_rates = np.log(1 + rates)
            interp_func_log = interp1d(
                maturities, log_rates, kind="linear", fill_value="extrapolate", bounds_error=False
            )

            def interp_func(t):
                return np.exp(interp_func_log(t)) - 1

        else:
            raise ValueError(f"Unknown interpolation method: {interpolation_method}")

        self.ois_curve = {
            "maturities": maturities,
            "rates": rates,
            "interpolation": interp_func,
            "method": interpolation_method,
            "type": "OIS",
        }

        return self.ois_curve

    def build_libor_curve(
        self, maturities: List[float], rates: List[float], interpolation_method: str = "cubic"
    ) -> Dict:
        """
        Build LIBOR/SOFR forwarding curve

        Used for projecting forward rates
        Typically higher than OIS due to credit/liquidity spreads

        Args:
            maturities: List of maturities in years
            rates: List of LIBOR/SOFR rates (as decimal)
            interpolation_method: Interpolation method

        Returns:
            Dictionary with curve data
        """
        maturities = np.array(maturities)
        rates = np.array(rates)

        if interpolation_method == "linear":
            interp_func = interp1d(
                maturities, rates, kind="linear", fill_value="extrapolate", bounds_error=False
            )
        elif interpolation_method == "cubic":
            interp_func = interp1d(
                maturities, rates, kind="cubic", fill_value="extrapolate", bounds_error=False
            )
        else:
            interp_func = interp1d(
                maturities, rates, kind="linear", fill_value="extrapolate", bounds_error=False
            )

        self.libor_curve = {
            "maturities": maturities,
            "rates": rates,
            "interpolation": interp_func,
            "method": interpolation_method,
            "type": "LIBOR/SOFR",
        }

        return self.libor_curve

    def get_discount_factor(self, maturity: float, curve_type: str = "OIS") -> float:
        """
        Get discount factor from appropriate curve

        Args:
            maturity: Time to maturity in years
            curve_type: 'OIS' or 'LIBOR' (default: OIS for discounting)

        Returns:
            Discount factor
        """
        if curve_type == "OIS":
            if self.ois_curve is None:
                # Fallback to risk-free rate
                rate = self.valuator.risk_free_rate
            else:
                rate = float(self.ois_curve["interpolation"](maturity))
        elif curve_type == "LIBOR":
            if self.libor_curve is None:
                rate = self.valuator.risk_free_rate + 0.002  # Assume 20bp spread
            else:
                rate = float(self.libor_curve["interpolation"](maturity))
        else:
            rate = self.valuator.risk_free_rate

        return np.exp(-rate * maturity)

    def get_forward_rate(self, t1: float, t2: float, curve_type: str = "LIBOR") -> float:
        """
        Calculate forward rate between times t1 and t2

        Args:
            t1: Start time (years)
            t2: End time (years)
            curve_type: Curve to use for forwarding

        Returns:
            Forward rate (as decimal)
        """
        if curve_type == "LIBOR" and self.libor_curve is not None:
            r1 = float(self.libor_curve["interpolation"](t1))
            r2 = float(self.libor_curve["interpolation"](t2))

            # Forward rate: (r2*t2 - r1*t1) / (t2 - t1)
            if abs(t2 - t1) < 1e-6:
                return r2
            forward_rate = (r2 * t2 - r1 * t1) / (t2 - t1)
            return forward_rate
        else:
            # Fallback calculation
            disc_t1 = self.get_discount_factor(t1)
            disc_t2 = self.get_discount_factor(t2)

            if abs(t2 - t1) < 1e-6:
                return self.valuator.risk_free_rate

            forward_rate = np.log(disc_t1 / disc_t2) / (t2 - t1)
            return forward_rate

    def calculate_basis_spread(self, maturity: float) -> float:
        """
        Calculate basis spread between LIBOR and OIS curves

        Basis spread = LIBOR_rate - OIS_rate
        Represents credit and liquidity premium

        Args:
            maturity: Maturity in years

        Returns:
            Basis spread in basis points
        """
        if self.libor_curve is None or self.ois_curve is None:
            return 0.002  # Default 20bp spread

        libor_rate = float(self.libor_curve["interpolation"](maturity))
        ois_rate = float(self.ois_curve["interpolation"](maturity))

        basis_spread = libor_rate - ois_rate
        return basis_spread

    def price_bond_with_multi_curve(self, bond: Bond, use_ois_discounting: bool = True) -> Dict:
        """
        Price bond using multi-curve framework

        Args:
            bond: Bond object
            use_ois_discounting: Use OIS for discounting (industry standard)

        Returns:
            Dictionary with pricing details
        """
        time_to_maturity = bond.time_to_maturity

        if time_to_maturity <= 0:
            return {
                "value": bond.face_value,
                "discount_curve": "OIS" if use_ois_discounting else "LIBOR",
                "note": "Bond has matured",
            }

        # Use OIS for discounting, forward rates for projection
        if use_ois_discounting:
            discount_curve = "OIS"
        else:
            discount_curve = "LIBOR"

        # Calculate present value of cash flows
        periods = int(time_to_maturity * bond.frequency)
        coupon_payment = (bond.coupon_rate / 100) * bond.face_value / bond.frequency

        pv_coupons = 0
        period_dt = time_to_maturity / periods

        for i in range(1, periods + 1):
            t = i * period_dt
            discount_factor = self.get_discount_factor(t, curve_type=discount_curve)
            pv_coupons += coupon_payment * discount_factor

        # Present value of face value
        discount_factor_face = self.get_discount_factor(time_to_maturity, curve_type=discount_curve)
        pv_face = bond.face_value * discount_factor_face

        total_value = pv_coupons + pv_face

        # Compare to single-curve pricing
        single_curve_value = self.valuator.calculate_fair_value(bond)
        difference = total_value - single_curve_value
        difference_pct = (difference / single_curve_value) * 100 if single_curve_value > 0 else 0

        return {
            "multi_curve_value": total_value,
            "single_curve_value": single_curve_value,
            "difference": difference,
            "difference_pct": difference_pct,
            "discount_curve": discount_curve,
            "pv_coupons": pv_coupons,
            "pv_face": pv_face,
            "basis_spread": self.calculate_basis_spread(time_to_maturity) * 10000,  # In bps
        }

    def initialize_default_curves(self):
        """Initialize default curves for demonstration"""
        # Default OIS curve (lower rates)
        ois_maturities = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]
        ois_rates = [0.025, 0.027, 0.029, 0.030, 0.031, 0.032, 0.033, 0.034, 0.036, 0.038]
        self.build_ois_curve(ois_maturities, ois_rates)

        # Default LIBOR curve (higher rates, ~20bp spread)
        libor_maturities = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]
        libor_rates = [0.027, 0.029, 0.031, 0.032, 0.033, 0.034, 0.035, 0.036, 0.038, 0.040]
        self.build_libor_curve(libor_maturities, libor_rates)

        logger.info("Default multi-curve framework initialized")
