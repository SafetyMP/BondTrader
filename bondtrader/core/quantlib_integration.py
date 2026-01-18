"""
QuantLib Integration Module
Optional QuantLib-Python integration for industry-standard fixed income calculations
Gracefully degrades if QuantLib not installed
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# Optional QuantLib-Python for industry-standard calculations
try:
    import QuantLib as ql

    HAS_QUANTLIB = True
except ImportError:
    HAS_QUANTLIB = False
    ql = None

from bondtrader.core.bond_models import Bond, BondType
from bondtrader.utils.utils import logger


class QuantLibIntegration:
    """
    QuantLib integration wrapper for bond calculations

    Provides industry-standard fixed income calculations when QuantLib is available.
    Falls back to standard implementations if QuantLib not installed.
    """

    def __init__(self):
        """Initialize QuantLib integration"""
        if not HAS_QUANTLIB:
            logger.warning(
                "QuantLib not installed. Install with: pip install QuantLib-Python (requires system dependencies). "
                "Falling back to standard implementations."
            )

    @staticmethod
    def is_available() -> bool:
        """Check if QuantLib is available"""
        return HAS_QUANTLIB

    def _bond_type_to_ql(self, bond_type: BondType) -> str:
        """Convert BondType to QuantLib calendar type"""
        if bond_type == BondType.TREASURY:
            return "UnitedStates"
        elif bond_type == BondType.CORPORATE:
            return "UnitedStates"
        elif bond_type == BondType.MUNICIPAL:
            return "UnitedStates"
        else:
            return "UnitedStates"  # Default

    def _datetime_to_ql_date(self, dt: datetime) -> "ql.Date":
        """Convert datetime to QuantLib Date"""
        if not HAS_QUANTLIB:
            raise ImportError("QuantLib not installed")
        return ql.Date(dt.day, dt.month, dt.year)

    def calculate_bond_price_quantlib(self, bond: Bond, yield_rate: float) -> Optional[float]:
        """
        Calculate bond price using QuantLib

        Args:
            bond: Bond object
            yield_rate: Yield rate (as decimal, e.g., 0.05 for 5%)

        Returns:
            Bond price calculated with QuantLib, or None if QuantLib not available
        """
        if not HAS_QUANTLIB:
            return None

        try:
            # Setup dates
            settlement_date = ql.Date.todaysDate()
            issue_date = self._datetime_to_ql_date(bond.issue_date)
            maturity_date = self._datetime_to_ql_date(bond.maturity_date)

            # Calendar and day count
            calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
            day_count = ql.ActualActual(ql.ActualActual.Bond)

            # Schedule
            schedule = ql.Schedule(
                issue_date,
                maturity_date,
                ql.Period(12 // bond.frequency, ql.Months),
                calendar,
                ql.ModifiedFollowing,
                ql.ModifiedFollowing,
                ql.DateGeneration.Backward,
                False,
            )

            # Bond attributes
            face_value = bond.face_value
            coupon_rate = bond.coupon_rate / 100  # Convert to decimal

            # Create bond
            if bond.bond_type == BondType.ZERO_COUPON:
                # Zero coupon bond
                bond_ql = ql.ZeroCouponBond(2, calendar, face_value, maturity_date, ql.ModifiedFollowing)
            else:
                # Fixed rate bond
                bond_ql = ql.FixedRateBond(2, calendar, face_value, schedule, [coupon_rate], day_count)

            # Set pricing engine
            discount_curve = ql.FlatForward(settlement_date, yield_rate, day_count, ql.Compounded, ql.Semiannual)
            bond_engine = ql.DiscountingBondEngine(discount_curve)
            bond_ql.setPricingEngine(bond_engine)

            # Calculate clean price
            clean_price = bond_ql.cleanPrice()

            return clean_price

        except Exception as e:
            logger.warning(f"QuantLib calculation failed: {e}. Falling back to standard implementation.")
            return None

    def calculate_ytm_quantlib(self, bond: Bond, market_price: float) -> Optional[float]:
        """
        Calculate yield to maturity using QuantLib

        Args:
            bond: Bond object
            market_price: Current market price

        Returns:
            YTM (as decimal), or None if QuantLib not available
        """
        if not HAS_QUANTLIB:
            return None

        try:
            # Setup dates
            settlement_date = ql.Date.todaysDate()
            issue_date = self._datetime_to_ql_date(bond.issue_date)
            maturity_date = self._datetime_to_ql_date(bond.maturity_date)

            # Calendar and day count
            calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
            day_count = ql.ActualActual(ql.ActualActual.Bond)

            # Schedule
            schedule = ql.Schedule(
                issue_date,
                maturity_date,
                ql.Period(12 // bond.frequency, ql.Months),
                calendar,
                ql.ModifiedFollowing,
                ql.ModifiedFollowing,
                ql.DateGeneration.Backward,
                False,
            )

            # Bond attributes
            face_value = bond.face_value
            coupon_rate = bond.coupon_rate / 100

            # Create bond
            if bond.bond_type == BondType.ZERO_COUPON:
                bond_ql = ql.ZeroCouponBond(2, calendar, face_value, maturity_date, ql.ModifiedFollowing)
            else:
                bond_ql = ql.FixedRateBond(2, calendar, face_value, schedule, [coupon_rate], day_count)

            # Calculate YTM using bond price
            try:
                ytm = bond_ql.bondYield(market_price, day_count, ql.Compounded, ql.Semiannual)
                return ytm
            except RuntimeError:
                # If QuantLib fails, return None to fall back
                return None

        except Exception as e:
            logger.warning(f"QuantLib YTM calculation failed: {e}. Falling back to standard implementation.")
            return None

    def calculate_accrued_interest(self, bond: Bond) -> Optional[float]:
        """
        Calculate accrued interest using QuantLib day count conventions

        Args:
            bond: Bond object

        Returns:
            Accrued interest, or None if QuantLib not available
        """
        if not HAS_QUANTLIB:
            return None

        try:
            settlement_date = ql.Date.todaysDate()
            issue_date = self._datetime_to_ql_date(bond.issue_date)
            maturity_date = self._datetime_to_ql_date(bond.maturity_date)

            calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
            day_count = ql.ActualActual(ql.ActualActual.Bond)

            schedule = ql.Schedule(
                issue_date,
                maturity_date,
                ql.Period(12 // bond.frequency, ql.Months),
                calendar,
                ql.ModifiedFollowing,
                ql.ModifiedFollowing,
                ql.DateGeneration.Backward,
                False,
            )

            face_value = bond.face_value
            coupon_rate = bond.coupon_rate / 100

            if bond.bond_type == BondType.ZERO_COUPON:
                return 0.0

            bond_ql = ql.FixedRateBond(2, calendar, face_value, schedule, [coupon_rate], day_count)
            accrued = bond_ql.accruedAmount(settlement_date)
            return accrued

        except Exception as e:
            logger.warning(f"QuantLib accrued interest calculation failed: {e}")
            return None

    def build_yield_curve_quantlib(
        self, maturities: List[float], rates: List[float], interpolation: str = "cubic"
    ) -> Optional[Dict]:
        """
        Build yield curve using QuantLib

        Args:
            maturities: List of maturities (in years)
            rates: List of rates (as decimals)
            interpolation: Interpolation method ('linear', 'cubic', 'log')

        Returns:
            Dictionary with curve information, or None if QuantLib not available
        """
        if not HAS_QUANTLIB:
            return None

        try:
            settlement_date = ql.Date.todaysDate()
            calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
            day_count = ql.ActualActual(ql.ActualActual.Bond)

            # Convert maturities to QuantLib dates
            dates = [settlement_date + ql.Period(int(m * 365), ql.Days) for m in maturities]

            # Create yield term structure
            if interpolation.lower() == "linear":
                interpolation_method = ql.Linear()
            elif interpolation.lower() == "cubic":
                interpolation_method = ql.Cubic()
            else:
                interpolation_method = ql.LogLinear()

            # Create curve
            curve = ql.InterpolatedZeroCurve(dates, rates, day_count, calendar, interpolation_method)

            return {
                "curve": curve,
                "settlement_date": settlement_date,
                "dates": dates,
                "rates": rates,
                "interpolation": interpolation,
            }

        except Exception as e:
            logger.warning(f"QuantLib yield curve construction failed: {e}")
            return None

    def get_day_count_fraction(self, start_date: datetime, end_date: datetime, convention: str = "ACT/365") -> Optional[float]:
        """
        Calculate day count fraction using QuantLib conventions

        Args:
            start_date: Start date
            end_date: End date
            convention: Day count convention ('ACT/365', 'ACT/360', '30/360', 'ACT/ACT')

        Returns:
            Day count fraction, or None if QuantLib not available
        """
        if not HAS_QUANTLIB:
            return None

        try:
            ql_start = self._datetime_to_ql_date(start_date)
            ql_end = self._datetime_to_ql_date(end_date)

            # Map conventions
            convention_map = {
                "ACT/365": ql.Actual365Fixed(),
                "ACT/360": ql.Actual360(),
                "30/360": ql.Thirty360(ql.Thirty360.BondBasis),
                "ACT/ACT": ql.ActualActual(ql.ActualActual.Bond),
            }

            day_count = convention_map.get(convention.upper(), ql.Actual365Fixed())
            return day_count.yearFraction(ql_start, ql_end)

        except Exception as e:
            logger.warning(f"QuantLib day count calculation failed: {e}")
            return None


# Global instance
_quantlib_integration: Optional[QuantLibIntegration] = None


def get_quantlib_integration() -> QuantLibIntegration:
    """Get global QuantLib integration instance"""
    global _quantlib_integration
    if _quantlib_integration is None:
        _quantlib_integration = QuantLibIntegration()
    return _quantlib_integration


def is_quantlib_available() -> bool:
    """Check if QuantLib is available"""
    return HAS_QUANTLIB
