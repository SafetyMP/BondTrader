"""
Key Rate Duration (KRD) Module
Industry-standard risk metric used by all major bond trading firms
Measures sensitivity to specific points on the yield curve
"""

from typing import Dict, List, Optional

import numpy as np

from bondtrader.core.bond_models import Bond
from bondtrader.core.bond_valuation import BondValuator
from bondtrader.utils.utils import logger


class KeyRateDuration:
    """
    Calculate Key Rate Duration (KRD) - sensitivity to specific yield curve points
    Used by Goldman Sachs, JPMorgan, BlackRock, PIMCO for yield curve risk management
    """

    # Standard key rate points (years)
    STANDARD_KEY_RATES = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]

    def __init__(self, valuator: BondValuator = None, shock_size: float = 0.0001):
        """
        Initialize KRD calculator

        Args:
            valuator: Bond valuator instance
            shock_size: Size of yield shock (as decimal, default 1bp = 0.0001)
        """
        self.valuator = valuator if valuator else BondValuator()
        self.shock_size = shock_size

    def calculate_krd(self, bond: Bond, key_rates: Optional[List[float]] = None) -> Dict:
        """
        Calculate Key Rate Duration for a bond

        KRD measures price sensitivity to changes at specific points on the yield curve
        while keeping other points constant (localized shock)

        Args:
            bond: Bond object
            key_rates: List of key rate points in years (uses standard if None)

        Returns:
            Dictionary with KRD at each key rate point
        """
        if key_rates is None:
            key_rates = self.STANDARD_KEY_RATES

        time_to_maturity = bond.time_to_maturity

        # Base price
        base_price = bond.current_price
        base_ytm = self.valuator.calculate_yield_to_maturity(bond)

        krd_results = {}
        krd_values = []

        for key_rate in key_rates:
            # Calculate KRD at this key rate point
            krd = self._calculate_krd_at_point(bond, key_rate, base_price, base_ytm, time_to_maturity)
            krd_values.append(krd)
            krd_results[f"{key_rate}y"] = krd

        # Sum of KRDs should approximately equal Macaulay duration
        total_krd = sum(krd_values)
        macaulay_duration = self.valuator.calculate_duration(bond, base_ytm)

        return {
            "krd_by_rate": krd_results,
            "krd_values": krd_values,
            "key_rates": key_rates,
            "total_krd": total_krd,
            "macaulay_duration": macaulay_duration,
            "krd_sum_vs_duration_diff": abs(total_krd - macaulay_duration),
            "base_price": base_price,
            "base_ytm": base_ytm * 100,
        }

    def _calculate_krd_at_point(
        self, bond: Bond, key_rate: float, base_price: float, base_ytm: float, time_to_maturity: float
    ) -> float:
        """
        Calculate KRD at a specific key rate point

        Uses localized shock: only the specified maturity point is shocked,
        with linear interpolation for adjacent points
        """
        # Build yield curve with shock at key rate point
        # For simplicity, use a simplified curve interpolation
        # In production, would use actual yield curve construction

        shock = self.shock_size

        # Calculate price after localized shock
        # Simplified approach: shock the yield at this point proportionally
        # based on bond's maturity relative to key rate

        if time_to_maturity <= key_rate:
            # Bond matures before or at key rate
            # Shock affects bond directly
            new_ytm = base_ytm + shock
            shocked_price = self._calculate_price_at_ytm(bond, new_ytm)
        else:
            # Bond matures after key rate
            # Shock affects proportionally based on key rate weight
            weight = self._get_key_rate_weight(time_to_maturity, key_rate)
            new_ytm = base_ytm + shock * weight
            shocked_price = self._calculate_price_at_ytm(bond, new_ytm)

        # KRD = -dP/dy / P (sensitivity per unit yield change)
        price_change = shocked_price - base_price
        krd = -price_change / (base_price * shock)

        return krd

    def _get_key_rate_weight(self, maturity: float, key_rate: float) -> float:
        """Calculate weight of key rate shock on bond price"""
        # Simplified: linear interpolation
        # More sophisticated methods use cubic splines or B-splines

        # Find adjacent key rates
        sorted_rates = sorted(self.STANDARD_KEY_RATES + [maturity])
        idx = sorted_rates.index(key_rate)

        if idx == 0:
            # First key rate or before it
            next_rate = sorted_rates[idx + 1] if idx + 1 < len(sorted_rates) else key_rate * 2
            weight = max(0, 1 - abs(maturity - key_rate) / abs(next_rate - key_rate))
        elif idx == len(sorted_rates) - 1:
            # Last key rate or after it
            prev_rate = sorted_rates[idx - 1]
            weight = max(0, 1 - abs(maturity - key_rate) / abs(key_rate - prev_rate))
        else:
            # Between key rates
            prev_rate = sorted_rates[idx - 1]
            next_rate = sorted_rates[idx + 1]

            if maturity < key_rate:
                weight = max(0, 1 - abs(maturity - key_rate) / abs(key_rate - prev_rate))
            else:
                weight = max(0, 1 - abs(maturity - key_rate) / abs(next_rate - key_rate))

        return min(1.0, max(0.0, weight))

    def _calculate_price_at_ytm(self, bond: Bond, ytm: float) -> float:
        """Calculate bond price at given YTM"""
        # Use reverse YTM calculation: price from YTM
        time_to_maturity = bond.time_to_maturity
        if time_to_maturity <= 0:
            return bond.face_value

        if bond.bond_type.value == "Zero Coupon":
            return bond.face_value / ((1 + ytm) ** time_to_maturity)

        periods = int(time_to_maturity * bond.frequency)
        coupon_payment = (bond.coupon_rate / 100) * bond.face_value / bond.frequency

        # PV of coupons
        periods_array = np.arange(1, periods + 1)
        discount_factors = (1 + ytm / bond.frequency) ** periods_array
        pv_coupons = np.sum(coupon_payment / discount_factors)

        # PV of face value
        pv_face = bond.face_value / ((1 + ytm / bond.frequency) ** periods)

        return pv_coupons + pv_face

    def calculate_portfolio_krd(
        self, bonds: List[Bond], weights: Optional[List[float]] = None, key_rates: Optional[List[float]] = None
    ) -> Dict:
        """
        Calculate portfolio-level Key Rate Duration

        Args:
            bonds: List of bonds in portfolio
            weights: Portfolio weights (if None, equal weights)
            key_rates: Key rate points (uses standard if None)

        Returns:
            Portfolio KRD analysis
        """
        if weights is None:
            weights = [1.0 / len(bonds)] * len(bonds)

        if len(weights) != len(bonds):
            raise ValueError("Weights must match bonds length")

        if key_rates is None:
            key_rates = self.STANDARD_KEY_RATES

        # Calculate KRD for each bond
        bond_krds = []
        for bond in bonds:
            krd_result = self.calculate_krd(bond, key_rates)
            bond_krds.append(krd_result["krd_values"])

        # Weighted average of KRDs
        portfolio_krd = np.zeros(len(key_rates))
        for i, bond_krd in enumerate(bond_krds):
            portfolio_krd += np.array(bond_krd) * weights[i]

        # Build results dictionary
        portfolio_krd_dict = {}
        for i, key_rate in enumerate(key_rates):
            portfolio_krd_dict[f"{key_rate}y"] = portfolio_krd[i]

        return {
            "portfolio_krd": portfolio_krd_dict,
            "portfolio_krd_values": portfolio_krd.tolist(),
            "key_rates": key_rates,
            "total_portfolio_krd": float(np.sum(portfolio_krd)),
            "num_bonds": len(bonds),
            "weights": weights,
        }

    def yield_curve_shock_analysis(
        self, bonds: List[Bond], weights: Optional[List[float]] = None, shock_scenarios: Optional[List[str]] = None
    ) -> Dict:
        """
        Analyze portfolio impact of yield curve shock scenarios

        Common scenarios:
        - Parallel shift: all rates move by same amount
        - Steepening: long rates rise more than short rates
        - Flattening: short rates rise more than long rates
        - Twist: short rates rise, long rates fall (or vice versa)

        Args:
            bonds: List of bonds
            weights: Portfolio weights
            shock_scenarios: List of scenario names

        Returns:
            Shock scenario analysis
        """
        if shock_scenarios is None:
            shock_scenarios = ["parallel_shift", "steepening", "flattening", "twist"]

        if weights is None:
            weights = [1.0 / len(bonds)] * len(bonds)

        portfolio_value = sum(b.current_price * w for b, w in zip(bonds, weights))

        scenarios = {}

        for scenario in shock_scenarios:
            scenario_shocks = self._get_scenario_shocks(scenario)

            # Calculate portfolio impact using KRD
            portfolio_krd = self.calculate_portfolio_krd(bonds, weights)
            krd_values = portfolio_krd["portfolio_krd_values"]
            key_rates = portfolio_krd["key_rates"]

            # Apply shocks to KRD
            portfolio_change = 0
            for i, (rate, shock) in enumerate(zip(key_rates, scenario_shocks)):
                if i < len(krd_values):
                    portfolio_change += krd_values[i] * shock * portfolio_value

            portfolio_change_pct = (portfolio_change / portfolio_value) * 100 if portfolio_value > 0 else 0

            scenarios[scenario] = {
                "shocks": scenario_shocks,
                "portfolio_change": portfolio_change,
                "portfolio_change_pct": portfolio_change_pct,
                "new_portfolio_value": portfolio_value + portfolio_change,
            }

        return {"base_portfolio_value": portfolio_value, "scenarios": scenarios, "key_rates": self.STANDARD_KEY_RATES}

    def _get_scenario_shocks(self, scenario: str, shock_size: float = 0.01) -> List[float]:
        """Get yield shocks for different scenarios (in decimal)"""
        num_rates = len(self.STANDARD_KEY_RATES)

        if scenario == "parallel_shift":
            return [shock_size] * num_rates
        elif scenario == "steepening":
            # Long rates rise more than short rates
            return [shock_size * (i + 1) / num_rates for i in range(num_rates)]
        elif scenario == "flattening":
            # Short rates rise more than long rates
            return [shock_size * (num_rates - i) / num_rates for i in range(num_rates)]
        elif scenario == "twist":
            # Short rates rise, long rates fall
            midpoint = num_rates // 2
            shocks = []
            for i in range(num_rates):
                if i < midpoint:
                    shocks.append(shock_size * (i + 1) / midpoint)
                else:
                    shocks.append(-shock_size * (num_rates - i) / (num_rates - midpoint))
            return shocks
        else:
            return [0.0] * num_rates
