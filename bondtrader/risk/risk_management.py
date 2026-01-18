"""
Risk Management Module
Calculates VaR, credit risk, interest rate sensitivity, and other risk metrics
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

# Optional Numba JIT for performance
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

from bondtrader.core.bond_models import Bond
from bondtrader.core.bond_valuation import BondValuator


class RiskManager:
    """Risk management and analysis for bond portfolios"""

    def __init__(self, valuator: BondValuator = None):
        """
        Initialize risk manager

        Args:
            valuator: Bond valuator instance
        """
        self.valuator = valuator if valuator else BondValuator()

    def calculate_var(
        self,
        bonds: List[Bond],
        weights: List[float] = None,
        confidence_level: float = 0.95,
        time_horizon: int = 1,
        method: str = "historical",
    ) -> Dict:
        """
        Calculate Value at Risk (VaR) for a bond portfolio

        Args:
            bonds: List of bonds in portfolio
            weights: Portfolio weights (if None, equal weights)
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            time_horizon: Time horizon in days
            method: 'historical', 'parametric', or 'monte_carlo'

        Returns:
            Dictionary with VaR metrics
        """
        if weights is None:
            weights = [1.0 / len(bonds)] * len(bonds)

        # Get portfolio value
        portfolio_value = sum(b.current_price * w for b, w in zip(bonds, weights))

        if method == "historical":
            return self._var_historical(bonds, weights, confidence_level, time_horizon, portfolio_value)
        elif method == "parametric":
            return self._var_parametric(bonds, weights, confidence_level, time_horizon, portfolio_value)
        elif method == "monte_carlo":
            return self._var_monte_carlo(bonds, weights, confidence_level, time_horizon, portfolio_value)
        else:
            raise ValueError(f"Unknown VaR method: {method}")

    def _var_historical(
        self,
        bonds: List[Bond],
        weights: List[float],
        confidence_level: float,
        time_horizon: int,
        portfolio_value: float,
        num_simulations: int = 1000,
    ) -> Dict:
        """Calculate VaR using historical simulation"""
        # Simulate returns based on historical volatility
        portfolio_returns = []

        for _ in range(num_simulations):
            portfolio_return = 0
            for bond, weight in zip(bonds, weights):
                # Simulate yield change (simplified)
                yield_change = np.random.normal(0, 0.001)  # 0.1% volatility
                new_ytm = self.valuator.calculate_yield_to_maturity(bond) + yield_change

                # Approximate price change using duration
                duration = self.valuator.calculate_duration(bond, new_ytm)
                price_change_pct = -duration * yield_change

                portfolio_return += weight * price_change_pct

            portfolio_returns.append(portfolio_return)

        # Calculate VaR
        var_percentile = (1 - confidence_level) * 100
        var_return = np.percentile(portfolio_returns, var_percentile)
        var_value = abs(var_return * portfolio_value * np.sqrt(time_horizon))

        return {
            "var_value": var_value,
            "var_percentage": abs(var_return) * 100 * np.sqrt(time_horizon),
            "confidence_level": confidence_level,
            "time_horizon": time_horizon,
            "method": "historical",
        }

    def _var_parametric(
        self, bonds: List[Bond], weights: List[float], confidence_level: float, time_horizon: int, portfolio_value: float
    ) -> Dict:
        """Calculate VaR using parametric method"""
        # Calculate portfolio duration
        portfolio_duration = 0
        for bond, weight in zip(bonds, weights):
            ytm = self.valuator.calculate_yield_to_maturity(bond)
            duration = self.valuator.calculate_duration(bond, ytm)
            portfolio_duration += weight * duration

        # Assume yield volatility
        yield_volatility = 0.001  # 0.1% daily volatility

        # Calculate VaR using normal distribution
        z_score = stats.norm.ppf(1 - confidence_level)
        var_return = -z_score * portfolio_duration * yield_volatility * np.sqrt(time_horizon)
        var_value = abs(var_return * portfolio_value)

        return {
            "var_value": var_value,
            "var_percentage": abs(var_return) * 100,
            "confidence_level": confidence_level,
            "time_horizon": time_horizon,
            "method": "parametric",
            "portfolio_duration": portfolio_duration,
        }

    def _var_monte_carlo(
        self,
        bonds: List[Bond],
        weights: List[float],
        confidence_level: float,
        time_horizon: int,
        portfolio_value: float,
        num_simulations: int = 10000,
    ) -> Dict:
        """Calculate VaR using Monte Carlo simulation"""
        portfolio_values = []

        for _ in range(num_simulations):
            portfolio_val = 0
            for bond, weight in zip(bonds, weights):
                # Simulate yield changes
                yield_change = np.random.normal(0, 0.001 * np.sqrt(time_horizon))
                new_ytm = self.valuator.calculate_yield_to_maturity(bond) + yield_change

                # Recalculate bond price
                duration = self.valuator.calculate_duration(bond, new_ytm)
                convexity = self.valuator.calculate_convexity(bond, new_ytm)

                # Price change using duration and convexity
                price_change_pct = -duration * yield_change + 0.5 * convexity * (yield_change**2)
                new_price = bond.current_price * (1 + price_change_pct)

                portfolio_val += new_price * weight * bond.face_value

            portfolio_values.append(portfolio_val)

        # Calculate VaR
        var_percentile = (1 - confidence_level) * 100
        var_portfolio_value = np.percentile(portfolio_values, var_percentile)
        var_value = portfolio_value - var_portfolio_value

        return {
            "var_value": var_value,
            "var_percentage": (var_value / portfolio_value) * 100,
            "confidence_level": confidence_level,
            "time_horizon": time_horizon,
            "method": "monte_carlo",
        }

    def calculate_credit_risk(self, bond: Bond) -> Dict:
        """
        Calculate credit risk metrics for a bond

        Returns:
            Dictionary with credit risk metrics
        """
        # Default probability based on rating
        default_prob = self._get_default_probability(bond.credit_rating)

        # Recovery rate
        recovery_rate = self._get_recovery_rate(bond.credit_rating)

        # Expected loss
        expected_loss = default_prob * (1 - recovery_rate) * bond.current_price

        # Credit spread
        credit_spread = self.valuator._get_credit_spread(bond.credit_rating)

        return {
            "default_probability": default_prob,
            "recovery_rate": recovery_rate,
            "expected_loss": expected_loss,
            "credit_spread": credit_spread,
            "loss_given_default": (1 - recovery_rate) * bond.current_price,
        }

    def _get_default_probability(self, rating: str) -> float:
        """Get default probability based on credit rating (annual)"""
        from bondtrader.utils.constants import get_default_probability

        return get_default_probability(rating)

    def _get_recovery_rate(self, rating: str) -> float:
        """Get recovery rate based on credit rating"""
        from bondtrader.utils.constants import get_recovery_rate_standard

        return get_recovery_rate_standard(rating)

    def calculate_interest_rate_sensitivity(self, bond: Bond, rate_change: float = 0.01) -> Dict:  # 1% change
        """
        Calculate price sensitivity to interest rate changes

        Args:
            bond: Bond object
            rate_change: Interest rate change (as decimal, e.g., 0.01 for 1%)

        Returns:
            Dictionary with sensitivity metrics
        """
        ytm = self.valuator.calculate_yield_to_maturity(bond)
        duration = self.valuator.calculate_duration(bond, ytm)
        convexity = self.valuator.calculate_convexity(bond, ytm)

        # Price change using duration approximation
        price_change_duration = -duration * rate_change

        # Price change using duration + convexity (more accurate)
        price_change_full = -duration * rate_change + 0.5 * convexity * (rate_change**2)

        new_price_duration = bond.current_price * (1 + price_change_duration)
        new_price_full = bond.current_price * (1 + price_change_full)

        return {
            "duration": duration,
            "convexity": convexity,
            "modified_duration": duration / (1 + ytm),
            "price_change_duration_pct": price_change_duration * 100,
            "price_change_full_pct": price_change_full * 100,
            "new_price_duration": new_price_duration,
            "new_price_full": new_price_full,
            "rate_change": rate_change,
        }

    def stress_test(self, bonds: List[Bond], scenario: str = "rate_shock", shock_size: float = 0.02) -> Dict:
        """
        Stress test portfolio under various scenarios

        Args:
            bonds: List of bonds
            scenario: 'rate_shock', 'credit_shock', or 'liquidity_crisis'
            shock_size: Size of shock (as decimal)

        Returns:
            Stress test results
        """
        results = {"scenario": scenario, "shock_size": shock_size, "bonds": []}

        total_value_before = sum(b.current_price for b in bonds)
        total_value_after = 0

        for bond in bonds:
            if scenario == "rate_shock":
                sensitivity = self.calculate_interest_rate_sensitivity(bond, shock_size)
                new_price = sensitivity["new_price_full"]
            elif scenario == "credit_shock":
                # Downgrade by one notch
                new_spread = self.valuator._get_credit_spread(bond.credit_rating) + shock_size
                new_ytm = self.valuator.risk_free_rate + new_spread
                new_price = self.valuator.calculate_fair_value(bond, required_yield=new_ytm)
            else:  # liquidity_crisis
                # Assume 5% liquidity discount
                new_price = bond.current_price * (1 - shock_size)

            total_value_after += new_price

            results["bonds"].append(
                {
                    "bond_id": bond.bond_id,
                    "price_before": bond.current_price,
                    "price_after": new_price,
                    "change_pct": ((new_price - bond.current_price) / bond.current_price) * 100,
                }
            )

        results["portfolio_value_before"] = total_value_before
        results["portfolio_value_after"] = total_value_after
        results["portfolio_change_pct"] = ((total_value_after - total_value_before) / total_value_before) * 100

        return results
