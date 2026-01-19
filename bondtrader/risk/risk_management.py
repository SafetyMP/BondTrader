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
from bondtrader.utils.utils import logger


class RiskManager:
    """Risk management and analysis for bond portfolios"""

    def __init__(self, valuator: BondValuator = None):
        """
        Initialize risk manager

        Args:
            valuator: Bond valuator instance
        """
        self.valuator = valuator if valuator else BondValuator()
        # Credit migration matrix (annual transition probabilities) for enhanced credit risk
        self.migration_matrix = self._get_default_migration_matrix()

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
        """Calculate VaR using historical simulation (optimized with vectorization)"""
        n_bonds = len(bonds)
        weights_array = np.array(weights)

        # Pre-calculate initial durations (cached)
        initial_ytms = np.array([self.valuator.calculate_yield_to_maturity(bond) for bond in bonds])
        initial_durations = np.array([self.valuator.calculate_duration(bond, ytm) for bond, ytm in zip(bonds, initial_ytms)])

        # OPTIMIZED: Vectorized simulation
        # Generate all yield changes at once (num_simulations x n_bonds)
        yield_changes = np.random.normal(0, 0.001, size=(num_simulations, n_bonds))

        # Approximate price changes using duration (vectorized)
        # Price change ≈ -duration * Δy
        price_change_pct = -initial_durations[np.newaxis, :] * yield_changes

        # Calculate portfolio returns for all simulations (vectorized)
        portfolio_returns = np.sum(price_change_pct * weights_array[np.newaxis, :], axis=1)

        # Calculate VaR (already vectorized)
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
        """Calculate VaR using Monte Carlo simulation (optimized with vectorization)"""
        n_bonds = len(bonds)
        weights_array = np.array(weights)

        # Pre-calculate initial YTM, duration, and convexity for all bonds (cached)
        initial_ytms = np.array([self.valuator.calculate_yield_to_maturity(bond) for bond in bonds])
        initial_durations = np.array([self.valuator.calculate_duration(bond, ytm) for bond, ytm in zip(bonds, initial_ytms)])
        initial_convexities = np.array(
            [self.valuator.calculate_convexity(bond, ytm) for bond, ytm in zip(bonds, initial_ytms)]
        )
        initial_prices = np.array([bond.current_price for bond in bonds])
        face_values = np.array([bond.face_value for bond in bonds])

        # Yield volatility for simulation
        yield_vol = 0.001 * np.sqrt(time_horizon)

        # OPTIMIZED: Vectorized Monte Carlo simulation
        # Generate all yield changes at once (num_simulations x n_bonds)
        yield_changes = np.random.normal(0, yield_vol, size=(num_simulations, n_bonds))
        new_ytms = initial_ytms[np.newaxis, :] + yield_changes

        # Approximate price changes using Taylor expansion (vectorized)
        # Price change ≈ -duration * Δy + 0.5 * convexity * (Δy)^2
        # Note: For large yield changes, we approximate duration/convexity as constant
        # This is acceptable for VaR calculations and much faster than recalculating
        price_change_pct = -initial_durations[np.newaxis, :] * yield_changes + 0.5 * initial_convexities[np.newaxis, :] * (
            yield_changes**2
        )
        new_prices = initial_prices[np.newaxis, :] * (1 + price_change_pct)

        # Calculate portfolio values for all simulations (vectorized)
        # OPTIMIZED: Use bond prices directly (not multiplied by face value, which is already in price)
        portfolio_values = np.sum(new_prices * weights_array[np.newaxis, :], axis=1)

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

    # Enhanced credit risk methods (from CreditRiskEnhanced)
    def _get_default_migration_matrix(self) -> Dict:
        """Get default credit migration matrix (annual transition probabilities)"""
        return {
            "AAA": {
                "AAA": 0.9360,
                "AA": 0.0600,
                "A": 0.0030,
                "BBB": 0.0010,
                "BB": 0.0000,
                "B": 0.0000,
                "CCC": 0.0000,
                "D": 0.0000,
            },
            "AA": {
                "AAA": 0.0070,
                "AA": 0.9250,
                "A": 0.0610,
                "BBB": 0.0050,
                "BB": 0.0010,
                "B": 0.0000,
                "CCC": 0.0000,
                "D": 0.0010,
            },
            "A": {
                "AAA": 0.0000,
                "AA": 0.0230,
                "A": 0.9150,
                "BBB": 0.0550,
                "BB": 0.0040,
                "B": 0.0020,
                "CCC": 0.0000,
                "D": 0.0010,
            },
            "BBB": {
                "AAA": 0.0000,
                "AA": 0.0020,
                "A": 0.0310,
                "BBB": 0.9040,
                "BB": 0.0520,
                "B": 0.0070,
                "CCC": 0.0020,
                "D": 0.0020,
            },
            "BB": {
                "AAA": 0.0000,
                "AA": 0.0000,
                "A": 0.0020,
                "BBB": 0.0480,
                "BB": 0.8250,
                "B": 0.1000,
                "CCC": 0.0150,
                "D": 0.0100,
            },
            "B": {
                "AAA": 0.0000,
                "AA": 0.0000,
                "A": 0.0010,
                "BBB": 0.0040,
                "BB": 0.0750,
                "B": 0.7750,
                "CCC": 0.1000,
                "D": 0.0450,
            },
            "CCC": {
                "AAA": 0.0000,
                "AA": 0.0000,
                "A": 0.0000,
                "BBB": 0.0020,
                "BB": 0.0100,
                "B": 0.0830,
                "CCC": 0.6250,
                "D": 0.2800,
            },
        }

    def merton_structural_model(
        self,
        bond: Bond,
        asset_value: Optional[float] = None,
        asset_volatility: float = 0.25,
        debt_value: Optional[float] = None,
    ) -> Dict:
        """
        Calculate default probability using Merton structural model

        Merton model treats equity as a call option on firm assets
        Default occurs when asset value falls below debt threshold

        Args:
            bond: Bond object
            asset_value: Firm asset value (if None, estimated from bond)
            asset_volatility: Asset value volatility (as decimal)
            debt_value: Total debt value (if None, uses bond face value)

        Returns:
            Dictionary with default probability and distance to default
        """
        time_to_maturity = bond.time_to_maturity

        if asset_value is None:
            leverage_ratio = 2.0
            asset_value = bond.face_value * leverage_ratio

        if debt_value is None:
            debt_value = bond.face_value

        rf_rate = self.valuator.risk_free_rate

        if time_to_maturity <= 0 or asset_volatility <= 0:
            return {
                "default_probability": 0.0 if bond.credit_rating in ["AAA", "AA"] else 0.05,
                "distance_to_default": 5.0,
                "error": "Invalid parameters",
            }

        ln_ratio = np.log(asset_value / debt_value)
        drift = (rf_rate - 0.5 * asset_volatility**2) * time_to_maturity
        denominator = asset_volatility * np.sqrt(time_to_maturity)
        distance_to_default = (ln_ratio + drift) / denominator
        default_probability = stats.norm.cdf(-distance_to_default)

        from bondtrader.utils.constants import get_recovery_rate_enhanced

        recovery_rate = get_recovery_rate_enhanced(bond.credit_rating)
        expected_loss = default_probability * (1 - recovery_rate) * bond.current_price

        return {
            "default_probability": default_probability,
            "distance_to_default": distance_to_default,
            "asset_value": asset_value,
            "debt_value": debt_value,
            "asset_volatility": asset_volatility,
            "recovery_rate": recovery_rate,
            "expected_loss": expected_loss,
            "loss_given_default": (1 - recovery_rate) * bond.current_price,
            "model": "Merton",
        }

    def credit_migration_analysis(self, bond: Bond, time_horizon: float = 1.0, num_scenarios: int = 10000) -> Dict:
        """
        Analyze credit migration risk using migration matrix

        Simulates rating transitions and impact on bond value

        Args:
            bond: Bond object
            time_horizon: Analysis horizon in years
            num_scenarios: Number of simulation scenarios

        Returns:
            Migration analysis results
        """
        current_rating = bond.credit_rating.upper()

        if current_rating not in self.migration_matrix:
            migration_probs = {"AAA": 0.01, "AA": 0.05, "A": 0.10, "BBB": 0.40, "BB": 0.30, "B": 0.10, "CCC": 0.03, "D": 0.01}
        else:
            migration_probs = self.migration_matrix[current_rating]

        ratings = list(migration_probs.keys())
        probs = list(migration_probs.values())

        total_prob = sum(probs)
        if total_prob > 0:
            probs = [p / total_prob for p in probs]
        else:
            probs = [1.0 / len(ratings)] * len(ratings)

        rating_outcomes = np.random.choice(ratings, size=num_scenarios, p=probs)

        value_impacts = []
        current_value = bond.current_price
        current_ytm = self.valuator.calculate_yield_to_maturity(bond)

        for rating in ratings:
            new_spread = self.valuator._get_credit_spread(rating)
            new_ytm = self.valuator.risk_free_rate + new_spread
            duration = self.valuator.calculate_duration(bond, current_ytm)
            spread_change = new_spread - self.valuator._get_credit_spread(current_rating)
            price_change_pct = -duration * spread_change
            new_value = current_value * (1 + price_change_pct)
            value_impacts.append(new_value)

        scenario_values = [value_impacts[ratings.index(outcome)] for outcome in rating_outcomes]
        mean_value = np.mean(scenario_values)
        std_value = np.std(scenario_values)

        value_by_rating = {}
        for i, rating in enumerate(ratings):
            count = np.sum(rating_outcomes == rating)
            value_by_rating[rating] = {
                "probability": count / num_scenarios,
                "expected_value": value_impacts[i],
                "value_change": value_impacts[i] - current_value,
                "value_change_pct": ((value_impacts[i] - current_value) / current_value) * 100,
            }

        return {
            "current_rating": current_rating,
            "current_value": current_value,
            "mean_value": mean_value,
            "std_value": std_value,
            "value_distribution": value_by_rating,
            "time_horizon": time_horizon,
            "num_scenarios": num_scenarios,
            "migration_probabilities": migration_probs,
        }

    def calculate_credit_var(
        self,
        bonds: List[Bond],
        weights: Optional[List[float]] = None,
        confidence_level: float = 0.95,
        time_horizon: float = 1.0,
    ) -> Dict:
        """
        Calculate Credit Value at Risk (CVaR)

        Measures potential loss due to credit events (downgrades, defaults)

        Args:
            bonds: List of bonds in portfolio
            weights: Portfolio weights (if None, equal weights)
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            time_horizon: Time horizon in years

        Returns:
            Credit VaR metrics
        """
        if weights is None:
            weights = [1.0 / len(bonds)] * len(bonds)

        if len(weights) != len(bonds):
            raise ValueError("Weights must match bonds length")

        portfolio_values = []
        for _ in range(10000):
            portfolio_value = 0
            for bond, weight in zip(bonds, weights):
                migration_result = self.credit_migration_analysis(bond, time_horizon=time_horizon, num_scenarios=1)
                new_rating = np.random.choice(
                    list(migration_result["value_distribution"].keys()),
                    p=[v["probability"] for v in migration_result["value_distribution"].values()],
                )
                new_value = migration_result["value_distribution"][new_rating]["expected_value"]
                portfolio_value += new_value * weight * bond.face_value
            portfolio_values.append(portfolio_value)

        portfolio_values = np.array(portfolio_values)
        current_portfolio_value = sum(b.current_price * w * b.face_value for b, w in zip(bonds, weights))
        var_percentile = (1 - confidence_level) * 100
        cvar_value = current_portfolio_value - np.percentile(portfolio_values, var_percentile)
        cvar_pct = (cvar_value / current_portfolio_value) * 100 if current_portfolio_value > 0 else 0

        return {
            "credit_var": cvar_value,
            "credit_var_pct": cvar_pct,
            "confidence_level": confidence_level,
            "time_horizon": time_horizon,
            "current_portfolio_value": current_portfolio_value,
            "mean_portfolio_value": np.mean(portfolio_values),
            "std_portfolio_value": np.std(portfolio_values),
            "percentile_5": np.percentile(portfolio_values, 5),
            "percentile_95": np.percentile(portfolio_values, 95),
        }

    def calculate_expected_credit_loss(self, bonds: List[Bond], weights: Optional[List[float]] = None) -> Dict:
        """
        Calculate expected credit loss across portfolio

        Args:
            bonds: List of bonds
            weights: Portfolio weights

        Returns:
            Expected credit loss metrics
        """
        if weights is None:
            weights = [1.0 / len(bonds)] * len(bonds)

        total_expected_loss = 0
        bond_losses = []

        for bond, weight in zip(bonds, weights):
            merton_result = self.merton_structural_model(bond)
            expected_loss = merton_result["expected_loss"] * weight
            total_expected_loss += expected_loss

            bond_losses.append(
                {
                    "bond_id": bond.bond_id,
                    "rating": bond.credit_rating,
                    "default_probability": merton_result["default_probability"],
                    "expected_loss": expected_loss,
                    "expected_loss_pct": (expected_loss / (bond.current_price * weight)) * 100,
                }
            )

        portfolio_value = sum(b.current_price * w for b, w in zip(bonds, weights))
        expected_loss_pct = (total_expected_loss / portfolio_value) * 100 if portfolio_value > 0 else 0

        return {
            "total_expected_loss": total_expected_loss,
            "expected_loss_pct": expected_loss_pct,
            "portfolio_value": portfolio_value,
            "bond_losses": bond_losses,
        }
