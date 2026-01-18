"""
Enhanced Credit Risk Module
Implements Merton structural model, credit migration matrices, and CVaR
Industry-standard credit risk analysis
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from bondtrader.core.bond_models import Bond
from bondtrader.core.bond_valuation import BondValuator
from bondtrader.utils.utils import logger


class CreditRiskEnhanced:
    """
    Enhanced credit risk analysis using structural and reduced-form models
    Includes Merton model, credit migration matrices, and Credit VaR
    """

    def __init__(self, valuator: BondValuator = None):
        """Initialize enhanced credit risk analyzer"""
        self.valuator = valuator if valuator else BondValuator()

        # Credit migration matrix (annual transition probabilities)
        self.migration_matrix = self._get_default_migration_matrix()

    def _get_default_migration_matrix(self) -> Dict:
        """Get default credit migration matrix (annual transition probabilities)"""
        # Transition probabilities (row = from, column = to)
        # Based on historical averages from rating agencies
        matrix = {
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
        return matrix

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
            # Estimate asset value from bond characteristics
            # Simplified: assume asset value = face value * leverage ratio
            leverage_ratio = 2.0  # Default leverage
            asset_value = bond.face_value * leverage_ratio

        if debt_value is None:
            debt_value = bond.face_value

        # Distance to default
        # d2 = [ln(V/D) + (r - 0.5*sigma^2)*T] / (sigma * sqrt(T))
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

        # Default probability (probability that assets fall below debt)
        # P(default) = N(-d2)
        default_probability = stats.norm.cdf(-distance_to_default)

        # Expected loss
        recovery_rate = self._get_recovery_rate(bond.credit_rating)
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

    def _get_recovery_rate(self, rating: str) -> float:
        """Get recovery rate based on credit rating"""
        recovery_map = {
            "AAA": 0.60,
            "AA": 0.58,
            "AA+": 0.58,
            "AA-": 0.56,
            "A+": 0.56,
            "A": 0.54,
            "A-": 0.52,
            "BBB+": 0.50,
            "BBB": 0.48,
            "BBB-": 0.46,
            "BB+": 0.44,
            "BB": 0.42,
            "BB-": 0.40,
            "B+": 0.38,
            "B": 0.36,
            "B-": 0.34,
            "CCC+": 0.32,
            "CCC": 0.30,
            "CCC-": 0.28,
            "D": 0.20,
            "NR": 0.40,
        }
        return recovery_map.get(rating.upper(), 0.40)

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

        # Get migration probabilities
        if current_rating not in self.migration_matrix:
            # Default probabilities for unknown ratings
            migration_probs = {"AAA": 0.01, "AA": 0.05, "A": 0.10, "BBB": 0.40, "BB": 0.30, "B": 0.10, "CCC": 0.03, "D": 0.01}
        else:
            migration_probs = self.migration_matrix[current_rating]

        # Simulate rating migrations
        ratings = list(migration_probs.keys())
        probs = list(migration_probs.values())

        # Normalize probabilities (ensure they sum to 1)
        total_prob = sum(probs)
        if total_prob > 0:
            probs = [p / total_prob for p in probs]
        else:
            # Fallback: equal probabilities
            probs = [1.0 / len(ratings)] * len(ratings)

        # Simulate scenarios
        rating_outcomes = np.random.choice(ratings, size=num_scenarios, p=probs)

        # Calculate value impact for each rating
        value_impacts = []
        current_value = bond.current_price
        current_ytm = self.valuator.calculate_yield_to_maturity(bond)

        for rating in ratings:
            # Calculate new YTM based on rating
            new_spread = self.valuator._get_credit_spread(rating)
            new_ytm = self.valuator.risk_free_rate + new_spread

            # Approximate price change using duration
            duration = self.valuator.calculate_duration(bond, current_ytm)
            spread_change = new_spread - self.valuator._get_credit_spread(current_rating)
            price_change_pct = -duration * spread_change

            new_value = current_value * (1 + price_change_pct)
            value_impacts.append(new_value)

        # Aggregate results
        scenario_values = []
        for outcome in rating_outcomes:
            idx = ratings.index(outcome)
            scenario_values.append(value_impacts[idx])

        mean_value = np.mean(scenario_values)
        std_value = np.std(scenario_values)

        # Value distribution by rating
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

        # Simulate credit scenarios
        portfolio_values = []

        for _ in range(10000):  # Monte Carlo simulation
            portfolio_value = 0

            for bond, weight in zip(bonds, weights):
                # Simulate credit migration
                migration_result = self.credit_migration_analysis(bond, time_horizon=time_horizon, num_scenarios=1)

                # Get value from migration
                new_rating = np.random.choice(
                    list(migration_result["value_distribution"].keys()),
                    p=[v["probability"] for v in migration_result["value_distribution"].values()],
                )
                new_value = migration_result["value_distribution"][new_rating]["expected_value"]

                portfolio_value += new_value * weight * bond.face_value

            portfolio_values.append(portfolio_value)

        # Calculate CVaR
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
