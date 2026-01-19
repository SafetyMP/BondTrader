"""
Tail Risk Metrics Module
Conditional VaR (CVaR), Expected Shortfall, Tail Expectation
Advanced tail risk beyond standard VaR
"""

from typing import Dict, List, Optional

import numpy as np
from scipy import stats

from bondtrader.core.bond_models import Bond
from bondtrader.core.bond_valuation import BondValuator
from bondtrader.risk.risk_management import RiskManager
from bondtrader.utils.utils import logger


class TailRiskAnalyzer:
    """
    Advanced tail risk analysis
    CVaR, Expected Shortfall, and tail expectation metrics
    More sophisticated than standard VaR used by most platforms
    """

    def __init__(self, valuator: BondValuator = None):
        """
        Initialize tail risk analyzer

        Args:
            valuator: Optional BondValuator instance. If None, gets from container.
        """
        if valuator is None:
            from bondtrader.core.container import get_container

            self.valuator = get_container().get_valuator()
        else:
            self.valuator = valuator
        self.risk_manager = RiskManager(self.valuator)

    def calculate_cvar(
        self,
        bonds: List[Bond],
        weights: Optional[List[float]] = None,
        confidence_level: float = 0.95,
        method: str = "historical",
    ) -> Dict:
        """
        Calculate Conditional Value at Risk (CVaR) / Expected Shortfall

        CVaR = Expected loss given loss > VaR
        More informative than VaR for tail risk

        Args:
            bonds: List of bonds
            weights: Portfolio weights
            confidence_level: Confidence level (e.g., 0.95)
            method: 'historical', 'parametric', or 'monte_carlo'

        Returns:
            CVaR analysis
        """
        if weights is None:
            weights = [1.0 / len(bonds)] * len(bonds)

        portfolio_value = sum(b.current_price * w for b, w in zip(bonds, weights))

        # Generate portfolio returns distribution
        portfolio_returns = []

        num_simulations = 10000
        for _ in range(num_simulations):
            portfolio_return = 0
            for bond, weight in zip(bonds, weights):
                # Simulate yield change
                yield_change = np.random.normal(0, 0.001 * np.sqrt(252))  # Daily to annual
                ytm = self.valuator.calculate_yield_to_maturity(bond)
                new_ytm = ytm + yield_change

                # Price change using duration
                duration = self.valuator.calculate_duration(bond, ytm)
                price_change_pct = -duration * yield_change

                portfolio_return += weight * price_change_pct

            portfolio_returns.append(portfolio_return)

        portfolio_returns = np.array(portfolio_returns)
        portfolio_values = portfolio_value * (1 + portfolio_returns)

        # VaR threshold
        var_percentile = (1 - confidence_level) * 100
        var_value = portfolio_value - np.percentile(portfolio_values, var_percentile)

        # CVaR: average of losses beyond VaR
        var_threshold = portfolio_value - var_value
        tail_losses = portfolio_values[portfolio_values <= var_threshold]

        if len(tail_losses) > 0:
            cvar_value = portfolio_value - np.mean(tail_losses)
        else:
            cvar_value = var_value  # Fallback

        cvar_pct = (cvar_value / portfolio_value) * 100 if portfolio_value > 0 else 0

        # Tail ratio: CVaR / VaR (shows tail heaviness)
        tail_ratio = cvar_value / var_value if var_value > 0 else 1.0

        return {
            "var_value": var_value,
            "var_pct": (var_value / portfolio_value) * 100 if portfolio_value > 0 else 0,
            "cvar_value": cvar_value,
            "cvar_pct": cvar_pct,
            "tail_ratio": tail_ratio,
            "confidence_level": confidence_level,
            "method": method,
            "expected_shortfall": cvar_value,  # CVaR = Expected Shortfall
        }

    def calculate_expected_shortfall_multiple_levels(
        self,
        bonds: List[Bond],
        weights: Optional[List[float]] = None,
        confidence_levels: List[float] = [0.90, 0.95, 0.99, 0.999],
    ) -> Dict:
        """
        Calculate Expected Shortfall at multiple confidence levels

        Shows how tail risk scales with confidence level

        Args:
            bonds: List of bonds
            weights: Portfolio weights
            confidence_levels: List of confidence levels

        Returns:
            Multi-level Expected Shortfall analysis
        """
        es_results = {}

        for conf_level in confidence_levels:
            cvar_result = self.calculate_cvar(bonds, weights, confidence_level=conf_level)
            es_results[f"{conf_level:.1%}"] = {
                "es_value": cvar_result["cvar_value"],
                "es_pct": cvar_result["cvar_pct"],
                "var_value": cvar_result["var_value"],
                "tail_ratio": cvar_result["tail_ratio"],
            }

        return es_results

    def calculate_tail_expectation(
        self,
        bonds: List[Bond],
        weights: Optional[List[float]] = None,
        tail_probability: float = 0.05,
    ) -> Dict:
        """
        Calculate tail expectation

        E[L | L > threshold]
        Expected loss in worst tail_probability scenarios

        Args:
            bonds: List of bonds
            weights: Portfolio weights
            tail_probability: Probability of tail event (e.g., 0.05 for worst 5%)

        Returns:
            Tail expectation metrics
        """
        if weights is None:
            weights = [1.0 / len(bonds)] * len(bonds)

        portfolio_value = sum(b.current_price * w for b, w in zip(bonds, weights))

        # Generate loss distribution
        losses = []
        num_simulations = 10000

        for _ in range(num_simulations):
            portfolio_loss = 0
            for bond, weight in zip(bonds, weights):
                yield_change = np.random.normal(0, 0.002)
                ytm = self.valuator.calculate_yield_to_maturity(bond)
                duration = self.valuator.calculate_duration(bond, ytm)
                price_loss = -(-duration * yield_change)  # Negative of return
                portfolio_loss += price_loss * weight * portfolio_value

            losses.append(portfolio_loss)

        losses = np.array(losses)

        # Tail threshold
        tail_threshold = np.percentile(losses, (1 - tail_probability) * 100)

        # Tail expectation
        tail_losses = losses[losses >= tail_threshold]
        tail_expectation = np.mean(tail_losses) if len(tail_losses) > 0 else tail_threshold

        return {
            "tail_probability": tail_probability,
            "tail_threshold": tail_threshold,
            "tail_expectation": tail_expectation,
            "tail_expectation_pct": (
                (tail_expectation / portfolio_value) * 100 if portfolio_value > 0 else 0
            ),
            "expected_tail_loss": tail_expectation,
        }

    def calculate_maximum_drawdown_distribution(
        self, bonds: List[Bond], weights: Optional[List[float]] = None, time_horizon: int = 252
    ) -> Dict:
        """
        Calculate maximum drawdown distribution

        Shows distribution of worst-case losses over time

        Args:
            bonds: List of bonds
            weights: Portfolio weights
            time_horizon: Time horizon in days

        Returns:
            Maximum drawdown analysis
        """
        if weights is None:
            weights = [1.0 / len(bonds)] * len(bonds)

        max_drawdowns = []
        num_paths = 1000

        for _ in range(num_paths):
            portfolio_values = []
            current_value = sum(b.current_price * w for b, w in zip(bonds, weights))

            for day in range(time_horizon):
                daily_return = 0
                for bond, weight in zip(bonds, weights):
                    yield_change = np.random.normal(0, 0.001)
                    ytm = self.valuator.calculate_yield_to_maturity(bond)
                    duration = self.valuator.calculate_duration(bond, ytm)
                    daily_return += weight * (-duration * yield_change)

                current_value *= 1 + daily_return
                portfolio_values.append(current_value)

            # Calculate maximum drawdown for this path
            portfolio_values = np.array(portfolio_values)
            running_max = np.maximum.accumulate(portfolio_values)
            drawdowns = (portfolio_values - running_max) / running_max
            max_dd = np.min(drawdowns)
            max_drawdowns.append(max_dd)

        max_drawdowns = np.array(max_drawdowns)

        return {
            "mean_max_drawdown": np.mean(max_drawdowns),
            "median_max_drawdown": np.median(max_drawdowns),
            "std_max_drawdown": np.std(max_drawdowns),
            "percentile_5": np.percentile(max_drawdowns, 5),
            "percentile_95": np.percentile(max_drawdowns, 95),
            "worst_case_drawdown": np.min(max_drawdowns),
            "distribution": max_drawdowns.tolist(),
        }
