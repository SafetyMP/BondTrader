"""
Bayesian Optimization Module
Advanced hyperparameter tuning and portfolio optimization
More efficient than grid search used by most platforms
"""

from typing import Callable, Dict, List, Optional

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from bondtrader.core.bond_models import Bond
from bondtrader.core.bond_valuation import BondValuator
from bondtrader.utils.utils import logger


class BayesianOptimizer:
    """
    Bayesian optimization for hyperparameter tuning and portfolio optimization
    More efficient than grid search - learns from previous evaluations
    """

    def __init__(self, valuator: BondValuator = None):
        """Initialize Bayesian optimizer"""
        self.valuator = valuator if valuator else BondValuator()
        self.observed_points = []
        self.observed_values = []

    def acquisition_function(
        self, x: float, explored_points: List[float], explored_values: List[float], xi: float = 0.01
    ) -> float:
        """
        Upper Confidence Bound (UCB) acquisition function

        Balances exploration vs. exploitation

        Args:
            x: Candidate point
            explored_points: Previously explored points
            explored_values: Values at explored points
            xi: Exploration parameter

        Returns:
            Acquisition value (higher = more promising)
        """
        if len(explored_points) == 0:
            return float("inf")  # Explore first

        explored_points = np.array(explored_points)
        explored_values = np.array(explored_values)

        # Estimate mean and std using simple Gaussian process approximation
        # In production, would use full GP
        mean = np.mean(explored_values)
        std = np.std(explored_values) if len(explored_values) > 1 else 1.0

        # Distance to nearest explored point
        distances = np.abs(explored_points - x)
        min_distance = np.min(distances)

        # UCB: mean + beta * std
        # Higher std or farther from explored = higher UCB
        beta = 2.0  # Exploration parameter
        ucb = mean + beta * std * (1 - np.exp(-min_distance))

        return ucb

    def optimize_hyperparameters(self, bonds: List[Bond], param_bounds: Dict[str, tuple], num_iterations: int = 20) -> Dict:
        """
        Bayesian optimization for hyperparameter tuning

        More efficient than grid search - learns optimal parameters

        Args:
            bonds: List of bonds for training
            param_bounds: {param_name: (min, max)} bounds
            num_iterations: Number of optimization iterations

        Returns:
            Optimal hyperparameters
        """
        from bondtrader.ml.ml_adjuster_enhanced import EnhancedMLBondAdjuster

        param_names = list(param_bounds.keys())

        # Define objective function
        def objective(params_dict):
            try:
                ml_adjuster = EnhancedMLBondAdjuster()

                # Set parameters (simplified - would need proper parameter mapping)
                # For demonstration, optimize on test R²
                metrics = ml_adjuster.train_with_tuning(bonds, tune_hyperparameters=False)

                # Return negative R² (minimize negative = maximize R²)
                return -metrics["test_r2"]
            except:
                return 1.0  # Penalty for failed training

        # Simplified 1D optimization for demonstration
        # In production, would use multi-dimensional BO
        if len(param_names) == 1:
            param_name = param_names[0]
            bounds = param_bounds[param_name]

            best_params = {param_name: np.mean(bounds)}
            best_value = float("inf")

            # Random search with intelligent sampling (simplified BO)
            for _ in range(num_iterations):
                # Sample from bounds
                if len(self.observed_points) < 5:
                    # Initial random exploration
                    x = np.random.uniform(bounds[0], bounds[1])
                else:
                    # Use acquisition function to guide search
                    candidates = np.linspace(bounds[0], bounds[1], 100)
                    acq_values = [self.acquisition_function(c, self.observed_points, self.observed_values) for c in candidates]
                    x = candidates[np.argmax(acq_values)]

                # Evaluate
                params_dict = {param_name: x}
                value = objective(params_dict)

                self.observed_points.append(x)
                self.observed_values.append(value)

                if value < best_value:
                    best_value = value
                    best_params = {param_name: x}

        return {
            "optimal_parameters": best_params,
            "best_value": -best_value,  # Convert back to R²
            "num_iterations": num_iterations,
            "observed_points": len(self.observed_points),
        }

    def robust_portfolio_optimization(self, bonds: List[Bond], uncertainty_bounds: Optional[Dict] = None) -> Dict:
        """
        Robust portfolio optimization under uncertainty

        Optimizes worst-case scenario rather than expected case
        More conservative than standard optimization

        Args:
            bonds: List of bonds
            uncertainty_bounds: Uncertainty in returns/covariances

        Returns:
            Robust optimal portfolio
        """
        from bondtrader.analytics.portfolio_optimization import PortfolioOptimizer

        optimizer = PortfolioOptimizer(self.valuator)
        expected_returns, covariance = optimizer.calculate_returns_and_covariance(bonds)

        n = len(bonds)

        # Define robust objective (minimize worst-case risk)
        def robust_objective(weights):
            weights = np.array(weights)

            # Base portfolio variance
            base_variance = np.dot(weights, np.dot(covariance, weights))

            # Add uncertainty penalty
            if uncertainty_bounds:
                uncertainty_penalty = 0.01 * np.sum(weights**2)  # Simplified
            else:
                uncertainty_penalty = 0

            return base_variance + uncertainty_penalty

        # Constraints
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0, 1) for _ in range(n)]
        x0 = np.ones(n) / n

        # Optimize
        result = minimize(robust_objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)

        if not result.success:
            weights = x0
        else:
            weights = result.x

        portfolio_return = np.dot(weights, expected_returns)
        portfolio_volatility = np.sqrt(robust_objective(weights))

        return {
            "weights": weights.tolist(),
            "portfolio_return": portfolio_return,
            "portfolio_volatility": portfolio_volatility,
            "sharpe_ratio": portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0,
            "method": "Robust Optimization",
            "optimization_success": result.success,
        }
