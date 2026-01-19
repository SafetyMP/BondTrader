"""
Bayesian Optimization Module
Advanced hyperparameter tuning and portfolio optimization
More efficient than grid search used by most platforms
"""

from typing import Callable, Dict, List, Optional

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

# Optional Optuna for advanced hyperparameter optimization
try:
    import optuna

    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

from bondtrader.core.bond_models import Bond
from bondtrader.core.bond_valuation import BondValuator
from bondtrader.utils.utils import logger


class BayesianOptimizer:
    """
    Bayesian optimization for hyperparameter tuning and portfolio optimization
    More efficient than grid search - learns from previous evaluations
    """

    def __init__(self, valuator: BondValuator = None):
        """
        Initialize Bayesian optimizer

        Args:
            valuator: Optional BondValuator instance. If None, gets from container.
        """
        if valuator is None:
            from bondtrader.core.container import get_container

            self.valuator = get_container().get_valuator()
        else:
            self.valuator = valuator
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

    def optimize_hyperparameters(
        self,
        bonds: List[Bond],
        param_bounds: Dict[str, tuple],
        num_iterations: int = 20,
        use_optuna: bool = False,
        model_type: str = "random_forest",
    ) -> Dict:
        """
        Bayesian optimization for hyperparameter tuning

        More efficient than grid search - learns optimal parameters

        Args:
            bonds: List of bonds for training
            param_bounds: {param_name: (min, max)} bounds
            num_iterations: Number of optimization iterations
            use_optuna: Use Optuna for optimization (more advanced if available)
            model_type: Type of model to optimize ('random_forest' or 'gradient_boosting')

        Returns:
            Optimal hyperparameters
        """
        # Use Optuna if requested and available
        if use_optuna and HAS_OPTUNA:
            return self._optimize_hyperparameters_optuna(bonds, param_bounds, num_iterations, model_type)

        import numpy as np
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        from sklearn.metrics import r2_score
        from sklearn.model_selection import TimeSeriesSplit, train_test_split
        from sklearn.preprocessing import StandardScaler

        from bondtrader.ml.ml_adjuster_enhanced import EnhancedMLBondAdjuster

        param_names = list(param_bounds.keys())

        # Prepare data once
        ml_adjuster = EnhancedMLBondAdjuster(model_type=model_type)
        fair_values = [ml_adjuster.valuator.calculate_fair_value(bond) for bond in bonds]
        X, feature_names = ml_adjuster._create_enhanced_features(bonds, fair_values)
        y = ml_adjuster._create_targets(bonds, fair_values)

        # Use time-based split for financial data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Define objective function that actually uses parameters
        def objective(params_dict):
            """Objective function that uses the provided parameters"""
            try:
                # Map optimization parameters to model hyperparameters
                model_params = {}

                # Handle integer parameters
                int_params = ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"]
                for param in int_params:
                    if param in params_dict:
                        model_params[param] = int(params_dict[param])

                # Handle float parameters
                float_params = ["learning_rate", "subsample", "min_samples_split", "min_samples_leaf"]
                for param in float_params:
                    if param in params_dict and param not in model_params:
                        model_params[param] = float(params_dict[param])

                # Handle string/categorical parameters
                if "max_features" in params_dict:
                    model_params["max_features"] = params_dict["max_features"]

                # Set default values for required parameters
                if model_type == "random_forest":
                    default_params = {
                        "n_estimators": 200,
                        "max_depth": 15,
                        "min_samples_split": 5,
                        "min_samples_leaf": 2,
                        "max_features": "sqrt",
                        "random_state": 42,
                        "n_jobs": -1,
                    }
                else:  # gradient_boosting
                    default_params = {
                        "n_estimators": 200,
                        "max_depth": 6,
                        "learning_rate": 0.1,
                        "min_samples_split": 5,
                        "min_samples_leaf": 2,
                        "subsample": 0.9,
                        "random_state": 42,
                    }

                # Merge defaults with provided params
                final_params = {**default_params, **model_params}

                # Create and train model
                if model_type == "random_forest":
                    model = RandomForestRegressor(**final_params)
                else:
                    model = GradientBoostingRegressor(**final_params)

                # Use TimeSeriesSplit for cross-validation
                tscv = TimeSeriesSplit(n_splits=5)
                cv_scores = []
                for train_idx, val_idx in tscv.split(X_train_scaled):
                    X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
                    y_tr, y_val = y_train[train_idx], y_train[val_idx]
                    model.fit(X_tr, y_tr)
                    y_pred = model.predict(X_val)
                    cv_scores.append(r2_score(y_val, y_pred))

                # Return negative mean CV score (minimize negative = maximize R²)
                mean_cv_score = np.mean(cv_scores)
                return -mean_cv_score

            except Exception as e:
                logger.warning(f"Objective evaluation failed: {e}")
                return 1.0  # Penalty for failed training

        # Multi-dimensional optimization
        best_params = {}
        best_value = float("inf")
        self.observed_points = []
        self.observed_values = []

        # Random initialization for multi-dimensional space
        for iteration in range(num_iterations):
            # Sample parameters from bounds
            params_dict = {}
            for param_name, (min_val, max_val) in param_bounds.items():
                if isinstance(min_val, int) and isinstance(max_val, int):
                    params_dict[param_name] = np.random.randint(min_val, max_val + 1)
                else:
                    params_dict[param_name] = np.random.uniform(min_val, max_val)

            # Evaluate objective
            value = objective(params_dict)

            # Store observation
            self.observed_points.append(params_dict.copy())
            self.observed_values.append(value)

            # Update best if improved
            if value < best_value:
                best_value = value
                best_params = params_dict.copy()

            # Log progress every 5 iterations
            if (iteration + 1) % 5 == 0:
                logger.info(f"Bayesian optimization iteration {iteration + 1}/{num_iterations}, best R²: {-best_value:.4f}")

        return {
            "optimal_parameters": best_params,
            "best_value": -best_value,  # Convert back to R²
            "num_iterations": num_iterations,
            "observed_points": len(self.observed_points),
            "method": "Bayesian Optimization (Simplified)",
        }

    def _optimize_hyperparameters_optuna(
        self, bonds: List[Bond], param_bounds: Dict[str, tuple], num_iterations: int, model_type: str = "random_forest"
    ) -> Dict:
        """Optimize hyperparameters using Optuna (if available)"""
        if not HAS_OPTUNA:
            raise ImportError("Optuna not installed. Install with: pip install optuna")

        import numpy as np
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        from sklearn.metrics import r2_score
        from sklearn.model_selection import TimeSeriesSplit, train_test_split
        from sklearn.preprocessing import StandardScaler

        from bondtrader.ml.ml_adjuster_enhanced import EnhancedMLBondAdjuster

        # Prepare data once
        ml_adjuster = EnhancedMLBondAdjuster(model_type=model_type)
        fair_values = [ml_adjuster.valuator.calculate_fair_value(bond) for bond in bonds]
        X, feature_names = ml_adjuster._create_enhanced_features(bonds, fair_values)
        y = ml_adjuster._create_targets(bonds, fair_values)

        # Use time-based split for financial data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        def objective(trial):
            """Optuna objective function that actually uses parameters"""
            try:
                # Suggest parameters based on bounds
                params = {}
                for param_name, (min_val, max_val) in param_bounds.items():
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        params[param_name] = trial.suggest_int(param_name, min_val, max_val)
                    else:
                        params[param_name] = trial.suggest_float(param_name, min_val, max_val)

                # Map to model hyperparameters
                model_params = {}
                int_params = ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"]
                for param in int_params:
                    if param in params:
                        model_params[param] = int(params[param])

                float_params = ["learning_rate", "subsample"]
                for param in float_params:
                    if param in params:
                        model_params[param] = float(params[param])

                if "max_features" in params:
                    model_params["max_features"] = params["max_features"]

                # Set defaults
                if model_type == "random_forest":
                    default_params = {"random_state": 42, "n_jobs": -1}
                    final_params = {**default_params, **model_params}
                    model = RandomForestRegressor(**final_params)
                else:
                    default_params = {"random_state": 42}
                    final_params = {**default_params, **model_params}
                    model = GradientBoostingRegressor(**final_params)

                # Use TimeSeriesSplit for cross-validation
                tscv = TimeSeriesSplit(n_splits=5)
                cv_scores = []
                for train_idx, val_idx in tscv.split(X_train_scaled):
                    X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
                    y_tr, y_val = y_train[train_idx], y_train[val_idx]
                    model.fit(X_tr, y_tr)
                    y_pred = model.predict(X_val)
                    cv_scores.append(r2_score(y_val, y_pred))

                # Return mean CV score (Optuna maximizes)
                return np.mean(cv_scores)

            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return 0.0  # Return worst score on error

        # Create study and optimize
        study = optuna.create_study(direction="maximize", study_name="bond_ml_optimization")
        study.optimize(objective, n_trials=num_iterations, show_progress_bar=False)

        return {
            "optimal_parameters": study.best_params,
            "best_value": study.best_value,
            "num_iterations": num_iterations,
            "observed_points": len(study.trials),
            "method": "Optuna",
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
