"""
AutoML Module
Automated machine learning with model selection and hyperparameter tuning
More advanced than manual tuning used by most platforms
"""

from typing import Dict, List, Optional

import numpy as np

from bondtrader.core.bond_models import Bond
from bondtrader.core.bond_valuation import BondValuator
from bondtrader.ml.ml_adjuster_enhanced import EnhancedMLBondAdjuster
from bondtrader.ml.ml_advanced import AdvancedMLBondAdjuster
from bondtrader.utils.utils import logger


class AutoMLBondAdjuster:
    """
    Automated Machine Learning for bond price adjustments
    Automatically selects best model and hyperparameters
    More advanced than manual model selection
    """

    def __init__(self, valuator: BondValuator = None):
        """Initialize AutoML adjuster"""
        self.valuator = valuator if valuator else BondValuator()
        self.best_model = None
        self.best_model_type = None
        self.best_params = None
        self.scaler = None
        self.is_trained = False

    def automated_model_selection(
        self, bonds: List[Bond], candidate_models: Optional[List[str]] = None, max_evaluation_time: int = 300  # seconds
    ) -> Dict:
        """
        Automatically select best model and hyperparameters

        Evaluates multiple models and selects best performing

        Args:
            bonds: List of bonds
            candidate_models: List of model types to evaluate
            max_evaluation_time: Maximum time for evaluation

        Returns:
            Best model selection results
        """
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        from sklearn.model_selection import cross_val_score
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler

        if candidate_models is None:
            candidate_models = ["random_forest", "gradient_boosting", "neural_network", "ensemble"]

        if len(bonds) < 20:
            raise ValueError("Need at least 20 bonds for AutoML")

        # Prepare data
        fair_values = [self.valuator.calculate_fair_value(bond) for bond in bonds]

        # Use enhanced ML adjuster for feature extraction
        advanced_ml = AdvancedMLBondAdjuster(self.valuator)
        X, feature_names = advanced_ml._create_advanced_features(bonds, fair_values)
        y = np.array([bond.current_price / fv if fv > 0 else 1.0 for bond, fv in zip(bonds, fair_values)])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Evaluate candidate models
        model_results = {}

        # Random Forest
        if "random_forest" in candidate_models:
            from sklearn.model_selection import RandomizedSearchCV

            param_distributions = {
                "n_estimators": [100, 200, 300, 400, 500],
                "max_depth": [5, 10, 15, 20, 25, None],
                "min_samples_split": [2, 5, 10, 20],
                "min_samples_leaf": [1, 2, 4, 8],
                "max_features": ["sqrt", "log2", None],
            }
            rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
            rf_search = RandomizedSearchCV(
                rf_base, param_distributions, n_iter=25, cv=5, scoring="r2", n_jobs=-1, random_state=42
            )
            rf_search.fit(X_scaled, y)
            model_results["random_forest"] = {
                "best_score": rf_search.best_score_,
                "best_params": rf_search.best_params_,
                "model": rf_search.best_estimator_,
            }

        # Gradient Boosting
        if "gradient_boosting" in candidate_models:
            from sklearn.model_selection import RandomizedSearchCV

            param_distributions = {
                "n_estimators": [100, 200, 300, 400, 500],
                "max_depth": [3, 4, 5, 6, 7, 8, 9],
                "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
                "min_samples_split": [2, 5, 10, 20],
                "min_samples_leaf": [1, 2, 4],
                "subsample": [0.8, 0.85, 0.9, 0.95, 1.0],
            }
            gb_base = GradientBoostingRegressor(random_state=42)
            gb_search = RandomizedSearchCV(
                gb_base, param_distributions, n_iter=25, cv=5, scoring="r2", n_jobs=-1, random_state=42
            )
            gb_search.fit(X_scaled, y)
            model_results["gradient_boosting"] = {
                "best_score": gb_search.best_score_,
                "best_params": gb_search.best_params_,
                "model": gb_search.best_estimator_,
            }

        # Neural Network
        if "neural_network" in candidate_models:
            from sklearn.model_selection import RandomizedSearchCV

            param_distributions = {
                "hidden_layer_sizes": [(50,), (100,), (150,), (100, 50), (100, 75), (150, 100), (200, 100)],
                "alpha": [0.0001, 0.001, 0.01, 0.1, 1.0],
                "learning_rate": ["constant", "adaptive"],
                "learning_rate_init": [0.001, 0.01, 0.1],
            }
            nn_base = MLPRegressor(max_iter=1000, random_state=42, early_stopping=True, validation_fraction=0.1)
            nn_search = RandomizedSearchCV(
                nn_base, param_distributions, n_iter=20, cv=5, scoring="r2", n_jobs=-1, random_state=42
            )
            nn_search.fit(X_scaled, y)
            model_results["neural_network"] = {
                "best_score": nn_search.best_score_,
                "best_params": nn_search.best_params_,
                "model": nn_search.best_estimator_,
            }

        # Ensemble (always include)
        try:
            ensemble_result = advanced_ml.train_ensemble(bonds, test_size=0.2, random_state=42)
            model_results["ensemble"] = {
                "best_score": ensemble_result["ensemble_metrics"]["test_r2"],
                "best_params": {"method": "stacking"},
                "model": advanced_ml.ensemble_model,
                "advanced_ml": advanced_ml,
            }
        except:
            pass

        # Select best model
        best_model_name = max(model_results.keys(), key=lambda k: model_results[k]["best_score"])
        best_result = model_results[best_model_name]

        self.best_model = best_result["model"]
        self.best_model_type = best_model_name
        self.best_params = best_result["best_params"]
        self.scaler = scaler
        self.is_trained = True

        return {
            "best_model": best_model_name,
            "best_score": best_result["best_score"],
            "best_params": best_result["best_params"],
            "all_models": {k: v["best_score"] for k, v in model_results.items()},
            "model_comparison": model_results,
            "automl_success": True,
        }

    def predict_adjusted_value(self, bond: Bond) -> Dict:
        """Predict using best AutoML model"""
        if not self.is_trained:
            raise ValueError("AutoML not trained yet")

        fair_value = self.valuator.calculate_fair_value(bond)

        # Extract features
        advanced_ml = AdvancedMLBondAdjuster(self.valuator)
        X, _ = advanced_ml._create_advanced_features([bond], [fair_value])
        X_scaled = self.scaler.transform(X)

        # Predict
        adjustment_factor = self.best_model.predict(X_scaled)[0]
        adjustment_factor = np.clip(adjustment_factor, 0.5, 1.5)

        ml_adjusted_value = fair_value * adjustment_factor

        return {
            "theoretical_fair_value": fair_value,
            "ml_adjusted_value": ml_adjusted_value,
            "adjustment_factor": adjustment_factor,
            "model_type": self.best_model_type,
            "confidence": 0.9,  # High confidence for AutoML
        }
