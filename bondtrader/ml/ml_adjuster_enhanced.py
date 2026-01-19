"""
Enhanced Machine Learning Model for Bond Price Adjustments
Includes hyperparameter tuning, feature engineering, and model evaluation
"""

import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

from bondtrader.core.bond_models import Bond
from bondtrader.core.bond_valuation import BondValuator
from bondtrader.utils.utils import logger


class EnhancedMLBondAdjuster:
    """
    Enhanced ML model with hyperparameter tuning and advanced features
    """

    def __init__(self, model_type: str = "random_forest"):
        """Initialize enhanced ML adjuster"""
        self.model_type = model_type
        self.model = None
        self.best_params = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.valuator = BondValuator()
        self.feature_names = []
        self.training_metrics = {}

    def _create_enhanced_features(self, bonds: List[Bond], fair_values: List[float]) -> Tuple[np.ndarray, List[str]]:
        """Create enhanced feature matrix with macroeconomic and time features"""
        features = []
        feature_names = [
            "coupon_rate",
            "time_to_maturity",
            "credit_rating_numeric",
            "price_to_par_ratio",
            "years_since_issue",
            "frequency",
            "callable",
            "convertible",
            "ytm",
            "duration",
            "convexity",
            "price_to_fair_ratio",
            "face_value",
            "modified_duration",
            "spread_over_rf",
            "time_decay",
            "quarter",
            "day_of_year",
        ]

        current_date = datetime.now()

        for bond, fv in zip(bonds, fair_values):
            char = bond.get_bond_characteristics()
            ytm = self.valuator.calculate_yield_to_maturity(bond)
            duration = self.valuator.calculate_duration(bond, ytm)
            convexity = self.valuator.calculate_convexity(bond, ytm)

            # Base features
            feature_vector = [
                char["coupon_rate"],
                char["time_to_maturity"],
                char["credit_rating_numeric"],
                char["current_price"] / char["face_value"],
                char["years_since_issue"],
                char["frequency"],
                char["callable"],
                char["convertible"],
                ytm * 100,
                duration,
                convexity,
                bond.current_price / fv if fv > 0 else 1.0,
                bond.face_value,
            ]

            # Enhanced features
            modified_duration = duration / (1 + ytm) if ytm > 0 else duration
            spread_over_rf = ytm - self.valuator.risk_free_rate
            time_decay = (
                char["time_to_maturity"] / (char["years_since_issue"] + char["time_to_maturity"])
                if (char["years_since_issue"] + char["time_to_maturity"]) > 0
                else 0
            )
            quarter = current_date.month // 4 + 1
            day_of_year = current_date.timetuple().tm_yday

            feature_vector.extend(
                [modified_duration, spread_over_rf * 100, time_decay, quarter, day_of_year / 365.25]  # Normalized
            )

            features.append(feature_vector)

        return np.array(features), feature_names

    def _create_targets(self, bonds: List[Bond], fair_values: List[float]) -> np.ndarray:
        """Create target values for training"""
        targets = []
        for bond, fv in zip(bonds, fair_values):
            if fv > 0:
                adjustment = bond.current_price / fv
                targets.append(adjustment)
            else:
                targets.append(1.0)
        return np.array(targets)

    def train_with_tuning(
        self, bonds: List[Bond], test_size: float = 0.2, random_state: int = 42, tune_hyperparameters: bool = True
    ) -> Dict:
        """Train model with optional hyperparameter tuning"""
        if len(bonds) < 10:
            raise ValueError("Need at least 10 bonds for training")

        # Calculate fair values
        fair_values = [self.valuator.calculate_fair_value(bond) for bond in bonds]

        # Create features and targets
        X, feature_names = self._create_enhanced_features(bonds, fair_values)
        self.feature_names = feature_names
        y = self._create_targets(bonds, fair_values)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Hyperparameter tuning
        if tune_hyperparameters:
            best_params = self._tune_hyperparameters(X_train_scaled, y_train, random_state)
            self.best_params = best_params
        else:
            # Use default parameters
            if self.model_type == "random_forest":
                best_params = {"n_estimators": 100, "max_depth": 10, "min_samples_split": 5}
            else:
                best_params = {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1}

        # Train model with best parameters
        if self.model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=best_params["n_estimators"],
                max_depth=best_params.get("max_depth", 10),
                min_samples_split=best_params.get("min_samples_split", 5),
                random_state=random_state,
                n_jobs=-1,
            )
        elif self.model_type == "gradient_boosting":
            self.model = GradientBoostingRegressor(
                n_estimators=best_params["n_estimators"],
                max_depth=best_params.get("max_depth", 5),
                learning_rate=best_params.get("learning_rate", 0.1),
                random_state=random_state,
            )

        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)

        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring="r2")

        # Metrics
        metrics = {
            "train_mse": mean_squared_error(y_train, train_pred),
            "test_mse": mean_squared_error(y_test, test_pred),
            "train_rmse": np.sqrt(mean_squared_error(y_train, train_pred)),
            "test_rmse": np.sqrt(mean_squared_error(y_test, test_pred)),
            "train_mae": mean_absolute_error(y_train, train_pred),
            "test_mae": mean_absolute_error(y_test, test_pred),
            "train_r2": r2_score(y_train, train_pred),
            "test_r2": r2_score(y_test, test_pred),
            "cv_r2_mean": cv_scores.mean(),
            "cv_r2_std": cv_scores.std(),
            "n_samples": len(bonds),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "best_params": best_params,
        }

        self.training_metrics = metrics
        self.is_trained = True

        return metrics

    def _tune_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray, random_state: int = 42) -> Dict:
        """Tune hyperparameters using randomized search for better exploration"""
        from sklearn.model_selection import RandomizedSearchCV

        if self.model_type == "random_forest":
            # Expanded parameter space for better performance
            param_distributions = {
                "n_estimators": [100, 200, 300, 400, 500],
                "max_depth": [5, 10, 15, 20, 25, None],
                "min_samples_split": [2, 5, 10, 20],
                "min_samples_leaf": [1, 2, 4, 8],
                "max_features": ["sqrt", "log2", None],
            }
            base_model = RandomForestRegressor(random_state=random_state, n_jobs=-1)
            n_iter = 25  # Balance between exploration and speed
        else:  # gradient_boosting
            # Expanded parameter space for gradient boosting
            param_distributions = {
                "n_estimators": [100, 200, 300, 400, 500],
                "max_depth": [3, 4, 5, 6, 7, 8, 9],
                "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
                "min_samples_split": [2, 5, 10, 20],
                "min_samples_leaf": [1, 2, 4],
                "subsample": [0.8, 0.85, 0.9, 0.95, 1.0],
            }
            base_model = GradientBoostingRegressor(random_state=random_state)
            n_iter = 25  # Balance between exploration and speed

        # Use RandomizedSearchCV for more efficient search over larger space
        random_search = RandomizedSearchCV(
            base_model, param_distributions, n_iter=n_iter, cv=5, scoring="r2", n_jobs=-1, verbose=0, random_state=random_state
        )

        random_search.fit(X_train, y_train)
        return random_search.best_params_

    def get_feature_importance(self) -> Dict:
        """Get feature importance from trained model"""
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained yet")

        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            feature_importance = dict(zip(self.feature_names, importances))
            # Sort by importance
            sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            return dict(sorted_importance)
        else:
            return {}

    def predict_adjusted_value(self, bond: Bond) -> Dict:
        """Predict ML-adjusted fair value"""
        if not self.is_trained:
            fair_value = self.valuator.calculate_fair_value(bond)
            return {
                "theoretical_fair_value": fair_value,
                "ml_adjusted_fair_value": fair_value,
                "adjustment_factor": 1.0,
                "ml_confidence": 0.0,
            }

        fair_value = self.valuator.calculate_fair_value(bond)
        features, _ = self._create_enhanced_features([bond], [fair_value])
        features_scaled = self.scaler.transform(features)

        adjustment_factor = self.model.predict(features_scaled)[0]
        adjustment_factor = np.clip(adjustment_factor, 0.5, 1.5)

        ml_adjusted_value = fair_value * adjustment_factor

        return {
            "theoretical_fair_value": fair_value,
            "ml_adjusted_fair_value": ml_adjusted_value,
            "adjustment_factor": adjustment_factor,
            "ml_confidence": 0.8 if self.is_trained else 0.0,
        }

    def save_model(self, filepath: str):
        """
        Save trained model with atomic writes

        FIXED: Uses atomic writes (temp file + rename) to prevent corruption
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        # Create directory if needed
        dir_path = os.path.dirname(filepath) if os.path.dirname(filepath) else "."
        os.makedirs(dir_path, exist_ok=True)

        # Save to temporary file first (atomic on Unix, best-effort on Windows)
        temp_filepath = filepath + ".tmp"
        try:
            joblib.dump(
                {
                    "model": self.model,
                    "scaler": self.scaler,
                    "model_type": self.model_type,
                    "best_params": self.best_params,
                    "feature_names": self.feature_names,
                    "training_metrics": self.training_metrics,
                },
                temp_filepath,
            )

            # Atomic rename (Unix) or copy+remove (Windows fallback)
            if os.name == "nt":  # Windows
                # On Windows, rename might fail if file exists, so remove first
                if os.path.exists(filepath):
                    os.remove(filepath)
                os.rename(temp_filepath, filepath)
            else:  # Unix/Linux/Mac
                os.rename(temp_filepath, filepath)

        except (OSError, IOError, PermissionError) as e:
            # Clean up temp file on file I/O errors
            if os.path.exists(temp_filepath):
                try:
                    os.remove(temp_filepath)
                except OSError:
                    pass  # Ignore cleanup errors
            raise
        except Exception as e:
            # Clean up temp file on unexpected errors
            if os.path.exists(temp_filepath):
                try:
                    os.remove(temp_filepath)
                except OSError:
                    pass  # Ignore cleanup errors
            logger.error(f"Unexpected error saving model to {filepath}: {e}", exc_info=True)
            raise

    def load_model(self, filepath: str):
        """
        Load trained model

        Args:
            filepath: Path to saved model file

        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model file is corrupted or missing required keys
            TypeError: If model file contains invalid data types
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        try:
            data = joblib.load(filepath)
        except Exception as e:
            raise ValueError(f"Failed to load model from {filepath}: {e}") from e

        # Validate required keys
        required_keys = ["model", "scaler"]
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            raise ValueError(f"Model file missing required keys: {missing_keys}")

        try:
            self.model = data["model"]
            self.scaler = data["scaler"]
            self.model_type = data.get("model_type", "random_forest")
            self.best_params = data.get("best_params")
            self.feature_names = data.get("feature_names", [])
            self.training_metrics = data.get("training_metrics", {})
            self.is_trained = True
        except (KeyError, TypeError, AttributeError) as e:
            raise ValueError(f"Invalid model data structure: {e}") from e
