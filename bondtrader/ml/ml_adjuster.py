"""
Machine Learning Model for Bond Price Adjustments
Uses regression to predict fair value adjustments based on market factors
"""

import os
from typing import List, Tuple

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Optional ML libraries
try:
    from xgboost import XGBRegressor

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMRegressor

    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from catboost import CatBoostRegressor

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

from bondtrader.core.bond_models import Bond
from bondtrader.core.bond_valuation import BondValuator


class MLBondAdjuster:
    """
    Machine Learning model to adjust bond valuations
    Learns from market data and bond characteristics to improve price predictions
    """

    def __init__(self, model_type: str = "random_forest"):
        """
        Initialize ML adjuster

        Args:
            model_type: 'random_forest', 'gradient_boosting', 'xgboost', 'lightgbm', or 'catboost'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.valuator = BondValuator()

        # Validate model type
        available_models = ["random_forest", "gradient_boosting"]
        if HAS_XGBOOST:
            available_models.append("xgboost")
        if HAS_LIGHTGBM:
            available_models.append("lightgbm")
        if HAS_CATBOOST:
            available_models.append("catboost")

        if model_type not in available_models:
            raise ValueError(f"Model type '{model_type}' not available. Available models: {', '.join(available_models)}")

    def _create_features(self, bonds: List[Bond], fair_values: List[float]) -> np.ndarray:
        """Create feature matrix from bonds"""
        features = []
        for bond, fv in zip(bonds, fair_values):
            char = bond.get_bond_characteristics()

            # Base characteristics
            feature_vector = [
                char["coupon_rate"],
                char["time_to_maturity"],
                char["credit_rating_numeric"],
                char["current_price"] / char["face_value"],
                char["years_since_issue"],
                char["frequency"],
                char["callable"],
                char["convertible"],
            ]

            # Additional derived features
            ytm = self.valuator.calculate_yield_to_maturity(bond)
            duration = self.valuator.calculate_duration(bond, ytm)
            convexity = self.valuator.calculate_convexity(bond, ytm)

            # Price relative to fair value
            price_to_fair_ratio = bond.current_price / fv if fv > 0 else 1.0

            feature_vector.extend(
                [
                    ytm * 100,  # YTM as percentage
                    duration,
                    convexity,
                    price_to_fair_ratio,
                    bond.face_value,  # Size factor
                ]
            )

            features.append(feature_vector)

        return np.array(features)

    def _create_targets(self, bonds: List[Bond], fair_values: List[float]) -> np.ndarray:
        """
        Create target values for training
        Target is the adjustment factor: actual_price / theoretical_fair_value
        """
        targets = []
        for bond, fv in zip(bonds, fair_values):
            if fv > 0:
                # Adjustment factor (what we need to multiply fair value by)
                adjustment = bond.current_price / fv
                targets.append(adjustment)
            else:
                targets.append(1.0)  # No adjustment if fair value is 0

        return np.array(targets)

    def train(self, bonds: List[Bond], test_size: float = 0.2, random_state: int = 42) -> dict:
        """
        Train the ML model on bond data

        Args:
            bonds: List of bonds with market prices
            test_size: Proportion of data for testing
            random_state: Random seed

        Returns:
            Dictionary with training metrics
        """
        if len(bonds) < 10:
            raise ValueError("Need at least 10 bonds for training")

        # Calculate fair values for all bonds
        fair_values = [self.valuator.calculate_fair_value(bond) for bond in bonds]

        # Create features and targets
        X = self._create_features(bonds, fair_values)
        y = self._create_targets(bonds, fair_values)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Initialize and train model with improved default hyperparameters
        if self.model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=random_state,
                n_jobs=-1,
            )
        elif self.model_type == "gradient_boosting":
            self.model = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                min_samples_split=5,
                min_samples_leaf=2,
                subsample=0.9,
                random_state=random_state,
            )
        elif self.model_type == "xgboost" and HAS_XGBOOST:
            self.model = XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=random_state,
                n_jobs=-1,
                verbosity=0,
            )
        elif self.model_type == "lightgbm" and HAS_LIGHTGBM:
            self.model = LGBMRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=random_state,
                n_jobs=-1,
                verbose=-1,
            )
        elif self.model_type == "catboost" and HAS_CATBOOST:
            self.model = CatBoostRegressor(
                iterations=200,
                depth=6,
                learning_rate=0.1,
                random_state=random_state,
                verbose=False,
            )
        else:
            raise ValueError(
                f"Model type '{self.model_type}' not available. Please install required package or use 'random_forest' or 'gradient_boosting'"
            )

        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)

        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)

        self.is_trained = True

        return {
            "train_mse": train_mse,
            "test_mse": test_mse,
            "train_rmse": np.sqrt(train_mse),
            "test_rmse": np.sqrt(test_mse),
            "train_r2": train_r2,
            "test_r2": test_r2,
            "n_samples": len(bonds),
            "n_train": len(X_train),
            "n_test": len(X_test),
        }

    def predict_adjusted_value(self, bond: Bond) -> dict:
        """
        Predict ML-adjusted fair value for a bond

        Returns:
            Dictionary with adjusted fair value and adjustment factor
        """
        if not self.is_trained:
            # If not trained, return basic fair value
            fair_value = self.valuator.calculate_fair_value(bond)
            return {
                "theoretical_fair_value": fair_value,
                "ml_adjusted_fair_value": fair_value,
                "adjustment_factor": 1.0,
                "ml_confidence": 0.0,
            }

        # Calculate base fair value
        fair_value = self.valuator.calculate_fair_value(bond)

        # Create feature vector
        features = self._create_features([bond], [fair_value])
        features_scaled = self.scaler.transform(features)

        # Predict adjustment factor
        adjustment_factor = self.model.predict(features_scaled)[0]

        # Ensure reasonable bounds
        adjustment_factor = np.clip(adjustment_factor, 0.5, 1.5)

        # Calculate adjusted fair value
        ml_adjusted_value = fair_value * adjustment_factor

        return {
            "theoretical_fair_value": fair_value,
            "ml_adjusted_fair_value": ml_adjusted_value,
            "adjustment_factor": adjustment_factor,
            "ml_confidence": 0.8 if self.is_trained else 0.0,
        }

    def save_model(self, filepath: str):
        """
        Save trained model and scaler with atomic writes

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
            joblib.dump({"model": self.model, "scaler": self.scaler, "model_type": self.model_type}, temp_filepath)

            # Atomic rename (Unix) or copy+remove (Windows fallback)
            if os.name == "nt":  # Windows
                # On Windows, rename might fail if file exists, so remove first
                if os.path.exists(filepath):
                    os.remove(filepath)
                os.rename(temp_filepath, filepath)
            else:  # Unix/Linux/Mac
                os.rename(temp_filepath, filepath)

        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(temp_filepath):
                try:
                    os.remove(temp_filepath)
                except:
                    pass
            raise e

    def load_model(self, filepath: str):
        """Load trained model and scaler"""
        data = joblib.load(filepath)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.model_type = data.get("model_type", "random_forest")
        self.is_trained = True
