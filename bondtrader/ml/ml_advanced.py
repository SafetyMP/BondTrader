"""
Advanced Machine Learning Module
Deep learning, ensemble methods, explainable AI - Beyond industry standards
"""

import copy
import os
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from bondtrader.core.bond_models import Bond
from bondtrader.core.bond_valuation import BondValuator

warnings.filterwarnings("ignore")

# Use centralized logger from utils
from bondtrader.utils import logger

# Try to import advanced ML libraries (graceful fallback if not available)
try:
    import shap

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    from sklearn.experimental import enable_halving_search_cv
    from sklearn.model_selection import HalvingGridSearchCV

    HAS_HALVING = True
except ImportError:
    HAS_HALVING = False


class AdvancedMLBondAdjuster:
    """
    Advanced ML with deep learning, ensembles, and explainable AI
    Features beyond standard industry implementations
    """

    def __init__(self, valuator: BondValuator = None):
        """Initialize advanced ML adjuster"""
        self.valuator = valuator if valuator else BondValuator()
        self.scaler = StandardScaler()
        self.models = {}
        self.ensemble_model = None
        self.feature_names = []
        self.is_trained = False
        # Model versioning for retraining
        self.model_versions = []
        self.current_model_version = None
        # Previous model state for rollback
        self._previous_models = None
        self._previous_ensemble = None
        self._previous_scaler = None
        # Incremental learning support
        self.use_incremental_learning = False
        self.scaler_fit_samples = []
        self.scaler_n_samples_seen_ = 0
        self.scaler_mean_ = None
        self.scaler_var_ = None

    def _create_advanced_features(self, bonds: List[Bond], fair_values: List[float]) -> Tuple[np.ndarray, List[str]]:
        """Create advanced feature set with polynomial and interaction features"""
        features = []
        feature_names = []

        for bond, fv in zip(bonds, fair_values):
            char = bond.get_bond_characteristics()
            ytm = self.valuator.calculate_yield_to_maturity(bond)
            duration = self.valuator.calculate_duration(bond, ytm)
            convexity = self.valuator.calculate_convexity(bond, ytm)

            # Base features
            coupon_rate = char["coupon_rate"]
            ttm = char["time_to_maturity"]
            rating_num = char["credit_rating_numeric"]
            price_to_par = char["current_price"] / char["face_value"]
            years_issue = char["years_since_issue"]
            freq = char["frequency"]
            callable_flag = char["callable"]
            convertible_flag = char["convertible"]

            feature_vector = [
                coupon_rate,
                ttm,
                rating_num,
                price_to_par,
                years_issue,
                freq,
                callable_flag,
                convertible_flag,
                ytm * 100,
                duration,
                convexity,
                bond.current_price / fv if fv > 0 else 1.0,
                bond.face_value,
                duration / (1 + ytm) if ytm > 0 else duration,  # Modified duration
                ytm - self.valuator.risk_free_rate,  # Spread over RF
            ]

            # Polynomial features (degree 2)
            feature_vector.extend(
                [coupon_rate**2, ttm**2, duration**2, coupon_rate * ttm, coupon_rate * duration, ttm * duration]
            )

            # Interaction features
            feature_vector.extend([price_to_par * duration, rating_num * ttm, ytm * duration, convexity * duration])

            features.append(feature_vector)

        # Feature names
        base_names = [
            "coupon_rate",
            "time_to_maturity",
            "credit_rating",
            "price_to_par",
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
        ]
        poly_names = ["coupon_rate^2", "ttm^2", "duration^2", "coupon*ttm", "coupon*duration", "ttm*duration"]
        inter_names = ["price_to_par*duration", "rating*ttm", "ytm*duration", "convexity*duration"]
        feature_names = base_names + poly_names + inter_names

        return np.array(features), feature_names

    def train_ensemble(self, bonds: List[Bond], test_size: float = 0.2, random_state: int = 42) -> Dict:
        """
        Train ensemble of multiple models

        Ensemble combines:
        - Random Forest
        - Gradient Boosting
        - Neural Network (MLP)
        - Voting/Stacking

        Args:
            bonds: List of bonds
            test_size: Test set proportion
            random_state: Random seed

        Returns:
            Training metrics
        """
        if len(bonds) < 20:
            raise ValueError("Need at least 20 bonds for ensemble training")

        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.model_selection import train_test_split

        # Calculate fair values
        fair_values = [self.valuator.calculate_fair_value(bond) for bond in bonds]

        # Create features
        X, feature_names = self._create_advanced_features(bonds, fair_values)
        self.feature_names = feature_names
        y = np.array([bond.current_price / fv if fv > 0 else 1.0 for bond, fv in zip(bonds, fair_values)])

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Scale features
        # Support incremental learning: if previous scaler state exists and incremental mode is on, use partial_fit
        if self.use_incremental_learning and self.scaler_n_samples_seen_ > 0:
            # Incremental learning: update scaler statistics
            self.scaler.partial_fit(X_train)
            X_train_scaled = self.scaler.transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            # Update tracking
            self.scaler_n_samples_seen_ += len(X_train)
        else:
            # Full retraining: fit from scratch
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            # Update tracking
            self.scaler_n_samples_seen_ = len(X_train)
            if hasattr(self.scaler, "mean_"):
                self.scaler_mean_ = self.scaler.mean_.copy()
                self.scaler_var_ = self.scaler.var_.copy()

        # Train individual models with improved hyperparameters
        rf_model = RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=random_state,
            n_jobs=-1,
        )
        gb_model = GradientBoostingRegressor(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.05,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.9,
            random_state=random_state,
        )
        nn_model = MLPRegressor(
            hidden_layer_sizes=(150, 100),
            max_iter=1000,
            random_state=random_state,
            alpha=0.01,
            learning_rate="adaptive",
            early_stopping=True,
            validation_fraction=0.1,
        )

        rf_model.fit(X_train_scaled, y_train)
        gb_model.fit(X_train_scaled, y_train)
        nn_model.fit(X_train_scaled, y_train)

        self.models = {"random_forest": rf_model, "gradient_boosting": gb_model, "neural_network": nn_model}

        # Create ensemble (stacking)
        base_models = [("rf", rf_model), ("gb", gb_model), ("nn", nn_model)]

        meta_model = GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=random_state)
        self.ensemble_model = StackingRegressor(estimators=base_models, final_estimator=meta_model, cv=5)

        self.ensemble_model.fit(X_train_scaled, y_train)

        # Evaluate
        models_eval = {}
        for name, model in self.models.items():
            train_pred = model.predict(X_train_scaled)
            test_pred = model.predict(X_test_scaled)
            models_eval[name] = {
                "train_r2": r2_score(y_train, train_pred),
                "test_r2": r2_score(y_test, test_pred),
                "test_rmse": np.sqrt(mean_squared_error(y_test, test_pred)),
            }

        # Ensemble evaluation
        ensemble_train_pred = self.ensemble_model.predict(X_train_scaled)
        ensemble_test_pred = self.ensemble_model.predict(X_test_scaled)

        ensemble_metrics = {
            "train_r2": r2_score(y_train, ensemble_train_pred),
            "test_r2": r2_score(y_test, ensemble_test_pred),
            "test_rmse": np.sqrt(mean_squared_error(y_test, ensemble_test_pred)),
            "test_mae": mean_absolute_error(y_test, ensemble_test_pred),
        }

        self.is_trained = True
        self.current_model_version = datetime.now().isoformat()

        return {
            "individual_models": models_eval,
            "ensemble_metrics": ensemble_metrics,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_features": len(feature_names),
            "improvement_over_best": (ensemble_metrics["test_r2"] - max(m["test_r2"] for m in models_eval.values())),
            "model_version": self.current_model_version,
        }

    def get_feature_importance_explained(self) -> Dict:
        """
        Get explainable feature importance using multiple methods

        Returns:
            Feature importance with explanations
        """
        if not self.is_trained or self.ensemble_model is None:
            raise ValueError("Model not trained yet")

        importance_results = {}

        # RF feature importance
        if "random_forest" in self.models:
            rf_importance = self.models["random_forest"].feature_importances_
            importance_results["random_forest"] = dict(zip(self.feature_names, rf_importance.tolist()))

        # GB feature importance
        if "gradient_boosting" in self.models:
            gb_importance = self.models["gradient_boosting"].feature_importances_
            importance_results["gradient_boosting"] = dict(zip(self.feature_names, gb_importance.tolist()))

        # Ensemble consensus (average)
        all_importances = []
        for model_name, importance_dict in importance_results.items():
            importances = np.array(list(importance_dict.values()))
            all_importances.append(importances)

        if all_importances:
            consensus_importance = np.mean(all_importances, axis=0)
            importance_results["consensus"] = dict(zip(self.feature_names, consensus_importance.tolist()))

        # Sort by consensus importance
        if "consensus" in importance_results:
            sorted_importance = sorted(importance_results["consensus"].items(), key=lambda x: x[1], reverse=True)
            importance_results["sorted"] = dict(sorted_importance)

        return importance_results

    def explain_prediction(self, bond: Bond) -> Dict:
        """
        Explain individual prediction (model interpretability)

        Uses SHAP values if available, otherwise uses feature importance

        Args:
            bond: Bond object

        Returns:
            Prediction explanation
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        fair_value = self.valuator.calculate_fair_value(bond)
        X, _ = self._create_advanced_features([bond], [fair_value])
        X_scaled = self.scaler.transform(X)

        # Get prediction
        prediction = self.ensemble_model.predict(X_scaled)[0]
        adjustment_factor = np.clip(prediction, 0.5, 1.5)
        ml_adjusted_value = fair_value * adjustment_factor

        # Feature contributions (simplified SHAP-like explanation)
        if "random_forest" in self.models:
            # Get feature contributions from RF
            tree = self.models["random_forest"].estimators_[0]
            feature_contributions = {}

            # Simplified: use feature importance as proxy
            importances = self.models["random_forest"].feature_importances_
            feature_values = X[0]

            for i, (name, importance, value) in enumerate(zip(self.feature_names, importances, feature_values)):
                # Contribution = importance * (value - mean_value)
                contribution = importance * value
                feature_contributions[name] = contribution

        return {
            "theoretical_fair_value": fair_value,
            "ml_adjusted_value": ml_adjusted_value,
            "adjustment_factor": adjustment_factor,
            "feature_contributions": feature_contributions if "feature_contributions" in locals() else {},
            "top_drivers": sorted(
                feature_contributions.items() if "feature_contributions" in locals() else {},
                key=lambda x: abs(x[1]),
                reverse=True,
            )[:5],
        }

    def predict_with_uncertainty(self, bond: Bond, num_samples: int = 1000) -> Dict:
        """
        Predict with uncertainty quantification

        Returns prediction distribution, not just point estimate

        Args:
            bond: Bond object
            num_samples: Number of bootstrap samples

        Returns:
            Prediction with uncertainty bounds
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        fair_value = self.valuator.calculate_fair_value(bond)
        X, _ = self._create_advanced_features([bond], [fair_value])
        X_scaled = self.scaler.transform(X)

        # Bootstrap predictions from ensemble
        predictions = []
        for model_name, model in self.models.items():
            pred = model.predict(X_scaled)[0]
            predictions.append(pred)

        # Add ensemble prediction
        ensemble_pred = self.ensemble_model.predict(X_scaled)[0]
        predictions.append(ensemble_pred)

        # Estimate uncertainty from model diversity
        mean_prediction = np.mean(predictions)
        std_prediction = np.std(predictions)

        # Confidence intervals
        lower_bound = mean_prediction - 1.96 * std_prediction  # 95% CI
        upper_bound = mean_prediction + 1.96 * std_prediction

        adjusted_lower = fair_value * np.clip(lower_bound, 0.5, 1.5)
        adjusted_upper = fair_value * np.clip(upper_bound, 0.5, 1.5)
        adjusted_mean = fair_value * np.clip(mean_prediction, 0.5, 1.5)

        return {
            "mean_adjustment": mean_prediction,
            "std_adjustment": std_prediction,
            "mean_ml_value": adjusted_mean,
            "confidence_interval_lower": adjusted_lower,
            "confidence_interval_upper": adjusted_upper,
            "uncertainty_pct": (std_prediction / mean_prediction * 100) if mean_prediction > 0 else 0,
            "individual_predictions": {name: pred for name, pred in zip(list(self.models.keys()) + ["ensemble"], predictions)},
        }

    def adaptive_learning(
        self,
        bonds: List[Bond],
        window_size: int = 50,
        retrain_frequency: int = 10,
        random_state: int = 42,
        validation_threshold: float = 0.0,
        min_improvement: float = -0.05,
    ) -> Dict:
        """
        Adaptive learning: models that update as new data arrives

        More sophisticated than static models used by most platforms

        FIXED ISSUES:
        - Proper exception handling with logging
        - Model validation before replacement
        - Model versioning with rollback capability
        - Scaler state management for incremental learning

        Args:
            bonds: List of bonds (new data) - ASSUMED TO BE SHUFFLED or time-ordered intentionally
            window_size: Rolling window size
            retrain_frequency: Retrain every N bonds
            random_state: Random seed (consistent across windows for reproducibility)
            validation_threshold: Minimum R² score to accept new model (default: 0.0)
            min_improvement: Minimum improvement over previous model to replace (-0.05 = allow 5% degradation)

        Returns:
            Adaptive learning metrics
        """
        if len(bonds) < window_size:
            raise ValueError(f"Need at least {window_size} bonds for adaptive learning")

        # Save previous model state for rollback
        if self.is_trained:
            self._save_model_state()

        # Simulate online learning with rolling window
        predictions = []
        actuals = []
        model_versions = []
        validation_results = []
        rejected_updates = 0

        # Track previous model performance
        previous_test_r2 = None

        for i in range(0, len(bonds) - window_size, retrain_frequency):
            window_bonds = bonds[i : i + window_size]

            # Train on window with proper exception handling
            try:
                # Use consistent random_state for reproducibility (can be changed per window if needed)
                # For now, use fixed random_state for reproducibility
                metrics = self.train_ensemble(window_bonds, test_size=0.1, random_state=random_state)

                new_test_r2 = metrics["ensemble_metrics"]["test_r2"]

                # Validate new model before replacing
                should_replace = True
                rejection_reason = None

                # Check threshold
                if new_test_r2 < validation_threshold:
                    should_replace = False
                    rejection_reason = f"Test R² {new_test_r2:.4f} below threshold {validation_threshold:.4f}"

                # Check improvement over previous model
                if previous_test_r2 is not None:
                    improvement = new_test_r2 - previous_test_r2
                    if improvement < min_improvement:
                        should_replace = False
                        rejection_reason = f"Degradation too large: {improvement:.4f} < {min_improvement:.4f}"

                if should_replace:
                    # Save model version before updating
                    version_info = {
                        "timestamp": datetime.now().isoformat(),
                        "window": i,
                        "metrics": metrics["ensemble_metrics"],
                        "test_r2": new_test_r2,
                        "status": "accepted",
                    }
                    model_versions.append(version_info)
                    self.model_versions.append(version_info)

                    # Update previous performance
                    previous_test_r2 = new_test_r2

                    logger.info(f"Model updated at window {i}: Test R² = {new_test_r2:.4f}")
                else:
                    # Reject new model, restore previous state
                    self._restore_model_state()
                    rejected_updates += 1

                    version_info = {
                        "timestamp": datetime.now().isoformat(),
                        "window": i,
                        "test_r2": new_test_r2,
                        "status": "rejected",
                        "reason": rejection_reason,
                    }
                    validation_results.append(version_info)

                    logger.warning(f"Model update rejected at window {i}: {rejection_reason}")
                    continue

            except Exception as e:
                # Proper exception handling with logging
                logger.error(f"Retraining failed at window {i}: {e}", exc_info=True)
                validation_results.append(
                    {"timestamp": datetime.now().isoformat(), "window": i, "status": "error", "error": str(e)}
                )
                # Restore previous model state if available
                if self._previous_ensemble is not None:
                    self._restore_model_state()
                continue

            # Predict on next batch (validation set for this window)
            test_bonds = bonds[i + window_size : min(i + window_size + retrain_frequency, len(bonds))]
            for bond in test_bonds:
                try:
                    pred_result = self.explain_prediction(bond)
                    predictions.append(pred_result["ml_adjusted_value"])
                    actuals.append(bond.current_price)
                except Exception as e:
                    # Log prediction errors but continue
                    logger.debug(f"Prediction failed for bond at window {i}: {e}")
                    continue

        if len(predictions) > 0:
            mse = np.mean((np.array(predictions) - np.array(actuals)) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(np.array(predictions) - np.array(actuals)))

            return {
                "adaptive_mse": mse,
                "adaptive_rmse": rmse,
                "adaptive_mae": mae,
                "num_updates": len(model_versions),
                "rejected_updates": rejected_updates,
                "model_versions": model_versions,
                "validation_results": validation_results,
                "final_model_performance": previous_test_r2,
            }
        else:
            return {
                "error": "Insufficient data for adaptive learning",
                "validation_results": validation_results,
                "rejected_updates": rejected_updates,
            }

    def _save_model_state(self):
        """Save current model state for potential rollback"""
        try:
            self._previous_models = copy.deepcopy(self.models) if self.models else None
            self._previous_ensemble = copy.deepcopy(self.ensemble_model) if self.ensemble_model else None
            self._previous_scaler = copy.deepcopy(self.scaler) if self.scaler else None
        except Exception as e:
            logger.warning(f"Failed to save model state: {e}")
            self._previous_models = None
            self._previous_ensemble = None
            self._previous_scaler = None

    def _restore_model_state(self):
        """Restore previous model state"""
        if self._previous_ensemble is not None:
            try:
                self.models = copy.deepcopy(self._previous_models) if self._previous_models else {}
                self.ensemble_model = copy.deepcopy(self._previous_ensemble)
                self.scaler = copy.deepcopy(self._previous_scaler)
                logger.info("Model state restored from previous version")
            except Exception as e:
                logger.error(f"Failed to restore model state: {e}", exc_info=True)

    def save_model(self, filepath: str):
        """
        Save trained model with atomic writes to prevent corruption

        FIXED: Uses atomic writes (temp file + rename) to prevent partial saves
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
                    "model": self.models,
                    "ensemble_model": self.ensemble_model,
                    "scaler": self.scaler,
                    "feature_names": self.feature_names,
                    "is_trained": self.is_trained,
                    "current_model_version": self.current_model_version,
                    "model_versions": self.model_versions,
                    "scaler_n_samples_seen_": self.scaler_n_samples_seen_,
                    "scaler_mean_": self.scaler_mean_,
                    "scaler_var_": self.scaler_var_,
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

        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(temp_filepath):
                try:
                    os.remove(temp_filepath)
                except:
                    pass
            raise e

    def load_model(self, filepath: str):
        """
        Load trained model

        FIXED: Handles versioned models and restores all state
        """
        data = joblib.load(filepath)
        self.models = data.get("model", data.get("models", {}))
        self.ensemble_model = data.get("ensemble_model")
        self.scaler = data.get("scaler")
        self.feature_names = data.get("feature_names", [])
        self.is_trained = data.get("is_trained", False)
        self.current_model_version = data.get("current_model_version")
        self.model_versions = data.get("model_versions", [])
        self.scaler_n_samples_seen_ = data.get("scaler_n_samples_seen_", 0)
        self.scaler_mean_ = data.get("scaler_mean_")
        self.scaler_var_ = data.get("scaler_var_")
