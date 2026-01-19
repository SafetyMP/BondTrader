"""
Unified Machine Learning Model for Bond Price Adjustments
Consolidates MLBondAdjuster, EnhancedMLBondAdjuster, and AdvancedMLBondAdjuster
Uses shared feature engineering and persistence modules to eliminate duplication
"""

import copy
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    RandomizedSearchCV,
    TimeSeriesSplit,
    cross_val_score,
    train_test_split,
)
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from bondtrader.core.bond_models import Bond
from bondtrader.core.bond_valuation import BondValuator
from bondtrader.ml.feature_engineering import BondFeatureEngineer
from bondtrader.ml.model_persistence import ModelPersistence
from bondtrader.utils.cache import ModelCache, cache_model
from bondtrader.utils.utils import logger

warnings.filterwarnings("ignore")

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

# MLflow tracking (optional)
try:
    from bondtrader.ml.mlflow_tracking import MLflowTracker

    HAS_MLFLOW_TRACKING = True
except ImportError:
    HAS_MLFLOW_TRACKING = False


class MLBondAdjuster:
    """
    Unified ML model to adjust bond valuations

    Supports multiple feature levels and training modes:
    - feature_level: "basic", "enhanced", "advanced"
    - use_ensemble: True for ensemble training (advanced features)
    - tune_hyperparameters: True for hyperparameter tuning
    """

    def __init__(
        self,
        model_type: str = "random_forest",
        feature_level: str = "basic",
        valuator: Optional[BondValuator] = None,
        use_ensemble: bool = False,
    ):
        """
        Initialize unified ML adjuster

        Args:
            model_type: 'random_forest', 'gradient_boosting', 'xgboost', 'lightgbm', or 'catboost'
            feature_level: 'basic', 'enhanced', or 'advanced'
            valuator: Optional BondValuator instance. If None, gets from container.
            use_ensemble: If True, uses ensemble training (requires feature_level='advanced')
        """
        self.model_type = model_type
        self.feature_level = feature_level
        self.use_ensemble = use_ensemble
        self.model = None
        self.models = {}  # For ensemble mode
        self.ensemble_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.best_params = None
        self.feature_names = []
        self.training_metrics = {}

        # Use provided valuator or get from container
        if valuator is None:
            from bondtrader.core.container import get_container

            self.valuator = get_container().get_valuator()
        else:
            self.valuator = valuator

        # Validate model type
        available_models = ["random_forest", "gradient_boosting"]
        if HAS_XGBOOST:
            available_models.append("xgboost")
        if HAS_LIGHTGBM:
            available_models.append("lightgbm")
        if HAS_CATBOOST:
            available_models.append("catboost")

        if model_type not in available_models:
            raise ValueError(
                f"Model type '{model_type}' not available. Available: {', '.join(available_models)}"
            )

        if use_ensemble and feature_level != "advanced":
            logger.warning(
                "Ensemble mode requires feature_level='advanced'. Setting feature_level='advanced'"
            )
            self.feature_level = "advanced"

        # Model versioning (for advanced features)
        self.model_versions = []
        self.current_model_version = None
        self._previous_models = None
        self._previous_ensemble = None
        self._previous_scaler = None

    def _create_features(
        self, bonds: List[Bond], fair_values: List[float]
    ) -> Tuple[np.ndarray, Optional[List[str]]]:
        """Create features using shared feature engineering module"""
        if self.feature_level == "basic":
            features = BondFeatureEngineer.create_basic_features(bonds, fair_values, self.valuator)
            return features, None
        elif self.feature_level == "enhanced":
            features, feature_names = BondFeatureEngineer.create_enhanced_features(
                bonds, fair_values, self.valuator
            )
            return features, feature_names
        else:  # advanced
            features, feature_names = BondFeatureEngineer.create_advanced_features(
                bonds, fair_values, self.valuator
            )
            return features, feature_names

    def _create_targets(self, bonds: List[Bond], fair_values: List[float]) -> np.ndarray:
        """Create targets using shared feature engineering module"""
        return BondFeatureEngineer.create_targets(bonds, fair_values)

    def train(
        self,
        bonds: List[Bond],
        test_size: float = 0.2,
        random_state: int = 42,
        tune_hyperparameters: bool = False,
        use_mlflow: bool = False,
        mlflow_run_name: Optional[str] = None,
    ) -> Dict:
        """
        Train the ML model

        Args:
            bonds: List of bonds with market prices
            test_size: Proportion of data for testing
            random_state: Random seed
            tune_hyperparameters: If True, performs hyperparameter tuning
            use_mlflow: If True, logs to MLflow
            mlflow_run_name: Optional MLflow run name

        Returns:
            Dictionary with training metrics
        """
        if len(bonds) < 10:
            raise ValueError("Need at least 10 bonds for training")

        if self.use_ensemble:
            return self.train_ensemble(bonds, test_size, random_state)

        # Initialize MLflow tracking
        mlflow_tracker = None
        if use_mlflow and HAS_MLFLOW_TRACKING:
            try:
                mlflow_tracker = MLflowTracker()
                run_name = (
                    mlflow_run_name
                    or f"ml_{self.model_type}_{self.feature_level}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                mlflow_tracker.start_run(
                    run_name=run_name,
                    tags={"model_type": self.model_type, "feature_level": self.feature_level},
                )
            except Exception as e:
                logger.warning(f"Failed to initialize MLflow tracking: {e}")
                mlflow_tracker = None

        try:
            # Calculate fair values
            fair_values = [self.valuator.calculate_fair_value(bond) for bond in bonds]

            # Create features and targets
            X, feature_names = self._create_features(bonds, fair_values)
            if feature_names:
                self.feature_names = feature_names
            y = self._create_targets(bonds, fair_values)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Hyperparameter tuning
            if tune_hyperparameters:
                best_params = self._tune_hyperparameters(X_train_scaled, y_train, random_state)
                self.best_params = best_params
            else:
                best_params = self._get_default_params()

            # Log to MLflow
            if mlflow_tracker:
                mlflow_tracker.log_params(
                    {
                        "model_type": self.model_type,
                        "feature_level": self.feature_level,
                        "test_size": test_size,
                        "random_state": random_state,
                        "tune_hyperparameters": tune_hyperparameters,
                        "n_samples": len(bonds),
                        "n_features": X.shape[1],
                    }
                )
                mlflow_tracker.log_params({f"param_{k}": v for k, v in best_params.items()})

            # Create and train model
            self.model = self._create_model(best_params, random_state)
            self.model.fit(X_train_scaled, y_train)

            # Evaluate
            train_pred = self.model.predict(X_train_scaled)
            test_pred = self.model.predict(X_test_scaled)

            metrics = {
                "train_mse": mean_squared_error(y_train, train_pred),
                "test_mse": mean_squared_error(y_test, test_pred),
                "train_rmse": np.sqrt(mean_squared_error(y_train, train_pred)),
                "test_rmse": np.sqrt(mean_squared_error(y_test, test_pred)),
                "train_r2": r2_score(y_train, train_pred),
                "test_r2": r2_score(y_test, test_pred),
                "n_samples": len(bonds),
                "n_train": len(X_train),
                "n_test": len(X_test),
            }

            # Add MAE for enhanced/advanced
            if self.feature_level in ["enhanced", "advanced"]:
                metrics["train_mae"] = mean_absolute_error(y_train, train_pred)
                metrics["test_mae"] = mean_absolute_error(y_test, test_pred)

            # Cross-validation for enhanced/advanced
            if self.feature_level in ["enhanced", "advanced"]:
                tscv = TimeSeriesSplit(n_splits=5)
                cv_scores = cross_val_score(
                    self.model, X_train_scaled, y_train, cv=tscv, scoring="r2"
                )
                metrics["cv_r2_mean"] = cv_scores.mean()
                metrics["cv_r2_std"] = cv_scores.std()

            if tune_hyperparameters:
                metrics["best_params"] = best_params

            self.training_metrics = metrics
            self.is_trained = True

            # Log to MLflow
            if mlflow_tracker:
                mlflow_tracker.log_metrics(metrics)
                input_example = X_test_scaled[:1] if len(X_test_scaled) > 0 else None
                mlflow_tracker.log_model(
                    model=self.model, artifact_path="model", input_example=input_example
                )
                mlflow_tracker.end_run()

            return metrics

        except Exception as e:
            if mlflow_tracker:
                mlflow_tracker.end_run()
            raise

    def train_ensemble(
        self, bonds: List[Bond], test_size: float = 0.2, random_state: int = 42
    ) -> Dict:
        """Train ensemble of multiple models (advanced feature)"""
        if len(bonds) < 20:
            raise ValueError("Need at least 20 bonds for ensemble training")

        if self.feature_level != "advanced":
            raise ValueError("Ensemble training requires feature_level='advanced'")

        # Calculate fair values
        fair_values = [self.valuator.calculate_fair_value(bond) for bond in bonds]

        # Create features
        X, feature_names = self._create_features(bonds, fair_values)
        self.feature_names = feature_names
        y = self._create_targets(bonds, fair_values)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train individual models
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

        self.models = {
            "random_forest": rf_model,
            "gradient_boosting": gb_model,
            "neural_network": nn_model,
        }

        # Create ensemble (stacking)
        base_models = [("rf", rf_model), ("gb", gb_model), ("nn", nn_model)]
        meta_model = GradientBoostingRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=random_state
        )
        self.ensemble_model = StackingRegressor(
            estimators=base_models, final_estimator=meta_model, cv=5
        )
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
            "improvement_over_best": (
                ensemble_metrics["test_r2"] - max(m["test_r2"] for m in models_eval.values())
            ),
            "model_version": self.current_model_version,
        }

    def _get_default_params(self) -> Dict:
        """Get default hyperparameters based on model type"""
        if self.model_type == "random_forest":
            return {"n_estimators": 200, "max_depth": 15, "min_samples_split": 5}
        else:
            return {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1}

    def _create_model(self, params: Dict, random_state: int):
        """Create model instance with given parameters"""
        if self.model_type == "random_forest":
            return RandomForestRegressor(
                n_estimators=params.get("n_estimators", 200),
                max_depth=params.get("max_depth", 15),
                min_samples_split=params.get("min_samples_split", 5),
                min_samples_leaf=params.get("min_samples_leaf", 2),
                max_features=params.get("max_features", "sqrt"),
                random_state=random_state,
                n_jobs=-1,
            )
        elif self.model_type == "gradient_boosting":
            return GradientBoostingRegressor(
                n_estimators=params.get("n_estimators", 200),
                max_depth=params.get("max_depth", 6),
                learning_rate=params.get("learning_rate", 0.1),
                min_samples_split=params.get("min_samples_split", 5),
                min_samples_leaf=params.get("min_samples_leaf", 2),
                subsample=params.get("subsample", 0.9),
                random_state=random_state,
                validation_fraction=0.1,
                n_iter_no_change=10,
                tol=1e-4,
            )
        elif self.model_type == "xgboost" and HAS_XGBOOST:
            return XGBRegressor(
                n_estimators=params.get("n_estimators", 200),
                max_depth=params.get("max_depth", 6),
                learning_rate=params.get("learning_rate", 0.1),
                subsample=params.get("subsample", 0.9),
                colsample_bytree=params.get("colsample_bytree", 0.9),
                random_state=random_state,
                n_jobs=-1,
                verbosity=0,
            )
        elif self.model_type == "lightgbm" and HAS_LIGHTGBM:
            return LGBMRegressor(
                n_estimators=params.get("n_estimators", 200),
                max_depth=params.get("max_depth", 6),
                learning_rate=params.get("learning_rate", 0.1),
                subsample=params.get("subsample", 0.9),
                colsample_bytree=params.get("colsample_bytree", 0.9),
                random_state=random_state,
                n_jobs=-1,
                verbose=-1,
            )
        elif self.model_type == "catboost" and HAS_CATBOOST:
            return CatBoostRegressor(
                iterations=params.get("n_estimators", 200),
                depth=params.get("max_depth", 6),
                learning_rate=params.get("learning_rate", 0.1),
                random_state=random_state,
                verbose=False,
            )
        else:
            raise ValueError(f"Model type '{self.model_type}' not available")

    def _tune_hyperparameters(
        self, X_train: np.ndarray, y_train: np.ndarray, random_state: int = 42
    ) -> Dict:
        """Tune hyperparameters using randomized search"""
        if self.model_type == "random_forest":
            param_distributions = {
                "n_estimators": [100, 200, 300, 400, 500],
                "max_depth": [5, 10, 15, 20, 25, None],
                "min_samples_split": [2, 5, 10, 20],
                "min_samples_leaf": [1, 2, 4, 8],
                "max_features": ["sqrt", "log2", None],
            }
            base_model = RandomForestRegressor(random_state=random_state, n_jobs=-1)
        else:  # gradient_boosting
            param_distributions = {
                "n_estimators": [100, 200, 300, 400, 500],
                "max_depth": [3, 4, 5, 6, 7, 8, 9],
                "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
                "min_samples_split": [2, 5, 10, 20],
                "min_samples_leaf": [1, 2, 4],
                "subsample": [0.8, 0.85, 0.9, 0.95, 1.0],
            }
            base_model = GradientBoostingRegressor(
                random_state=random_state, validation_fraction=0.1, n_iter_no_change=10, tol=1e-4
            )

        tscv = TimeSeriesSplit(n_splits=5)
        random_search = RandomizedSearchCV(
            base_model,
            param_distributions,
            n_iter=25,
            cv=tscv,
            scoring="r2",
            n_jobs=-1,
            verbose=0,
            random_state=random_state,
        )
        random_search.fit(X_train, y_train)
        return random_search.best_params_

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
        X, _ = self._create_features([bond], [fair_value])
        X_scaled = self.scaler.transform(X)

        if self.use_ensemble and self.ensemble_model is not None:
            adjustment_factor = self.ensemble_model.predict(X_scaled)[0]
        else:
            adjustment_factor = self.model.predict(X_scaled)[0]

        adjustment_factor = np.clip(adjustment_factor, 0.5, 1.5)
        ml_adjusted_value = fair_value * adjustment_factor

        return {
            "theoretical_fair_value": fair_value,
            "ml_adjusted_fair_value": ml_adjusted_value,
            "adjustment_factor": adjustment_factor,
            "ml_confidence": 0.8 if self.is_trained else 0.0,
        }

    def get_feature_importance(self) -> Dict:
        """Get feature importance from trained model"""
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained yet")

        if hasattr(self.model, "feature_importances_") and self.feature_names:
            importances = self.model.feature_importances_
            feature_importance = dict(zip(self.feature_names, importances))
            return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        return {}

    def save_model(self, filepath: str):
        """Save trained model using shared persistence module"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "model_type": self.model_type,
            "feature_level": self.feature_level,
            "use_ensemble": self.use_ensemble,
        }

        if self.best_params:
            model_data["best_params"] = self.best_params
        if self.feature_names:
            model_data["feature_names"] = self.feature_names
        if self.training_metrics:
            model_data["training_metrics"] = self.training_metrics
        if self.use_ensemble:
            model_data["models"] = self.models
            model_data["ensemble_model"] = self.ensemble_model
            model_data["current_model_version"] = self.current_model_version
            model_data["model_versions"] = self.model_versions

        ModelPersistence.save_model(model_data, filepath)

    def load_model(self, filepath: str):
        """
        Load trained model using shared persistence module with caching.

        CRITICAL: Models are cached in memory to avoid reloading on every request.
        """
        # Generate cache key
        cache_key = cache_model(self.model_type, self.feature_level, self.use_ensemble)

        # Try to get from cache first
        cached_data = ModelCache.get(cache_key)
        if cached_data is not None:
            logger.info(f"Loading model from cache: {cache_key}")
            data = cached_data
        else:
            # Load from file
            logger.info(f"Loading model from file: {filepath}")
            data = ModelPersistence.load_model(filepath, required_keys=["model", "scaler"])
            # Cache the loaded model
            ModelCache.set(cache_key, data)

        self.model = data["model"]
        self.scaler = data["scaler"]
        self.model_type = data.get("model_type", "random_forest")
        self.feature_level = data.get("feature_level", "basic")
        self.use_ensemble = data.get("use_ensemble", False)
        self.best_params = data.get("best_params")
        self.feature_names = data.get("feature_names", [])
        self.training_metrics = data.get("training_metrics", {})

        if self.use_ensemble:
            self.models = data.get("models", {})
            self.ensemble_model = data.get("ensemble_model")
            self.current_model_version = data.get("current_model_version")
            self.model_versions = data.get("model_versions", [])

        self.is_trained = True
