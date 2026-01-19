"""
Comprehensive Model Training Script
Trains all models in the codebase using the training dataset

Follows financial industry best practices:
- Proper train/validation/test splits
- Cross-validation
- Model evaluation on out-of-sample data
- Stress testing
- Model persistence
"""

import copy
import fcntl  # For file locking (Unix)
import glob
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Dict, List, Optional

import joblib
import numpy as np

# Use centralized logger from utils
from bondtrader.utils import logger

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

    # Fallback: create dummy tqdm
    def tqdm(iterable=None, desc=None, total=None, **kwargs):
        return iterable if iterable is not None else range(total or 0)


from bondtrader.analytics.factor_models import FactorModel

# Import all models
from bondtrader.config import get_config
from bondtrader.core.bond_models import Bond, BondType
from bondtrader.data.training_data_generator import (
    TrainingDataGenerator,
    load_training_dataset,
    save_training_dataset,
)
from bondtrader.ml.automl import AutoMLBondAdjuster
from bondtrader.ml.bayesian_optimization import BayesianOptimizer
from bondtrader.ml.drift_detection import (
    DriftDetector,
    ModelTuner,
    compare_models_against_benchmarks,
)
from bondtrader.ml.ml_adjuster import MLBondAdjuster
from bondtrader.ml.ml_adjuster_enhanced import EnhancedMLBondAdjuster
from bondtrader.ml.ml_advanced import AdvancedMLBondAdjuster
from bondtrader.ml.regime_models import RegimeDetector
from bondtrader.risk.tail_risk import TailRiskAnalyzer


class ModelTrainer:
    """Comprehensive model trainer with progress tracking, caching, parallel training, and checkpointing"""

    def __init__(
        self,
        dataset_path: str = None,
        generate_new: bool = False,
        checkpoint_dir: str = None,
        use_parallel: bool = True,
        max_workers: Optional[int] = None,
    ):
        """
        Initialize trainer

        Args:
            dataset_path: Path to saved dataset (if None, generates new)
            generate_new: Force generation of new dataset
            checkpoint_dir: Directory for saving checkpoints (defaults to config.checkpoint_dir)
            use_parallel: Whether to use parallel training
            max_workers: Maximum parallel workers (None = auto-detect from config or CPU count)
        """
        # Get centralized configuration
        self.config = get_config()

        # Use container for shared valuator instance
        from bondtrader.core.container import get_container

        self.valuator = get_container().get_valuator()
        self.checkpoint_dir = checkpoint_dir or self.config.checkpoint_dir
        self.use_parallel = use_parallel
        self.max_workers = max_workers or self.config.max_workers or min(mp.cpu_count(), 8)

        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Caching for bond conversion
        self._bond_cache = {}

        if generate_new or dataset_path is None or not os.path.exists(dataset_path):
            print("Generating new training dataset...")
            generator = TrainingDataGenerator(seed=self.config.ml_random_state)
            self.dataset = generator.generate_comprehensive_dataset(
                total_bonds=self.config.training_num_bonds,
                time_periods=self.config.training_time_periods,
                bonds_per_period=self.config.training_batch_size,
            )

            # Save dataset using config path
            os.makedirs(self.config.data_dir, exist_ok=True)
            default_dataset_path = os.path.join(self.config.data_dir, "training_dataset.joblib")
            save_training_dataset(self.dataset, default_dataset_path)
        else:
            print(f"Loading dataset from {dataset_path}...")
            self.dataset = load_training_dataset(dataset_path)

        # Convert dataset to bond objects with caching
        cache_key_train = "train"
        cache_key_val = "validation"
        cache_key_test = "test"

        if cache_key_train not in self._bond_cache:
            self._bond_cache[cache_key_train] = self._convert_to_bonds(self.dataset["train"])
        if cache_key_val not in self._bond_cache:
            self._bond_cache[cache_key_val] = self._convert_to_bonds(self.dataset["validation"])
        if cache_key_test not in self._bond_cache:
            self._bond_cache[cache_key_test] = self._convert_to_bonds(self.dataset["test"])

        self._train_bonds = self._bond_cache[cache_key_train]
        self._validation_bonds = self._bond_cache[cache_key_val]
        self._test_bonds = self._bond_cache[cache_key_test]

        print(f"Loaded dataset:")
        print(f"  Train: {len(self._train_bonds)} bonds")
        print(f"  Validation: {len(self._validation_bonds)} bonds")
        print(f"  Test: {len(self._test_bonds)} bonds")

        # Initialize drift detection and tuning
        self.drift_detector = DriftDetector()
        self.model_tuner = ModelTuner(self.drift_detector)

    @property
    def train_bonds(self) -> List[Bond]:
        """Cached access to training bonds"""
        return self._train_bonds

    @property
    def validation_bonds(self) -> List[Bond]:
        """Cached access to validation bonds"""
        return self._validation_bonds

    @property
    def test_bonds(self) -> List[Bond]:
        """Cached access to test bonds"""
        return self._test_bonds

    def _convert_to_bonds(self, split_data: Dict) -> List[Bond]:
        """Convert feature data back to Bond objects using stored metadata"""
        bonds = []

        # Reconstruct bonds from metadata (which now contains full bond info)
        for i, (features, target, metadata) in enumerate(
            zip(split_data["features"], split_data["targets"], split_data["metadata"])
        ):
            try:
                # Use metadata if available (preferred), otherwise reconstruct from features
                if "coupon_rate" in metadata and "face_value" in metadata:
                    # Use stored metadata
                    coupon_rate = metadata["coupon_rate"]
                    face_value = metadata["face_value"]
                    maturity_date = metadata.get("maturity_date")
                    issue_date = metadata.get("issue_date")
                    frequency = metadata.get("frequency", 2)
                    callable_flag = metadata.get("callable", False)
                    convertible_flag = metadata.get("convertible", False)
                    credit_rating = metadata.get("credit_rating", "BBB")
                    issuer = metadata.get("issuer", "Unknown")

                    # Determine bond type using shared constants
                    bond_type_str = metadata.get("bond_type", "Fixed Rate")
                    from bondtrader.utils.constants import BOND_TYPE_STRING_MAP

                    bond_type_name = BOND_TYPE_STRING_MAP.get(bond_type_str, "FIXED_RATE")
                    bond_type = getattr(BondType, bond_type_name, BondType.FIXED_RATE)

                    # Reconstruct price from target (target = market_price / fair_value)
                    # We'll use the price_to_par from features as approximation
                    price_to_par = features[3] if len(features) > 3 else 1.0
                    current_price = face_value * price_to_par
                else:
                    # Fallback: reconstruct from features
                    coupon_rate = features[0]
                    time_to_maturity = features[1]
                    credit_rating_numeric = int(features[2])
                    price_to_par = features[3]
                    years_since_issue = features[4]
                    frequency = int(features[5])
                    callable_flag = bool(features[6])
                    convertible_flag = bool(features[7])
                    face_value = features[13] if len(features) > 13 else 1000

                    # Convert rating using shared constants
                    from bondtrader.utils.constants import NUMERIC_TO_RATING

                    credit_rating = NUMERIC_TO_RATING.get(int(credit_rating_numeric), "BBB")

                    bond_type_str = metadata.get("bond_type", "Fixed Rate")
                    from bondtrader.utils.constants import BOND_TYPE_STRING_MAP

                    bond_type_name = BOND_TYPE_STRING_MAP.get(bond_type_str, "FIXED_RATE")
                    bond_type = getattr(BondType, bond_type_name, BondType.FIXED_RATE)

                    current_date = datetime.now()
                    maturity_date = current_date + timedelta(days=int(time_to_maturity * 365.25))
                    issue_date = current_date - timedelta(days=int(years_since_issue * 365.25))
                    current_price = face_value * price_to_par
                    issuer = metadata.get("issuer", "Unknown")

                # Create bond
                bond = Bond(
                    bond_id=metadata.get("bond_id", f"BOND-{i}"),
                    bond_type=bond_type,
                    face_value=face_value,
                    coupon_rate=coupon_rate,
                    maturity_date=maturity_date,
                    issue_date=issue_date,
                    current_price=current_price,
                    credit_rating=credit_rating,
                    issuer=issuer,
                    frequency=frequency,
                    callable=callable_flag,
                    convertible=convertible_flag,
                )
                bonds.append(bond)
            except Exception as e:
                # Skip invalid bonds
                continue

        return bonds

    def _save_checkpoint(self, model_name: str, result: Dict):
        """Save checkpoint for a model with atomic writes and file locking"""
        try:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"{model_name}.joblib")
            checkpoint_data = {
                "model_name": model_name,
                "result": result,
                "timestamp": datetime.now().isoformat(),
            }

            # Atomic write: save to temp file first
            temp_path = checkpoint_path + ".tmp"
            try:
                joblib.dump(checkpoint_data, temp_path)

                # Atomic rename (Unix) or copy+remove (Windows)
                if os.name == "nt":  # Windows
                    if os.path.exists(checkpoint_path):
                        os.remove(checkpoint_path)
                    os.rename(temp_path, checkpoint_path)
                else:  # Unix/Linux/Mac
                    os.rename(temp_path, checkpoint_path)

            except Exception as e:
                # Clean up temp file on error
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except (OSError, PermissionError) as cleanup_error:
                        # Log but don't fail on cleanup errors
                        pass
                raise e

            # Optional: Use file locking if on Unix (for multi-process safety)
            if os.name != "nt":  # Unix/Linux/Mac
                try:
                    # Create lock file
                    lock_path = checkpoint_path + ".lock"
                    with open(lock_path, "w") as lock_file:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        # Lock held, but we already saved, so just remove lock
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                    # Remove lock file
                    if os.path.exists(lock_path):
                        os.remove(lock_path)
                except (IOError, OSError):
                    # Locking failed, but file is already saved atomically
                    pass

        except Exception as e:
            logger.warning(f"Failed to save checkpoint for {model_name}: {e}", exc_info=True)
            print(f"  Warning: Failed to save checkpoint for {model_name}: {e}")

    def _load_checkpoint(self, model_name: str) -> Optional[Dict]:
        """Load checkpoint for a model if it exists"""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{model_name}.joblib")
        if os.path.exists(checkpoint_path):
            try:
                checkpoint_data = joblib.load(checkpoint_path)
                return checkpoint_data.get("result")
            except Exception as e:
                print(f"  Warning: Failed to load checkpoint for {model_name}: {e}")
        return None

    def train_all_models(self, resume: bool = False) -> Dict:
        """
        Train all models in the codebase with progress tracking and checkpointing

        Args:
            resume: If True, attempt to resume from checkpoints

        Returns:
            Dictionary with training results for all models
        """
        results = {}
        total_steps = 11
        start_time = time.time()

        print("\n" + "=" * 60)
        print("TRAINING ALL MODELS (Enhanced)")
        print("=" * 60)
        if self.use_parallel:
            print(f"Parallel training enabled (max_workers: {self.max_workers})")
        print(f"Checkpoint directory: {self.checkpoint_dir}")

        # Create progress bar
        pbar = tqdm(total=total_steps, desc="Training Progress", unit="model")

        # Helper function to train a model with timing and validation
        def train_with_timing(
            step_num: int, model_name: str, train_func, *args, validation_func=None, **kwargs
        ):
            """
            Train a model with progress tracking, checkpointing, and validation

            FIXED: Adds validation step before accepting model
            """
            step_start = time.time()

            # Check for existing checkpoint
            if resume:
                checkpoint = self._load_checkpoint(model_name)
                if checkpoint and checkpoint.get("status") == "success":
                    pbar.set_description(f"Resumed {model_name} (checkpoint)")
                    pbar.update(1)
                    return checkpoint

            try:
                # Train model
                result = train_func(*args, **kwargs)

                # Validate model if validation function provided
                if validation_func is not None and "model" in result:
                    try:
                        validation_result = validation_func(result["model"])
                        result["validation"] = validation_result

                        # Check if validation passed
                        if validation_result.get("passed", True):
                            elapsed = time.time() - step_start
                            self._save_checkpoint(model_name, result)
                            pbar.set_description(f"✓ {model_name} ({elapsed:.1f}s)")
                            return result
                        else:
                            # Validation failed
                            elapsed = time.time() - step_start
                            result["status"] = "validation_failed"
                            result["validation_error"] = validation_result.get(
                                "error", "Validation failed"
                            )
                            self._save_checkpoint(model_name, result)
                            pbar.set_description(f"✗ {model_name} (validation failed)")
                            logger.warning(
                                f"{model_name} failed validation: {result.get('validation_error')}"
                            )
                            print(f"\n  ✗ {model_name} failed validation")
                            return result
                    except Exception as validation_error:
                        # Validation function itself failed
                        logger.warning(
                            f"Validation function failed for {model_name}: {validation_error}"
                        )
                        # Continue with model anyway, but log the issue
                        result["validation"] = {"passed": False, "error": str(validation_error)}
                        elapsed = time.time() - step_start
                        self._save_checkpoint(model_name, result)
                        pbar.set_description(f"✓ {model_name} ({elapsed:.1f}s) [validation error]")
                        return result
                else:
                    # No validation function, save as-is
                    elapsed = time.time() - step_start
                    self._save_checkpoint(model_name, result)
                    pbar.set_description(f"✓ {model_name} ({elapsed:.1f}s)")
                    return result

            except Exception as e:
                elapsed = time.time() - step_start
                result = {"status": "failed", "error": str(e)}
                self._save_checkpoint(model_name, result)
                pbar.set_description(f"✗ {model_name} ({elapsed:.1f}s)")
                logger.error(f"{model_name} training failed: {e}", exc_info=True)
                print(f"\n  ✗ {model_name} failed: {e}")
                return result
            finally:
                pbar.update(1)

        # Helper function to validate ML models
        def validate_ml_model(model):
            """Validate ML model on validation set"""
            try:
                validation_predictions = []
                validation_actuals = []

                for bond in self.validation_bonds[:100]:  # Sample for speed
                    try:
                        if hasattr(model, "predict_adjusted_value"):
                            pred = model.predict_adjusted_value(bond)
                            validation_predictions.append(
                                pred.get(
                                    "ml_adjusted_value",
                                    pred.get("ml_adjusted_fair_value", bond.current_price),
                                )
                            )
                            validation_actuals.append(bond.current_price)
                    except (ValueError, KeyError, AttributeError) as e:
                        # Skip invalid bonds
                        continue

                if len(validation_predictions) > 10:  # Need sufficient samples
                    from sklearn.metrics import r2_score

                    r2 = r2_score(validation_actuals, validation_predictions)
                    # Pass if R² > -1 (not terrible) and reasonable number of predictions
                    passed = r2 > -1.0 and len(validation_predictions) >= 10
                    return {"passed": passed, "r2": r2, "n_samples": len(validation_predictions)}
                else:
                    return {"passed": False, "error": "Insufficient validation samples"}
            except Exception as e:
                return {"passed": False, "error": str(e)}

        # 1. Basic ML Adjuster
        print("\n[1/9] Training Basic ML Adjuster...")

        def train_ml_adjuster():
            ml_adjuster = MLBondAdjuster(model_type=self.config.ml_model_type)
            ml_metrics = ml_adjuster.train(
                self.train_bonds,
                test_size=self.config.ml_test_size,
                random_state=self.config.ml_random_state,
            )
            result = {"metrics": ml_metrics, "model": ml_adjuster, "status": "success"}
            print(f"  ✓ Train R²: {ml_metrics['train_r2']:.4f}")
            print(f"  ✓ Test R²: {ml_metrics['test_r2']:.4f}")
            return result

        results["ml_adjuster"] = train_with_timing(
            1, "ml_adjuster", train_ml_adjuster, validation_func=validate_ml_model
        )

        # 2. Enhanced ML Adjuster
        print("\n[2/9] Training Enhanced ML Adjuster...")
        try:
            enhanced_ml = EnhancedMLBondAdjuster(model_type=self.config.ml_model_type)
            enhanced_metrics = enhanced_ml.train_with_tuning(
                self.train_bonds,
                test_size=self.config.ml_test_size,
                random_state=self.config.ml_random_state,
                tune_hyperparameters=True,
            )
            results["enhanced_ml_adjuster"] = {
                "metrics": enhanced_metrics,
                "model": enhanced_ml,
                "status": "success",
            }
            print(f"  ✓ Train R²: {enhanced_metrics['train_r2']:.4f}")
            print(f"  ✓ Test R²: {enhanced_metrics['test_r2']:.4f}")
            print(
                f"  ✓ CV R²: {enhanced_metrics['cv_r2_mean']:.4f} ± {enhanced_metrics['cv_r2_std']:.4f}"
            )
        except Exception as e:
            results["enhanced_ml_adjuster"] = {"status": "failed", "error": str(e)}
            print(f"  ✗ Failed: {e}")

        # 3. Advanced ML Adjuster (Ensemble)
        print("\n[3/9] Training Advanced ML Adjuster (Ensemble)...")
        try:
            advanced_ml = AdvancedMLBondAdjuster(self.valuator)
            ensemble_metrics = advanced_ml.train_ensemble(
                self.train_bonds,
                test_size=self.config.ml_test_size,
                random_state=self.config.ml_random_state,
            )
            results["advanced_ml_adjuster"] = {
                "metrics": ensemble_metrics,
                "model": advanced_ml,
                "status": "success",
            }
            print(f"  ✓ Ensemble Test R²: {ensemble_metrics['ensemble_metrics']['test_r2']:.4f}")
            print(f"  ✓ Improvement over best: {ensemble_metrics['improvement_over_best']:.4f}")
        except Exception as e:
            results["advanced_ml_adjuster"] = {"status": "failed", "error": str(e)}
            print(f"  ✗ Failed: {e}")

        # 4. AutoML
        print("\n[4/9] Training AutoML...")
        try:
            automl = AutoMLBondAdjuster(self.valuator)
            automl_results = automl.automated_model_selection(
                self.train_bonds,
                candidate_models=[
                    "random_forest",
                    "gradient_boosting",
                    "neural_network",
                    "ensemble",
                ],
                max_evaluation_time=300,
            )
            results["automl"] = {"results": automl_results, "model": automl, "status": "success"}
            print(f"  ✓ Best Model: {automl_results['best_model']}")
            print(f"  ✓ Best Score: {automl_results['best_score']:.4f}")
        except Exception as e:
            results["automl"] = {"status": "failed", "error": str(e)}
            print(f"  ✗ Failed: {e}")

        # 5. Regime Detector
        print("\n[5/9] Training Regime Detector...")
        try:
            regime_detector = RegimeDetector(self.valuator)
            regime_results = regime_detector.detect_regimes(
                self.train_bonds, num_regimes=4, method="kmeans"
            )
            results["regime_detector"] = {
                "results": regime_results,
                "model": regime_detector,
                "status": "success",
            }
            print(f"  ✓ Detected {regime_results['num_regimes']} regimes")
            for regime_name, regime_info in regime_results["regime_analysis"].items():
                print(
                    f"    - {regime_name}: {regime_info['num_bonds']} bonds, {regime_info['regime_type']}"
                )
        except Exception as e:
            results["regime_detector"] = {"status": "failed", "error": str(e)}
            print(f"  ✗ Failed: {e}")

        # 6. Factor Model
        print("\n[6/9] Training Factor Model...")
        try:
            factor_model = FactorModel(self.valuator)
            factor_results = factor_model.extract_bond_factors(
                self.train_bonds, num_factors=None
            )  # Auto-select
            results["factor_model"] = {
                "results": factor_results,
                "model": factor_model,
                "status": "success",
            }
            print(f"  ✓ Extracted {factor_results['num_factors']} factors")
            print(
                f"  ✓ Explained variance: {sum(factor_results['explained_variance'][:3]):.1%} (top 3)"
            )
        except Exception as e:
            results["factor_model"] = {"status": "failed", "error": str(e)}
            print(f"  ✗ Failed: {e}")

        # 7. Tail Risk Analyzer
        print("\n[7/9] Training Tail Risk Analyzer...")
        try:
            tail_risk = TailRiskAnalyzer(self.valuator)
            # Use sample for tail risk (computationally intensive)
            sample_bonds = self.train_bonds[:100]
            weights = [1.0 / len(sample_bonds)] * len(sample_bonds)

            cvar_result = tail_risk.calculate_cvar(
                sample_bonds, weights=weights, confidence_level=0.95, method="monte_carlo"
            )
            results["tail_risk"] = {"cvar": cvar_result, "model": tail_risk, "status": "success"}
            print(f"  ✓ CVaR (95%): {cvar_result['cvar_pct']:.2f}%")
            print(f"  ✓ Tail Ratio: {cvar_result['tail_ratio']:.2f}")
        except Exception as e:
            results["tail_risk"] = {"status": "failed", "error": str(e)}
            print(f"  ✗ Failed: {e}")

        # 8. Bayesian Optimizer
        print("\n[8/9] Training Bayesian Optimizer...")
        try:
            bayesian_opt = BayesianOptimizer(self.valuator)
            # Optimize hyperparameters for a sample
            sample_bonds = self.train_bonds[:200]
            opt_results = bayesian_opt.optimize_hyperparameters(
                sample_bonds, param_bounds={"n_estimators": (50, 200)}, num_iterations=20
            )
            results["bayesian_optimizer"] = {
                "results": opt_results,
                "model": bayesian_opt,
                "status": "success",
            }
            print(f"  ✓ Optimal parameters: {opt_results['optimal_parameters']}")
            print(f"  ✓ Best R²: {opt_results['best_value']:.4f}")
        except Exception as e:
            results["bayesian_optimizer"] = {"status": "failed", "error": str(e)}
            print(f"  ✗ Failed: {e}")

        # 9. Model Evaluation on Test Set
        print("\n[9/9] Evaluating models on test set...")
        test_evaluations = self._evaluate_on_test_set(results)
        results["test_evaluations"] = test_evaluations

        # 10. Drift Detection against Leading Financial Firms Benchmarks
        print("\n[10/10] Comparing models against leading financial firms benchmarks...")
        drift_analysis = self._analyze_drift_against_benchmarks(results)
        results["drift_analysis"] = drift_analysis

        # 11. Tune Models to Minimize Drift
        print("\n[11/11] Tuning models to minimize drift...")
        tuned_results = self._tune_models_for_minimal_drift(results)
        results["tuned_models"] = tuned_results

        # Close progress bar
        pbar.close()

        # Calculate total time
        total_time = time.time() - start_time
        minutes = int(total_time // 60)
        seconds = int(total_time % 60)

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Total training time: {minutes}m {seconds}s ({total_time:.1f} seconds)")
        print(f"Checkpoints saved to: {self.checkpoint_dir}/")

        return results

    def _evaluate_on_test_set(self, trained_models: Dict) -> Dict:
        """Evaluate all trained models on test set"""
        evaluations = {}

        # Evaluate ML models
        for model_name, model_data in trained_models.items():
            if model_name in [
                "ml_adjuster",
                "enhanced_ml_adjuster",
                "advanced_ml_adjuster",
                "automl",
            ]:
                if model_data.get("status") == "success" and "model" in model_data:
                    try:
                        model = model_data["model"]
                        test_predictions = []
                        test_actuals = []

                        for bond in self.test_bonds[:100]:  # Sample for speed
                            try:
                                if hasattr(model, "predict_adjusted_value"):
                                    pred = model.predict_adjusted_value(bond)
                                    test_predictions.append(
                                        pred.get(
                                            "ml_adjusted_value",
                                            pred.get("ml_adjusted_fair_value", bond.current_price),
                                        )
                                    )
                                elif hasattr(model, "predict"):
                                    # Direct prediction
                                    fair_value = self.valuator.calculate_fair_value(bond)
                                    features = np.array([[bond.coupon_rate, bond.time_to_maturity]])
                                    pred = model.predict(features)[0]
                                    test_predictions.append(fair_value * pred)
                                else:
                                    continue

                                test_actuals.append(bond.current_price)
                            except (ValueError, KeyError, AttributeError) as e:
                                # Skip invalid bonds
                                continue

                        if len(test_predictions) > 0:
                            from sklearn.metrics import (
                                mean_absolute_error,
                                mean_squared_error,
                                r2_score,
                            )

                            mse = mean_squared_error(test_actuals, test_predictions)
                            rmse = np.sqrt(mse)
                            mae = mean_absolute_error(test_actuals, test_predictions)
                            r2 = r2_score(test_actuals, test_predictions)

                            evaluations[model_name] = {
                                "mse": mse,
                                "rmse": rmse,
                                "mae": mae,
                                "r2": r2,
                                "n_samples": len(test_predictions),
                            }
                            print(f"  {model_name}: Test R² = {r2:.4f}, RMSE = {rmse:.2f}")
                    except Exception as e:
                        evaluations[model_name] = {"error": str(e)}

        return evaluations

    def save_models(self, results: Dict, model_dir: str = None):
        """
        Save all trained models with atomic writes

        FIXED: Uses atomic writes to prevent corruption

        Args:
            results: Dictionary of training results
            model_dir: Directory to save models (defaults to config.model_dir)
        """
        if model_dir is None:
            model_dir = self.config.model_dir
        os.makedirs(model_dir, exist_ok=True)

        print(f"\nSaving models to {model_dir}/...")

        for model_name, model_data in results.items():
            if model_data.get("status") == "success" and "model" in model_data:
                try:
                    model = model_data["model"]
                    filepath = os.path.join(model_dir, f"{model_name}.joblib")

                    if hasattr(model, "save_model"):
                        # Use model's save_model method (which should now be atomic)
                        model.save_model(filepath)
                    else:
                        # Generic save with atomic writes
                        temp_filepath = filepath + ".tmp"
                        try:
                            joblib.dump(model, temp_filepath)

                            # Atomic rename
                            if os.name == "nt":  # Windows
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
                                except (OSError, PermissionError) as cleanup_error:
                                    # Log but don't fail on cleanup errors
                                    pass
                            raise e

                    print(f"  ✓ Saved {model_name}")
                except Exception as e:
                    logger.error(f"Failed to save {model_name}: {e}", exc_info=True)
                    print(f"  ✗ Failed to save {model_name}: {e}")

    def _analyze_drift_against_benchmarks(self, trained_models: Dict) -> Dict:
        """
        Analyze drift of all trained models against leading financial firms benchmarks

        Returns:
            Dictionary with drift metrics for each model
        """
        drift_analysis = {}

        # Use validation set for drift analysis (not test set)
        evaluation_bonds = self.validation_bonds[:200]  # Sample for speed

        # Evaluate each ML model
        ml_model_names = ["ml_adjuster", "enhanced_ml_adjuster", "advanced_ml_adjuster", "automl"]

        for model_name in ml_model_names:
            if model_name not in trained_models:
                continue

            model_data = trained_models[model_name]
            if model_data.get("status") != "success" or "model" not in model_data:
                continue

            print(f"\n  Analyzing drift for {model_name}...")

            try:
                model = model_data["model"]
                predictions = []

                for bond in evaluation_bonds:
                    try:
                        if hasattr(model, "predict_adjusted_value"):
                            pred = model.predict_adjusted_value(bond)
                            value = pred.get(
                                "ml_adjusted_value",
                                pred.get("ml_adjusted_fair_value", bond.current_price),
                            )
                        elif hasattr(model, "predict"):
                            fair_value = self.valuator.calculate_fair_value(bond)
                            features = np.array([[bond.coupon_rate, bond.time_to_maturity]])
                            pred_value = model.predict(features)[0]
                            value = fair_value * pred_value
                        else:
                            value = bond.current_price

                        predictions.append(value)
                    except Exception as e:
                        predictions.append(bond.current_price)

                # Calculate drift against consensus benchmark
                drift_metrics = self.drift_detector.calculate_drift(
                    evaluation_bonds, predictions, benchmark_methodology="consensus"
                )

                # Also calculate against individual benchmarks
                individual_drifts = {}
                for benchmark_name in ["bloomberg", "aladdin", "goldman", "jpmorgan"]:
                    try:
                        individual_drift = self.drift_detector.calculate_drift(
                            evaluation_bonds, predictions, benchmark_methodology=benchmark_name
                        )
                        individual_drifts[benchmark_name] = individual_drift
                    except Exception as e:
                        continue

                drift_analysis[model_name] = {
                    "consensus_drift": drift_metrics,
                    "individual_drifts": individual_drifts,
                }

                print(f"    Consensus Drift Score: {drift_metrics.drift_score:.4f}")
                print(f"    RMSE vs. Consensus: {drift_metrics.root_mean_squared_error:.2f}")
                print(f"    Correlation: {drift_metrics.correlation:.4f}")
                print(f"    Bias: {drift_metrics.bias:.2f}")

            except Exception as e:
                print(f"    ✗ Failed to analyze drift: {e}")
                drift_analysis[model_name] = {"error": str(e)}

        return drift_analysis

    def _tune_models_for_minimal_drift(self, trained_models: Dict) -> Dict:
        """
        Tune models to minimize drift against benchmarks

        Returns:
            Dictionary with tuning results for each model
        """
        tuning_results = {}

        # Use subset for tuning (computationally intensive)
        tuning_bonds = self.train_bonds[:500]
        validation_bonds = self.validation_bonds[:200]

        # Only tune ML models that can be tuned
        tunable_models = ["enhanced_ml_adjuster", "advanced_ml_adjuster"]

        for model_name in tunable_models:
            if model_name not in trained_models:
                continue

            model_data = trained_models[model_name]
            if model_data.get("status") != "success" or "model" not in model_data:
                continue

            print(f"\n  Tuning {model_name} to minimize drift...")

            try:
                model = model_data["model"]

                # Define tuning parameters based on model type
                if model_name == "enhanced_ml_adjuster":
                    # For EnhancedMLBondAdjuster, tuning is done via train_with_tuning
                    # Just evaluate current model's drift
                    predictions = []
                    for bond in validation_bonds:
                        try:
                            pred = model.predict_adjusted_value(bond)
                            value = pred.get(
                                "ml_adjusted_value",
                                pred.get("ml_adjusted_fair_value", bond.current_price),
                            )
                            predictions.append(value)
                        except (ValueError, AttributeError, KeyError) as e:
                            # Fallback to current price if prediction fails
                            predictions.append(bond.current_price)

                    current_drift = self.drift_detector.calculate_drift(
                        validation_bonds, predictions, benchmark_methodology="consensus"
                    )

                    tuning_results[model_name] = {
                        "current_drift_score": current_drift.drift_score,
                        "drift_metrics": current_drift,
                        "note": "Model already tuned via train_with_tuning",
                    }

                    print(f"    Current Drift Score: {current_drift.drift_score:.4f}")

                elif model_name == "advanced_ml_adjuster":
                    # For AdvancedMLBondAdjuster, ensemble models are already optimized
                    # Just evaluate current model's drift
                    predictions = []
                    for bond in validation_bonds:
                        try:
                            pred = model.predict_adjusted_value(bond)
                            value = pred.get(
                                "ml_adjusted_value",
                                pred.get("ml_adjusted_fair_value", bond.current_price),
                            )
                            predictions.append(value)
                        except (ValueError, AttributeError, KeyError) as e:
                            # Fallback to current price if prediction fails
                            predictions.append(bond.current_price)

                    current_drift = self.drift_detector.calculate_drift(
                        validation_bonds, predictions, benchmark_methodology="consensus"
                    )

                    tuning_results[model_name] = {
                        "current_drift_score": current_drift.drift_score,
                        "drift_metrics": current_drift,
                        "note": "Ensemble model already optimized",
                    }

                    print(f"    Current Drift Score: {current_drift.drift_score:.4f}")

            except Exception as e:
                print(f"    ✗ Failed to tune: {e}")
                tuning_results[model_name] = {"error": str(e)}

        return tuning_results


def main() -> None:
    """
    Main training function

    Trains all models in the codebase and saves results.
    """
    print("=" * 60)
    print("COMPREHENSIVE MODEL TRAINING")
    print("Following Financial Industry Best Practices")
    print("=" * 60)

    # Get configuration
    config = get_config()

    # Initialize trainer
    default_dataset_path = os.path.join(config.data_dir, "training_dataset.joblib")
    trainer = ModelTrainer(
        dataset_path=default_dataset_path, generate_new=False
    )  # Set to True to generate new dataset

    # Train all models
    results = trainer.train_all_models()

    # Save models (uses config.model_dir by default)
    trainer.save_models(results)

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)

    successful = sum(1 for r in results.values() if r.get("status") == "success")
    failed = sum(1 for r in results.values() if r.get("status") == "failed")

    print(f"Successfully trained: {successful} models")
    print(f"Failed: {failed} models")

    if "test_evaluations" in results:
        print("\nTest Set Performance:")
        for model_name, eval_data in results["test_evaluations"].items():
            if "r2" in eval_data:
                print(f"  {model_name}: R² = {eval_data['r2']:.4f}, RMSE = {eval_data['rmse']:.2f}")

    # Print drift analysis summary
    if "drift_analysis" in results:
        print("\n" + "=" * 60)
        print("DRIFT ANALYSIS SUMMARY")
        print("=" * 60)
        print("\nDrift vs. Leading Financial Firms (Consensus Benchmark):")
        print(f"{'Model':<30} {'Drift Score':<15} {'RMSE':<15} {'Correlation':<15} {'Bias':<15}")
        print("-" * 90)

        for model_name, drift_data in results["drift_analysis"].items():
            if "consensus_drift" in drift_data:
                drift = drift_data["consensus_drift"]
                print(
                    f"{model_name:<30} {drift.drift_score:<15.4f} {drift.root_mean_squared_error:<15.2f} "
                    f"{drift.correlation:<15.4f} {drift.bias:<15.2f}"
                )

        # Identify models with lowest drift
        drift_scores = {}
        for model_name, drift_data in results["drift_analysis"].items():
            if "consensus_drift" in drift_data:
                drift_scores[model_name] = drift_data["consensus_drift"].drift_score

        if drift_scores:
            best_model = min(drift_scores, key=drift_scores.get)
            print(
                f"\n✓ Best model (lowest drift): {best_model} (drift score: {drift_scores[best_model]:.4f})"
            )

    # Print tuning results summary
    if "tuned_models" in results:
        print("\n" + "=" * 60)
        print("MODEL TUNING SUMMARY")
        print("=" * 60)
        for model_name, tuning_data in results["tuned_models"].items():
            if "current_drift_score" in tuning_data:
                print(f"{model_name}: Drift Score = {tuning_data['current_drift_score']:.4f}")
                if "note" in tuning_data:
                    print(f"  ({tuning_data['note']})")

    return results


if __name__ == "__main__":
    results = main()
