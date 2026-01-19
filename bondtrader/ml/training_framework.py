"""
Unified Training Framework
Consolidates common training patterns to eliminate duplication across training scripts
"""

import os
import time
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import joblib
import numpy as np

from bondtrader.core.bond_models import Bond
from bondtrader.data.evaluation_dataset_generator import (
    EvaluationDatasetGenerator,
    load_evaluation_dataset,
)
from bondtrader.data.training_data_generator import (
    TrainingDataGenerator,
    load_training_dataset,
    save_training_dataset,
)
from bondtrader.utils import logger
from bondtrader.core.container import get_container

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

    def tqdm(iterable=None, desc=None, total=None, **kwargs):
        return iterable if iterable is not None else range(total or 0)


class TrainingConfig:
    """Configuration for training runs"""

    def __init__(
        self,
        model_type: str = "random_forest",
        feature_level: str = "basic",
        test_size: float = 0.2,
        random_state: int = 42,
        tune_hyperparameters: bool = False,
        use_ensemble: bool = False,
        use_mlflow: bool = False,
        checkpoint_dir: Optional[str] = None,
        model_dir: Optional[str] = None,
    ):
        self.model_type = model_type
        self.feature_level = feature_level
        self.test_size = test_size
        self.random_state = random_state
        self.tune_hyperparameters = tune_hyperparameters
        self.use_ensemble = use_ensemble
        self.use_mlflow = use_mlflow
        self.checkpoint_dir = checkpoint_dir
        self.model_dir = model_dir


class UnifiedTrainingFramework:
    """
    Unified framework for training ML models
    Handles dataset management, training, checkpointing, and evaluation
    """

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        model_dir: Optional[str] = None,
        generate_new_dataset: bool = False,
    ):
        """
        Initialize training framework

        Args:
            dataset_path: Path to training dataset (generates new if None)
            checkpoint_dir: Directory for checkpoints
            model_dir: Directory for saved models
            generate_new_dataset: Force generation of new dataset
        """
        from bondtrader.config import get_config

        self.config = get_config()
        self.valuator = get_container().get_valuator()

        self.dataset_path = dataset_path
        self.checkpoint_dir = checkpoint_dir or self.config.checkpoint_dir
        self.model_dir = model_dir or self.config.model_dir

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        # Load or generate dataset
        if generate_new_dataset or dataset_path is None or not os.path.exists(dataset_path or ""):
            logger.info("Generating new training dataset...")
            generator = TrainingDataGenerator(seed=self.config.ml_random_state)
            self.dataset = generator.generate_comprehensive_dataset(
                total_bonds=self.config.training_num_bonds,
                time_periods=self.config.training_time_periods,
                bonds_per_period=self.config.training_batch_size,
            )
            if dataset_path:
                os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
                save_training_dataset(self.dataset, dataset_path)
        else:
            logger.info(f"Loading dataset from {dataset_path}...")
            self.dataset = load_training_dataset(dataset_path)

        # Convert to bonds
        self.train_bonds = self._convert_to_bonds(self.dataset["train"])
        self.validation_bonds = self._convert_to_bonds(self.dataset["validation"])
        self.test_bonds = self._convert_to_bonds(self.dataset["test"])

        logger.info(
            f"Loaded dataset: {len(self.train_bonds)} train, {len(self.validation_bonds)} validation, {len(self.test_bonds)} test"
        )

    def _convert_to_bonds(self, split_data: Dict) -> List[Bond]:
        """Convert dataset split to Bond objects"""
        from datetime import timedelta

        from bondtrader.core.bond_models import BondType
        from bondtrader.utils.constants import BOND_TYPE_STRING_MAP

        bonds = []
        for i, (features, target, metadata) in enumerate(
            zip(split_data["features"], split_data["targets"], split_data["metadata"])
        ):
            try:
                if "coupon_rate" in metadata:
                    coupon_rate = metadata["coupon_rate"]
                    face_value = metadata["face_value"]
                    maturity_date = metadata.get("maturity_date")
                    issue_date = metadata.get("issue_date")
                    frequency = metadata.get("frequency", 2)
                    callable_flag = metadata.get("callable", False)
                    convertible_flag = metadata.get("convertible", False)
                    credit_rating = metadata.get("credit_rating", "BBB")
                    issuer = metadata.get("issuer", "Unknown")

                    bond_type_str = metadata.get("bond_type", "Fixed Rate")
                    bond_type_name = BOND_TYPE_STRING_MAP.get(bond_type_str, "FIXED_RATE")
                    bond_type = getattr(BondType, bond_type_name, BondType.FIXED_RATE)

                    current_date = datetime.now()
                    if not maturity_date:
                        time_to_maturity = metadata.get("time_to_maturity", features[1] if len(features) > 1 else 5.0)
                        maturity_date = current_date + timedelta(days=int(time_to_maturity * 365.25))
                    if not issue_date:
                        years_since_issue = metadata.get("years_since_issue", features[4] if len(features) > 4 else 0.0)
                        issue_date = current_date - timedelta(days=int(years_since_issue * 365.25))

                    price_to_par = features[3] if len(features) > 3 else 1.0
                    current_price = face_value * price_to_par

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
                logger.debug(f"Skipping invalid bond {i}: {e}")
                continue

        return bonds

    def train_model(
        self,
        config: TrainingConfig,
        model_name: str,
        resume: bool = False,
        validation_func: Optional[Callable] = None,
    ) -> Dict:
        """
        Train a single model with the given configuration

        Args:
            config: Training configuration
            model_name: Name for the model (used for checkpointing)
            resume: If True, resume from checkpoint if available
            validation_func: Optional validation function

        Returns:
            Training results dictionary
        """
        from bondtrader.ml.ml_adjuster_unified import MLBondAdjuster

        # Check for checkpoint
        if resume:
            checkpoint = self._load_checkpoint(model_name)
            if checkpoint and checkpoint.get("status") == "success":
                logger.info(f"Resuming from checkpoint for {model_name}")
                return checkpoint

        try:
            # Create model
            adjuster = MLBondAdjuster(
                model_type=config.model_type,
                feature_level=config.feature_level,
                valuator=self.valuator,
                use_ensemble=config.use_ensemble,
            )

            # Train
            start_time = time.time()
            if config.use_ensemble:
                metrics = adjuster.train_ensemble(
                    self.train_bonds, test_size=config.test_size, random_state=config.random_state
                )
            else:
                metrics = adjuster.train(
                    self.train_bonds,
                    test_size=config.test_size,
                    random_state=config.random_state,
                    tune_hyperparameters=config.tune_hyperparameters,
                    use_mlflow=config.use_mlflow,
                )

            elapsed = time.time() - start_time

            # Validate if function provided
            validation_result = None
            if validation_func:
                try:
                    validation_result = validation_func(adjuster)
                except Exception as e:
                    logger.warning(f"Validation failed for {model_name}: {e}")
                    validation_result = {"passed": False, "error": str(e)}

            # Save model
            model_path = os.path.join(self.model_dir, f"{model_name}.joblib")
            adjuster.save_model(model_path)

            result = {
                "metrics": metrics,
                "model": adjuster,
                "model_path": model_path,
                "status": "success",
                "elapsed_time": elapsed,
                "config": config.__dict__,
            }

            if validation_result:
                result["validation"] = validation_result

            # Save checkpoint
            self._save_checkpoint(model_name, result)

            return result

        except Exception as e:
            result = {"status": "failed", "error": str(e), "model_name": model_name}
            self._save_checkpoint(model_name, result)
            logger.error(f"Training failed for {model_name}: {e}", exc_info=True)
            raise

    def train_multiple_models(
        self,
        configs: List[Tuple[str, TrainingConfig]],
        resume: bool = False,
        show_progress: bool = True,
    ) -> Dict:
        """
        Train multiple models with different configurations

        Args:
            configs: List of (model_name, TrainingConfig) tuples
            resume: If True, resume from checkpoints
            show_progress: If True, show progress bar

        Returns:
            Dictionary of training results
        """
        results = {}
        pbar = tqdm(total=len(configs), desc="Training Models", unit="model") if show_progress else None

        for model_name, config in configs:
            try:
                result = self.train_model(config, model_name, resume=resume)
                results[model_name] = result
                if pbar:
                    pbar.set_description(f"✓ {model_name}")
            except Exception as e:
                results[model_name] = {"status": "failed", "error": str(e)}
                if pbar:
                    pbar.set_description(f"✗ {model_name}")
                logger.error(f"Failed to train {model_name}: {e}")
            finally:
                if pbar:
                    pbar.update(1)

        if pbar:
            pbar.close()

        return results

    def evaluate_model(self, model, bonds: List[Bond], sample_size: Optional[int] = None) -> Dict:
        """
        Evaluate a trained model on a set of bonds

        Args:
            model: Trained model with predict_adjusted_value method
            bonds: Bonds to evaluate on
            sample_size: Optional sample size (for speed)

        Returns:
            Evaluation metrics
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        if sample_size:
            bonds = bonds[:sample_size]

        predictions = []
        actuals = []

        for bond in bonds:
            try:
                if hasattr(model, "predict_adjusted_value"):
                    pred = model.predict_adjusted_value(bond)
                    predictions.append(
                        pred.get(
                            "ml_adjusted_value",
                            pred.get("ml_adjusted_fair_value", bond.current_price),
                        )
                    )
                    actuals.append(bond.current_price)
            except Exception as e:
                logger.debug(f"Prediction failed for bond: {e}")
                continue

        if len(predictions) < 10:
            return {"error": "Insufficient predictions", "n_samples": len(predictions)}

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        return {
            "r2": r2_score(actuals, predictions),
            "rmse": np.sqrt(mean_squared_error(actuals, predictions)),
            "mae": mean_absolute_error(actuals, predictions),
            "n_samples": len(predictions),
        }

    def _save_checkpoint(self, model_name: str, result: Dict):
        """Save checkpoint with atomic writes"""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{model_name}.joblib")
        checkpoint_data = {
            "model_name": model_name,
            "result": result,
            "timestamp": datetime.now().isoformat(),
        }

        temp_path = checkpoint_path + ".tmp"
        try:
            joblib.dump(checkpoint_data, temp_path)

            if os.name == "nt":  # Windows
                if os.path.exists(checkpoint_path):
                    os.remove(checkpoint_path)
                os.rename(temp_path, checkpoint_path)
            else:  # Unix/Linux/Mac
                os.rename(temp_path, checkpoint_path)
        except Exception as e:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            logger.warning(f"Failed to save checkpoint for {model_name}: {e}")

    def _load_checkpoint(self, model_name: str) -> Optional[Dict]:
        """Load checkpoint if it exists"""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{model_name}.joblib")
        if os.path.exists(checkpoint_path):
            try:
                checkpoint_data = joblib.load(checkpoint_path)
                return checkpoint_data.get("result")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint for {model_name}: {e}")
        return None
