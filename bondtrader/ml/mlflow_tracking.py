"""
MLflow Experiment Tracking and Model Registry Integration
Implements industry-standard experiment tracking and model registry

Best Practices:
- Automatic experiment tracking for all training runs
- Model registry with versioning and aliases
- Experiment comparison and visualization
- Artifact storage (models, metrics, plots)
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature
    from mlflow.tracking import MlflowClient

    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from bondtrader.config import get_config
from bondtrader.utils.utils import logger


class MLflowTracker:
    """
    MLflow experiment tracking and model registry integration

    Industry Best Practices:
    - Hierarchical experiment naming
    - Comprehensive metadata tracking
    - Model registry with aliases
    - Artifact storage
    """

    def __init__(self, experiment_name: str = None, tracking_uri: str = None):
        """
        Initialize MLflow tracker

        Args:
            experiment_name: Name of experiment (defaults to config or timestamp)
            tracking_uri: MLflow tracking URI (defaults to local file store)
        """
        if not HAS_MLFLOW:
            logger.warning("MLflow not available. Install with: pip install mlflow")
            self.enabled = False
            return

        self.enabled = True
        self.config = get_config()

        # Set tracking URI (default to local file store)
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        elif os.getenv("MLFLOW_TRACKING_URI"):
            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        else:
            # Default to local file store in project directory
            default_uri = os.path.join(self.config.model_dir, "mlruns")
            mlflow.set_tracking_uri(f"file://{os.path.abspath(default_uri)}")

        # Set experiment
        if experiment_name is None:
            experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "BondTrader_ML")

        try:
            experiment_id = mlflow.create_experiment(experiment_name, exist_ok=True)
            mlflow.set_experiment(experiment_name)
        except Exception as e:
            logger.warning(f"Failed to set MLflow experiment: {e}")
            mlflow.set_experiment(experiment_name)

        self.experiment_name = experiment_name
        self.client = MlflowClient()

    def start_run(self, run_name: str = None, tags: Dict[str, str] = None) -> Optional[str]:
        """
        Start a new MLflow run

        Args:
            run_name: Name for this run
            tags: Dictionary of tags to add to the run

        Returns:
            Run ID if successful, None otherwise
        """
        if not self.enabled:
            return None

        try:
            if run_name is None:
                run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            mlflow.start_run(run_name=run_name)

            # Add default tags
            default_tags = {
                "project": "BondTrader",
                "framework": "scikit-learn",
                "created_at": datetime.now().isoformat(),
            }
            if tags:
                default_tags.update(tags)

            mlflow.set_tags(default_tags)

            # Log git commit if available
            try:
                import subprocess

                git_commit = (
                    subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
                    .decode()
                    .strip()
                )
                mlflow.set_tag("git_commit", git_commit)
            except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
                # Git not available or not in a git repo - not critical
                pass

            return mlflow.active_run().info.run_id

        except Exception as e:
            logger.error(f"Failed to start MLflow run: {e}", exc_info=True)
            return None

    def end_run(self):
        """End the current MLflow run"""
        if not self.enabled:
            return

        try:
            mlflow.end_run()
        except Exception as e:
            logger.warning(f"Failed to end MLflow run: {e}")

    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters for the current run

        Args:
            params: Dictionary of parameters to log
        """
        if not self.enabled:
            return

        try:
            mlflow.log_params(params)
        except Exception as e:
            logger.warning(f"Failed to log parameters: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """
        Log metrics for the current run

        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number for time-series metrics
        """
        if not self.enabled:
            return

        try:
            if step is not None:
                for key, value in metrics.items():
                    mlflow.log_metric(key, value, step=step)
            else:
                mlflow.log_metrics(metrics)
        except Exception as e:
            logger.warning(f"Failed to log metrics: {e}")

    def log_model(
        self,
        model: BaseEstimator,
        artifact_path: str = "model",
        signature=None,
        input_example=None,
        registered_model_name: str = None,
    ):
        """
        Log model to MLflow

        Args:
            model: Trained model to log
            artifact_path: Path within run artifacts
            signature: Model signature (input/output schema)
            input_example: Example input for the model
            registered_model_name: Name for model registry (if None, just logs to run)
        """
        if not self.enabled:
            return

        try:
            # Infer signature if not provided
            if signature is None and input_example is not None:
                try:
                    signature = infer_signature(input_example)
                except (ValueError, TypeError, AttributeError):
                    # Signature inference failed - not critical, will use default
                    pass

            # Log model
            mlflow.sklearn.log_model(
                model,
                artifact_path=artifact_path,
                signature=signature,
                input_example=input_example,
                registered_model_name=registered_model_name,
            )

            logger.info(f"Logged model to MLflow: {artifact_path}")

        except Exception as e:
            logger.error(f"Failed to log model to MLflow: {e}", exc_info=True)

    def log_artifacts(self, local_dir: str, artifact_path: str = None):
        """
        Log artifacts (files) to MLflow

        Args:
            local_dir: Local directory containing artifacts
            artifact_path: Path within run artifacts
        """
        if not self.enabled:
            return

        try:
            mlflow.log_artifacts(local_dir, artifact_path=artifact_path)
        except Exception as e:
            logger.warning(f"Failed to log artifacts: {e}")

    def log_artifact(self, local_path: str, artifact_path: str = None):
        """
        Log a single artifact file to MLflow

        Args:
            local_path: Local path to artifact file
            artifact_path: Path within run artifacts
        """
        if not self.enabled:
            return

        try:
            mlflow.log_artifact(local_path, artifact_path=artifact_path)
        except Exception as e:
            logger.warning(f"Failed to log artifact: {e}")

    def register_model(
        self,
        model_name: str,
        model_version: str = None,
        stage: str = None,
        alias: str = None,
        metadata: Dict[str, Any] = None,
    ) -> Optional[str]:
        """
        Register model in MLflow Model Registry

        Args:
            model_name: Name for the registered model
            model_version: Version to register (defaults to latest)
            stage: Stage to assign (deprecated, use alias instead)
            alias: Alias to assign (e.g., "champion", "canary")
            metadata: Additional metadata to store

        Returns:
            Model version if successful, None otherwise
        """
        if not self.enabled:
            return None

        try:
            run_id = mlflow.active_run().info.run_id if mlflow.active_run() else None

            if run_id:
                # Register from current run
                model_uri = f"runs:/{run_id}/model"
            else:
                # Use provided model version
                if model_version is None:
                    raise ValueError("Must provide model_version if not in active run")
                model_uri = f"models:/{model_name}/{model_version}"

            # Create registered model if it doesn't exist
            try:
                self.client.create_registered_model(model_name)
            except Exception:
                pass  # Model already exists

            # Create model version
            mv = self.client.create_model_version(
                name=model_name,
                source=model_uri,
                run_id=run_id,
            )

            # Set alias if provided
            if alias:
                self.client.set_registered_model_alias(model_name, alias, mv.version)

            # Add metadata as tags
            if metadata:
                for key, value in metadata.items():
                    self.client.set_model_version_tag(model_name, mv.version, key, str(value))

            logger.info(f"Registered model: {model_name} version {mv.version}")
            return mv.version

        except Exception as e:
            logger.error(f"Failed to register model: {e}", exc_info=True)
            return None

    def load_model(self, model_name: str, version: str = None, alias: str = None):
        """
        Load model from MLflow Model Registry

        Args:
            model_name: Name of registered model
            version: Version to load (if None, uses alias or latest)
            alias: Alias to load (e.g., "champion")

        Returns:
            Loaded model
        """
        if not self.enabled:
            raise ValueError("MLflow not enabled")

        try:
            if alias:
                model_uri = f"models:/{model_name}@{alias}"
            elif version:
                model_uri = f"models:/{model_name}/{version}"
            else:
                # Get latest version
                latest = self.client.get_latest_versions(model_name, stages=[])[0]
                model_uri = f"models:/{model_name}/{latest.version}"

            model = mlflow.sklearn.load_model(model_uri)
            return model

        except Exception as e:
            logger.error(f"Failed to load model from registry: {e}", exc_info=True)
            raise

    def search_runs(
        self,
        filter_string: str = None,
        max_results: int = 100,
        order_by: List[str] = None,
    ) -> List[Dict]:
        """
        Search for runs in the current experiment

        Args:
            filter_string: MLflow filter string (e.g., "metrics.test_r2 > 0.8")
            max_results: Maximum number of results
            order_by: List of metrics/params to order by (e.g., ["metrics.test_r2 DESC"])

        Returns:
            List of run dictionaries
        """
        if not self.enabled:
            return []

        try:
            experiment = self.client.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                return []

            runs = self.client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=filter_string,
                max_results=max_results,
                order_by=order_by or [],
            )

            return [
                {
                    "run_id": run.info.run_id,
                    "run_name": run.info.run_name,
                    "status": run.info.status,
                    "start_time": run.info.start_time,
                    "end_time": run.info.end_time,
                    "metrics": run.data.metrics,
                    "params": run.data.params,
                    "tags": run.data.tags,
                }
                for run in runs
            ]

        except Exception as e:
            logger.error(f"Failed to search runs: {e}", exc_info=True)
            return []

    def compare_runs(self, run_ids: List[str]) -> pd.DataFrame:
        """
        Compare multiple runs side-by-side

        Args:
            run_ids: List of run IDs to compare

        Returns:
            DataFrame with comparison
        """
        if not self.enabled:
            return pd.DataFrame()

        try:
            runs_data = []
            for run_id in run_ids:
                run = self.client.get_run(run_id)
                run_dict = {
                    "run_id": run_id,
                    "run_name": run.info.run_name,
                    **run.data.metrics,
                    **{f"param_{k}": v for k, v in run.data.params.items()},
                }
                runs_data.append(run_dict)

            return pd.DataFrame(runs_data)

        except Exception as e:
            logger.error(f"Failed to compare runs: {e}", exc_info=True)
            return pd.DataFrame()


def track_training_run(
    model_name: str,
    model: BaseEstimator,
    metrics: Dict[str, float],
    params: Dict[str, Any],
    tags: Dict[str, str] = None,
    register_model: bool = False,
    registered_model_name: str = None,
    input_example=None,
) -> Optional[str]:
    """
    Convenience function to track a complete training run

    Args:
        model_name: Name for this training run
        model: Trained model
        metrics: Training metrics
        params: Training parameters
        tags: Additional tags
        register_model: Whether to register in model registry
        registered_model_name: Name for registered model
        input_example: Example input for model signature

    Returns:
        Run ID if successful
    """
    tracker = MLflowTracker()

    if not tracker.enabled:
        logger.warning("MLflow not enabled, skipping tracking")
        return None

    run_id = tracker.start_run(run_name=model_name, tags=tags)

    try:
        # Log parameters
        tracker.log_params(params)

        # Log metrics
        tracker.log_metrics(metrics)

        # Log model
        tracker.log_model(
            model=model,
            artifact_path="model",
            input_example=input_example,
            registered_model_name=registered_model_name if register_model else None,
        )

        # Register model if requested
        if register_model and registered_model_name:
            tracker.register_model(
                model_name=registered_model_name,
                alias="latest",
                metadata={"training_run": model_name, "metrics": json.dumps(metrics)},
            )

        return run_id

    finally:
        tracker.end_run()
