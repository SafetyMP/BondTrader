"""
A/B Testing Framework for Model Deployment
Implements industry-standard A/B testing for ML models

Industry Best Practices:
- Traffic splitting between models
- Metrics collection per variant
- Statistical significance testing
- Gradual rollout capability
"""

import json
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from bondtrader.config import get_config
from bondtrader.core.bond_models import Bond
from bondtrader.utils.utils import logger


class Variant(Enum):
    """A/B test variants"""

    CONTROL = "control"  # Current production model
    TREATMENT = "treatment"  # New model being tested


@dataclass
class ABTestConfig:
    """Configuration for A/B test"""

    test_name: str
    control_model_name: str
    treatment_model_name: str
    traffic_split: float = 0.5  # 0.5 = 50% to treatment
    min_samples: int = 1000  # Minimum samples per variant
    significance_level: float = 0.05  # p-value threshold
    duration_days: int = 7  # Test duration
    metrics_to_track: List[str] = None  # Metrics to compare


@dataclass
class ABTestResult:
    """Result of A/B test"""

    test_name: str
    start_time: datetime
    end_time: datetime
    control_metrics: Dict[str, float]
    treatment_metrics: Dict[str, float]
    improvement_pct: Dict[str, float]
    statistical_significance: Dict[str, bool]
    p_values: Dict[str, float]
    winner: Optional[Variant]
    recommendation: str
    n_control_samples: int
    n_treatment_samples: int


@dataclass
class PredictionRecord:
    """Record of a prediction in A/B test"""

    timestamp: datetime
    bond_id: str
    variant: Variant
    predicted_value: float
    actual_value: Optional[float] = None
    error: Optional[float] = None
    latency_ms: float = 0.0


class ABTestFramework:
    """
    A/B Testing Framework for ML models

    Industry Best Practices:
    - Traffic splitting
    - Statistical significance testing
    - Gradual rollout
    - Metrics collection
    """

    def __init__(
        self,
        config: ABTestConfig,
        control_model,
        treatment_model,
        results_dir: str = None,
    ):
        """
        Initialize A/B test framework

        Args:
            config: A/B test configuration
            control_model: Control model (current production)
            treatment_model: Treatment model (new model to test)
            results_dir: Directory to save test results
        """
        self.config = config
        self.control_model = control_model
        self.treatment_model = treatment_model
        self.config_obj = get_config()

        self.results_dir = results_dir or os.path.join(
            self.config_obj.model_dir, "ab_tests", config.test_name
        )
        os.makedirs(self.results_dir, exist_ok=True)

        # Prediction records
        self.control_records: List[PredictionRecord] = []
        self.treatment_records: List[PredictionRecord] = []

        # Test state
        self.start_time = None
        self.end_time = None
        self.is_running = False

        # Results
        self.test_result: Optional[ABTestResult] = None

    def assign_variant(self, bond_id: str) -> Variant:
        """
        Assign variant for a bond (traffic splitting)

        Uses consistent hashing to ensure same bond always gets same variant

        Args:
            bond_id: Bond identifier

        Returns:
            Assigned variant
        """
        # Use hash of bond_id for consistent assignment
        hash_value = hash(bond_id) % 100
        threshold = int(self.config.traffic_split * 100)

        if hash_value < threshold:
            return Variant.TREATMENT
        else:
            return Variant.CONTROL

    def predict(self, bond: Bond) -> Tuple[float, Variant]:
        """
        Get prediction from appropriate model variant

        Args:
            bond: Bond to predict

        Returns:
            Tuple of (prediction, variant)
        """
        variant = self.assign_variant(bond.bond_id)

        start_time = datetime.now()

        try:
            if variant == Variant.CONTROL:
                if hasattr(self.control_model, "predict_adjusted_value"):
                    result = self.control_model.predict_adjusted_value(bond)
                    prediction = result.get("ml_adjusted_value", bond.current_price)
                else:
                    prediction = self.control_model.predict([bond])[0]
            else:  # TREATMENT
                if hasattr(self.treatment_model, "predict_adjusted_value"):
                    result = self.treatment_model.predict_adjusted_value(bond)
                    prediction = result.get("ml_adjusted_value", bond.current_price)
                else:
                    prediction = self.treatment_model.predict([bond])[0]

            latency_ms = (datetime.now() - start_time).total_seconds() * 1000

            # Record prediction
            record = PredictionRecord(
                timestamp=datetime.now(),
                bond_id=bond.bond_id,
                variant=variant,
                predicted_value=prediction,
                actual_value=bond.current_price,
                error=abs(prediction - bond.current_price) if bond.current_price else None,
                latency_ms=latency_ms,
            )

            if variant == Variant.CONTROL:
                self.control_records.append(record)
            else:
                self.treatment_records.append(record)

            return prediction, variant

        except Exception as e:
            logger.error(f"Prediction failed for variant {variant.value}: {e}", exc_info=True)
            # Fallback to control
            if hasattr(self.control_model, "predict_adjusted_value"):
                result = self.control_model.predict_adjusted_value(bond)
                return result.get("ml_adjusted_value", bond.current_price), Variant.CONTROL
            else:
                return self.control_model.predict([bond])[0], Variant.CONTROL

    def start_test(self):
        """Start A/B test"""
        self.start_time = datetime.now()
        self.end_time = self.start_time + timedelta(days=self.config.duration_days)
        self.is_running = True
        self.control_records = []
        self.treatment_records = []

        logger.info(f"Started A/B test: {self.config.test_name}")
        logger.info(
            f"Traffic split: {self.config.traffic_split*100:.1f}% treatment, {(1-self.config.traffic_split)*100:.1f}% control"
        )
        logger.info(f"Duration: {self.config.duration_days} days")

    def end_test(self) -> ABTestResult:
        """
        End A/B test and compute results

        Returns:
            ABTestResult with statistical analysis
        """
        if not self.is_running:
            raise ValueError("Test is not running")

        self.is_running = False
        end_time = datetime.now()

        # Compute metrics for each variant
        control_metrics = self._compute_metrics(self.control_records)
        treatment_metrics = self._compute_metrics(self.treatment_records)

        # Statistical significance testing
        significance_results = {}
        p_values = {}
        improvement_pct = {}

        for metric in self.config.metrics_to_track or ["rmse", "mae", "error_rate"]:
            if metric in control_metrics and metric in treatment_metrics:
                control_values = self._get_metric_values(self.control_records, metric)
                treatment_values = self._get_metric_values(self.treatment_records, metric)

                if len(control_values) > 0 and len(treatment_values) > 0:
                    # Perform t-test
                    t_stat, p_value = stats.ttest_ind(control_values, treatment_values)
                    p_values[metric] = float(p_value)
                    significance_results[metric] = p_value < self.config.significance_level

                    # Compute improvement percentage
                    control_mean = np.mean(control_values)
                    if control_mean != 0:
                        improvement_pct[metric] = (
                            (control_mean - np.mean(treatment_values)) / control_mean * 100
                        )
                    else:
                        improvement_pct[metric] = 0.0

        # Determine winner
        winner = None
        recommendation = "No clear winner"

        # Check if treatment is significantly better
        primary_metric = self.config.metrics_to_track[0] if self.config.metrics_to_track else "rmse"
        if primary_metric in significance_results and significance_results[primary_metric]:
            if primary_metric in improvement_pct and improvement_pct[primary_metric] > 0:
                winner = Variant.TREATMENT
                recommendation = f"Deploy treatment model: {improvement_pct[primary_metric]:.2f}% improvement in {primary_metric}"
            elif primary_metric in improvement_pct and improvement_pct[primary_metric] < 0:
                winner = Variant.CONTROL
                recommendation = f"Keep control model: treatment performed worse"

        result = ABTestResult(
            test_name=self.config.test_name,
            start_time=self.start_time,
            end_time=end_time,
            control_metrics=control_metrics,
            treatment_metrics=treatment_metrics,
            improvement_pct=improvement_pct,
            statistical_significance=significance_results,
            p_values=p_values,
            winner=winner,
            recommendation=recommendation,
            n_control_samples=len(self.control_records),
            n_treatment_samples=len(self.treatment_records),
        )

        self.test_result = result
        self._save_results(result)

        logger.info(f"A/B test completed: {self.config.test_name}")
        logger.info(f"Winner: {winner.value if winner else 'None'}")
        logger.info(f"Recommendation: {recommendation}")

        return result

    def _compute_metrics(self, records: List[PredictionRecord]) -> Dict[str, float]:
        """Compute metrics from prediction records"""
        if len(records) == 0:
            return {}

        errors = [r.error for r in records if r.error is not None]

        if len(errors) == 0:
            return {
                "n_samples": len(records),
                "mean_latency_ms": np.mean([r.latency_ms for r in records]),
            }

        return {
            "n_samples": len(records),
            "rmse": float(np.sqrt(np.mean([e**2 for e in errors]))),
            "mae": float(np.mean(errors)),
            "max_error": float(np.max(errors)),
            "error_rate": sum(1 for e in errors if e > 50.0) / len(errors),  # > $50 error
            "mean_latency_ms": float(np.mean([r.latency_ms for r in records])),
        }

    def _get_metric_values(self, records: List[PredictionRecord], metric: str) -> List[float]:
        """Get metric values for statistical testing"""
        if metric == "rmse":
            # Use squared errors
            return [r.error**2 for r in records if r.error is not None]
        elif metric == "mae":
            return [r.error for r in records if r.error is not None]
        elif metric == "error_rate":
            return [
                1.0 if r.error and r.error > 50.0 else 0.0 for r in records if r.error is not None
            ]
        else:
            return []

    def _save_results(self, result: ABTestResult):
        """Save test results to disk"""
        result_file = os.path.join(
            self.results_dir, f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        result_dict = asdict(result)
        # Convert datetime to string
        result_dict["start_time"] = result.start_time.isoformat()
        result_dict["end_time"] = result.end_time.isoformat()
        if result.winner:
            result_dict["winner"] = result.winner.value

        with open(result_file, "w") as f:
            json.dump(result_dict, f, indent=2, default=str)

        logger.info(f"Saved A/B test results to {result_file}")

    def get_current_stats(self) -> Dict[str, Any]:
        """Get current test statistics"""
        if not self.is_running:
            return {}

        control_metrics = self._compute_metrics(self.control_records)
        treatment_metrics = self._compute_metrics(self.treatment_records)

        return {
            "test_name": self.config.test_name,
            "is_running": self.is_running,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "n_control_samples": len(self.control_records),
            "n_treatment_samples": len(self.treatment_records),
            "control_metrics": control_metrics,
            "treatment_metrics": treatment_metrics,
        }

    def is_test_complete(self) -> bool:
        """Check if test has enough samples and duration"""
        if not self.is_running:
            return False

        has_enough_samples = (
            len(self.control_records) >= self.config.min_samples
            and len(self.treatment_records) >= self.config.min_samples
        )

        has_enough_duration = datetime.now() >= self.end_time

        return has_enough_samples and has_enough_duration


def create_ab_test(
    test_name: str,
    control_model,
    treatment_model,
    traffic_split: float = 0.5,
    duration_days: int = 7,
    metrics_to_track: List[str] = None,
) -> ABTestFramework:
    """
    Convenience function to create A/B test

    Args:
        test_name: Name of the test
        control_model: Control model
        treatment_model: Treatment model
        traffic_split: Percentage of traffic to treatment (0.0-1.0)
        duration_days: Test duration in days
        metrics_to_track: Metrics to compare

    Returns:
        Configured ABTestFramework
    """
    config = ABTestConfig(
        test_name=test_name,
        control_model_name="control",
        treatment_model_name="treatment",
        traffic_split=traffic_split,
        duration_days=duration_days,
        metrics_to_track=metrics_to_track or ["rmse", "mae"],
    )

    return ABTestFramework(config, control_model, treatment_model)
