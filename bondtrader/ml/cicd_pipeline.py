"""
CI/CD Pipeline for ML Models
Automated testing and deployment for ML models

Industry Best Practices:
- Automated model validation
- Integration testing
- Deployment gates
- Rollback capability
"""

import json
import os
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from bondtrader.config import get_config
from bondtrader.ml.data_validation import DataValidator, ValidationResult
from bondtrader.ml.production_monitoring import ModelMonitor
from bondtrader.utils.utils import logger


class TestStatus(Enum):
    """Test status"""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class TestResult:
    """Result of a test"""

    test_name: str
    status: TestStatus
    message: str = ""
    duration_seconds: float = 0.0
    details: Dict[str, Any] = None


@dataclass
class ValidationGate:
    """Validation gate configuration"""

    gate_name: str
    test_function: Callable
    required: bool = True
    threshold: Optional[float] = None


@dataclass
class CICDResult:
    """Result of CI/CD pipeline"""

    pipeline_name: str
    status: str  # passed, failed, warning
    start_time: datetime
    end_time: datetime
    test_results: List[TestResult]
    model_version: Optional[str] = None
    deployed: bool = False
    error: Optional[str] = None


class MLModelCICD:
    """
    CI/CD Pipeline for ML models

    Industry Best Practices:
    - Automated testing
    - Validation gates
    - Deployment automation
    - Rollback capability
    """

    def __init__(
        self,
        pipeline_name: str = "ml_model_cicd",
        validation_gates: List[ValidationGate] = None,
        results_dir: str = None,
    ):
        """
        Initialize CI/CD pipeline

        Args:
            pipeline_name: Name of the pipeline
            validation_gates: List of validation gates
            results_dir: Directory to store results
        """
        self.pipeline_name = pipeline_name
        self.config = get_config()

        self.results_dir = results_dir or os.path.join(self.config.model_dir, "cicd_results")
        os.makedirs(self.results_dir, exist_ok=True)

        # Validation gates
        self.validation_gates = validation_gates or self._default_validation_gates()

        # Test results
        self.test_results: List[TestResult] = []

    def _default_validation_gates(self) -> List[ValidationGate]:
        """Create default validation gates"""
        return [
            ValidationGate(
                gate_name="data_validation",
                test_function=self._test_data_validation,
                required=True,
            ),
            ValidationGate(
                gate_name="model_performance",
                test_function=self._test_model_performance,
                required=True,
                threshold=0.7,  # Minimum R²
            ),
            ValidationGate(
                gate_name="model_size",
                test_function=self._test_model_size,
                required=False,
                threshold=100 * 1024 * 1024,  # 100MB max
            ),
            ValidationGate(
                gate_name="code_quality",
                test_function=self._test_code_quality,
                required=False,
            ),
        ]

    def run_pipeline(
        self,
        model,
        test_data: Dict,
        model_version: str = None,
    ) -> CICDResult:
        """
        Run complete CI/CD pipeline

        Args:
            model: Model to test
            test_data: Test data dictionary
            model_version: Model version

        Returns:
            CICDResult with all test results
        """
        start_time = datetime.now()
        self.test_results = []

        logger.info(f"Starting CI/CD pipeline: {self.pipeline_name}")

        try:
            # Run all validation gates
            for gate in self.validation_gates:
                logger.info(f"Running validation gate: {gate.gate_name}")

                test_start = datetime.now()
                try:
                    result = gate.test_function(model, test_data, gate)
                    test_duration = (datetime.now() - test_start).total_seconds()

                    result.duration_seconds = test_duration
                    self.test_results.append(result)

                    if result.status == TestStatus.FAILED and gate.required:
                        logger.error(f"Required gate {gate.gate_name} failed: {result.message}")
                        return self._create_failed_result(start_time, model_version)
                    elif result.status == TestStatus.FAILED:
                        logger.warning(f"Optional gate {gate.gate_name} failed: {result.message}")

                except Exception as e:
                    logger.error(f"Gate {gate.gate_name} raised exception: {e}", exc_info=True)
                    self.test_results.append(
                        TestResult(
                            test_name=gate.gate_name,
                            status=TestStatus.ERROR,
                            message=str(e),
                        )
                    )

                    if gate.required:
                        return self._create_failed_result(start_time, model_version, str(e))

            # All gates passed
            end_time = datetime.now()
            result = CICDResult(
                pipeline_name=self.pipeline_name,
                status="passed",
                start_time=start_time,
                end_time=end_time,
                test_results=self.test_results,
                model_version=model_version,
                deployed=False,
            )

            self._save_results(result)

            logger.info(f"CI/CD pipeline passed: {self.pipeline_name}")

            return result

        except Exception as e:
            logger.error(f"CI/CD pipeline failed: {e}", exc_info=True)
            return self._create_failed_result(start_time, model_version, str(e))

    def _test_data_validation(
        self,
        model,
        test_data: Dict,
        gate: ValidationGate,
    ) -> TestResult:
        """Test data validation"""
        try:
            from bondtrader.ml.data_validation import DataValidator, create_default_schema

            X = test_data.get("X")
            y = test_data.get("y")
            feature_names = test_data.get("feature_names", [])

            if X is None:
                return TestResult(
                    test_name=gate.gate_name,
                    status=TestStatus.FAILED,
                    message="Test data missing X",
                )

            schema = create_default_schema(feature_names) if feature_names else None
            validator = DataValidator(schema=schema)

            validation_result = validator.validate_complete(X, y, feature_names)

            if validation_result.passed:
                return TestResult(
                    test_name=gate.gate_name,
                    status=TestStatus.PASSED,
                    message=f"Data validation passed (quality score: {validation_result.quality_score:.2%})",
                    details={"quality_score": validation_result.quality_score},
                )
            else:
                return TestResult(
                    test_name=gate.gate_name,
                    status=TestStatus.FAILED,
                    message=f"Data validation failed: {', '.join(validation_result.errors)}",
                    details={"errors": validation_result.errors, "warnings": validation_result.warnings},
                )

        except Exception as e:
            return TestResult(
                test_name=gate.gate_name,
                status=TestStatus.ERROR,
                message=f"Data validation test error: {str(e)}",
            )

    def _test_model_performance(
        self,
        model,
        test_data: Dict,
        gate: ValidationGate,
    ) -> TestResult:
        """Test model performance"""
        try:
            X_test = test_data.get("X_test")
            y_test = test_data.get("y_test")

            if X_test is None or y_test is None:
                return TestResult(
                    test_name=gate.gate_name,
                    status=TestStatus.SKIPPED,
                    message="Test data not provided",
                )

            # Get predictions
            if hasattr(model, "predict"):
                predictions = model.predict(X_test)
            else:
                return TestResult(
                    test_name=gate.gate_name,
                    status=TestStatus.FAILED,
                    message="Model does not have predict method",
                )

            # Compute metrics
            import numpy as np
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

            r2 = r2_score(y_test, predictions)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            mae = mean_absolute_error(y_test, predictions)

            threshold = gate.threshold or 0.7

            if r2 >= threshold:
                return TestResult(
                    test_name=gate.gate_name,
                    status=TestStatus.PASSED,
                    message=f"Model performance acceptable (R²={r2:.4f})",
                    details={"r2": r2, "rmse": rmse, "mae": mae},
                )
            else:
                return TestResult(
                    test_name=gate.gate_name,
                    status=TestStatus.FAILED,
                    message=f"Model performance below threshold (R²={r2:.4f} < {threshold})",
                    details={"r2": r2, "rmse": rmse, "mae": mae},
                )

        except Exception as e:
            return TestResult(
                test_name=gate.gate_name,
                status=TestStatus.ERROR,
                message=f"Performance test error: {str(e)}",
            )

    def _test_model_size(
        self,
        model,
        test_data: Dict,
        gate: ValidationGate,
    ) -> TestResult:
        """Test model size"""
        try:
            import tempfile

            import joblib

            # Save model to temp file to check size
            with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as tmp_file:
                joblib.dump(model, tmp_file.name)
                model_size = os.path.getsize(tmp_file.name)
                os.unlink(tmp_file.name)

            threshold = gate.threshold or (100 * 1024 * 1024)  # 100MB default

            if model_size <= threshold:
                return TestResult(
                    test_name=gate.gate_name,
                    status=TestStatus.PASSED,
                    message=f"Model size acceptable ({model_size / 1024 / 1024:.2f}MB)",
                    details={"size_bytes": model_size, "size_mb": model_size / 1024 / 1024},
                )
            else:
                return TestResult(
                    test_name=gate.gate_name,
                    status=TestStatus.FAILED,
                    message=f"Model size too large ({model_size / 1024 / 1024:.2f}MB > {threshold / 1024 / 1024:.2f}MB)",
                    details={"size_bytes": model_size, "size_mb": model_size / 1024 / 1024},
                )

        except Exception as e:
            return TestResult(
                test_name=gate.gate_name,
                status=TestStatus.ERROR,
                message=f"Model size test error: {str(e)}",
            )

    def _test_code_quality(
        self,
        model,
        test_data: Dict,
        gate: ValidationGate,
    ) -> TestResult:
        """Test code quality (pylint, mypy, etc.)"""
        try:
            # Run pylint on ML modules
            result = subprocess.run(
                ["pylint", "bondtrader/ml/", "--errors-only"],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0:
                return TestResult(
                    test_name=gate.gate_name,
                    status=TestStatus.PASSED,
                    message="Code quality checks passed",
                )
            else:
                return TestResult(
                    test_name=gate.gate_name,
                    status=TestStatus.FAILED,
                    message=f"Code quality issues found: {result.stdout[:200]}",
                    details={"pylint_output": result.stdout},
                )

        except FileNotFoundError:
            return TestResult(
                test_name=gate.gate_name,
                status=TestStatus.SKIPPED,
                message="pylint not available",
            )
        except Exception as e:
            return TestResult(
                test_name=gate.gate_name,
                status=TestStatus.ERROR,
                message=f"Code quality test error: {str(e)}",
            )

    def _create_failed_result(
        self,
        start_time: datetime,
        model_version: Optional[str],
        error: str = None,
    ) -> CICDResult:
        """Create failed result"""
        end_time = datetime.now()
        result = CICDResult(
            pipeline_name=self.pipeline_name,
            status="failed",
            start_time=start_time,
            end_time=end_time,
            test_results=self.test_results,
            model_version=model_version,
            deployed=False,
            error=error,
        )

        self._save_results(result)
        return result

    def _save_results(self, result: CICDResult):
        """Save CI/CD results to file"""
        result_file = os.path.join(
            self.results_dir, f"cicd_{self.pipeline_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        result_dict = asdict(result)
        result_dict["start_time"] = result.start_time.isoformat()
        result_dict["end_time"] = result.end_time.isoformat()
        result_dict["test_results"] = [
            {
                **asdict(tr),
                "status": tr.status.value,
            }
            for tr in result.test_results
        ]

        with open(result_file, "w") as f:
            json.dump(result_dict, f, indent=2, default=str)

        logger.info(f"Saved CI/CD results to {result_file}")

    def deploy_model(
        self,
        model,
        model_version: str,
        deployment_target: str = "production",
    ) -> bool:
        """
        Deploy model (placeholder for actual deployment)

        Args:
            model: Model to deploy
            model_version: Model version
            deployment_target: Deployment target (production, staging)

        Returns:
            True if deployment successful
        """
        logger.info(f"Deploying model {model_version} to {deployment_target}")

        # In production, this would:
        # 1. Update model serving endpoint
        # 2. Update model registry
        # 3. Notify monitoring system
        # 4. Perform health checks
        # 5. Update deployment configuration

        # For now, just log
        logger.info(f"Model {model_version} deployed to {deployment_target}")

        return True


def create_cicd_pipeline(
    pipeline_name: str = "ml_model_cicd",
    custom_gates: List[ValidationGate] = None,
) -> MLModelCICD:
    """
    Convenience function to create CI/CD pipeline

    Args:
        pipeline_name: Name of pipeline
        custom_gates: Custom validation gates

    Returns:
        Configured MLModelCICD instance
    """
    return MLModelCICD(
        pipeline_name=pipeline_name,
        validation_gates=custom_gates,
    )
