"""
Data Validation Pipeline
Comprehensive data validation before training (industry best practice)

Implements:
- Schema validation
- Statistical validation
- Data drift detection
- Automated quality reports
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from bondtrader.utils.utils import logger


@dataclass
class ValidationResult:
    """Result of data validation"""

    passed: bool
    checks: Dict[str, bool]
    errors: List[str]
    warnings: List[str]
    statistics: Dict[str, Any]
    quality_score: float  # 0-1, where 1 is perfect quality


@dataclass
class DataSchema:
    """Schema definition for data validation"""

    feature_names: List[str]
    feature_types: Dict[str, type]
    feature_ranges: Dict[str, Tuple[float, float]]  # (min, max) for numeric features
    required_features: List[str]
    allow_missing: bool = False
    max_missing_pct: float = 0.05  # Maximum percentage of missing values allowed


class DataValidator:
    """
    Comprehensive data validation pipeline

    Industry Best Practices:
    - Schema validation
    - Statistical validation
    - Data drift detection
    - Automated quality reports
    """

    def __init__(self, schema: DataSchema = None):
        """
        Initialize data validator

        Args:
            schema: Data schema for validation
        """
        self.schema = schema
        self.reference_statistics = None

    def validate_schema(self, X: np.ndarray, feature_names: List[str] = None) -> Dict[str, Any]:
        """
        Validate data schema

        Args:
            X: Feature matrix
            feature_names: Optional feature names

        Returns:
            Validation results
        """
        errors = []
        warnings = []
        checks = {}

        if self.schema is None:
            warnings.append("No schema provided, skipping schema validation")
            return {"passed": True, "errors": errors, "warnings": warnings, "checks": checks}

        # Check feature count
        expected_features = len(self.schema.feature_names)
        actual_features = X.shape[1] if len(X.shape) > 1 else 1

        if actual_features != expected_features:
            errors.append(
                f"Feature count mismatch: expected {expected_features}, got {actual_features}"
            )
            checks["feature_count"] = False
        else:
            checks["feature_count"] = True

        # Check for missing values
        missing_count = np.isnan(X).sum() if len(X) > 0 else 0
        missing_pct = missing_count / X.size if X.size > 0 else 0

        if missing_pct > self.schema.max_missing_pct:
            errors.append(
                f"Too many missing values: {missing_pct:.2%} > {self.schema.max_missing_pct:.2%}"
            )
            checks["missing_values"] = False
        elif missing_pct > 0:
            warnings.append(f"Missing values detected: {missing_pct:.2%}")
            checks["missing_values"] = True
        else:
            checks["missing_values"] = True

        # Check for infinite values
        infinite_count = np.isinf(X).sum() if len(X) > 0 else 0
        if infinite_count > 0:
            errors.append(f"Infinite values detected: {infinite_count}")
            checks["infinite_values"] = False
        else:
            checks["infinite_values"] = True

        # Check feature ranges if provided
        if feature_names and self.schema.feature_ranges:
            for i, feature_name in enumerate(feature_names):
                if feature_name in self.schema.feature_ranges:
                    min_val, max_val = self.schema.feature_ranges[feature_name]
                    feature_data = X[:, i] if len(X.shape) > 1 else X

                    if np.any(feature_data < min_val) or np.any(feature_data > max_val):
                        out_of_range = np.sum((feature_data < min_val) | (feature_data > max_val))
                        warnings.append(
                            f"Feature {feature_name}: {out_of_range} values out of range [{min_val}, {max_val}]"
                        )
                        checks[f"range_{feature_name}"] = False
                    else:
                        checks[f"range_{feature_name}"] = True

        return {
            "passed": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "checks": checks,
        }

    def validate_statistics(
        self, X: np.ndarray, y: np.ndarray = None, reference_statistics: Dict = None
    ) -> Dict[str, Any]:
        """
        Validate statistical properties of data

        Args:
            X: Feature matrix
            y: Target vector (optional)
            reference_statistics: Reference statistics for drift detection

        Returns:
            Validation results
        """
        errors = []
        warnings = []
        checks = {}
        statistics = {}

        if len(X) == 0:
            errors.append("Empty dataset")
            return {
                "passed": False,
                "errors": errors,
                "warnings": warnings,
                "checks": checks,
                "statistics": statistics,
            }

        # Calculate statistics
        statistics["n_samples"] = len(X)
        statistics["n_features"] = X.shape[1] if len(X.shape) > 1 else 1

        # Feature statistics
        if len(X.shape) > 1:
            statistics["feature_means"] = np.nanmean(X, axis=0).tolist()
            statistics["feature_stds"] = np.nanstd(X, axis=0).tolist()
            statistics["feature_mins"] = np.nanmin(X, axis=0).tolist()
            statistics["feature_maxs"] = np.nanmax(X, axis=0).tolist()
        else:
            statistics["feature_means"] = [np.nanmean(X)]
            statistics["feature_stds"] = [np.nanstd(X)]
            statistics["feature_mins"] = [np.nanmin(X)]
            statistics["feature_maxs"] = [np.nanmax(X)]

        # Target statistics
        if y is not None and len(y) > 0:
            statistics["target_mean"] = float(np.nanmean(y))
            statistics["target_std"] = float(np.nanstd(y))
            statistics["target_min"] = float(np.nanmin(y))
            statistics["target_max"] = float(np.nanmax(y))

            # Check target distribution
            if statistics["target_std"] == 0:
                warnings.append("Target has zero variance (constant values)")
                checks["target_variance"] = False
            else:
                checks["target_variance"] = True

            # Check for outliers (values beyond 3 standard deviations)
            z_scores = np.abs((y - statistics["target_mean"]) / statistics["target_std"])
            outliers = np.sum(z_scores > 3)
            if outliers > len(y) * 0.05:  # More than 5% outliers
                warnings.append(
                    f"High number of target outliers: {outliers} ({outliers/len(y):.1%})"
                )
                checks["target_outliers"] = False
            else:
                checks["target_outliers"] = True

        # Detect data drift if reference statistics provided
        if reference_statistics:
            drift_detected = self._detect_drift(statistics, reference_statistics)
            if drift_detected:
                warnings.append("Data drift detected compared to reference statistics")
                checks["data_drift"] = False
            else:
                checks["data_drift"] = True

        # Check for constant features
        if len(X.shape) > 1:
            constant_features = []
            for i in range(X.shape[1]):
                if np.nanstd(X[:, i]) == 0:
                    constant_features.append(i)

            if constant_features:
                warnings.append(f"Constant features detected: {constant_features}")
                checks["constant_features"] = False
            else:
                checks["constant_features"] = True

        return {
            "passed": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "checks": checks,
            "statistics": statistics,
        }

    def _detect_drift(
        self, current_statistics: Dict, reference_statistics: Dict, threshold: float = 0.1
    ) -> bool:
        """
        Detect data drift by comparing statistics

        Args:
            current_statistics: Current data statistics
            reference_statistics: Reference statistics
            threshold: Threshold for drift detection (relative change)

        Returns:
            True if drift detected
        """
        if "feature_means" not in current_statistics or "feature_means" not in reference_statistics:
            return False

        current_means = np.array(current_statistics["feature_means"])
        reference_means = np.array(reference_statistics["feature_means"])

        if len(current_means) != len(reference_means):
            return True  # Different number of features indicates drift

        # Calculate relative change
        reference_std = np.array(
            reference_statistics.get("feature_stds", [1.0] * len(reference_means))
        )
        reference_std = np.where(reference_std == 0, 1.0, reference_std)  # Avoid division by zero

        relative_change = np.abs(current_means - reference_means) / reference_std

        # Drift detected if any feature changed by more than threshold
        return np.any(relative_change > threshold)

    def validate_complete(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
        feature_names: List[str] = None,
        reference_statistics: Dict = None,
    ) -> ValidationResult:
        """
        Complete data validation pipeline

        Args:
            X: Feature matrix
            y: Target vector (optional)
            feature_names: Feature names
            reference_statistics: Reference statistics for drift detection

        Returns:
            ValidationResult with all checks
        """
        all_errors = []
        all_warnings = []
        all_checks = {}
        all_statistics = {}

        # Schema validation
        schema_result = self.validate_schema(X, feature_names)
        all_errors.extend(schema_result["errors"])
        all_warnings.extend(schema_result["warnings"])
        all_checks.update({f"schema_{k}": v for k, v in schema_result["checks"].items()})

        # Statistical validation
        stats_result = self.validate_statistics(X, y, reference_statistics)
        all_errors.extend(stats_result["errors"])
        all_warnings.extend(stats_result["warnings"])
        all_checks.update({f"stats_{k}": v for k, v in stats_result["checks"].items()})
        all_statistics.update(stats_result.get("statistics", {}))

        # Calculate quality score
        passed_checks = sum(1 for v in all_checks.values() if v)
        total_checks = len(all_checks)
        quality_score = passed_checks / total_checks if total_checks > 0 else 0.0

        # Overall pass if no errors
        passed = len(all_errors) == 0

        return ValidationResult(
            passed=passed,
            checks=all_checks,
            errors=all_errors,
            warnings=all_warnings,
            statistics=all_statistics,
            quality_score=quality_score,
        )

    def set_reference_statistics(self, statistics: Dict):
        """Set reference statistics for drift detection"""
        self.reference_statistics = statistics

    def generate_quality_report(self, validation_result: ValidationResult) -> str:
        """
        Generate human-readable quality report

        Args:
            validation_result: Validation result

        Returns:
            Quality report string
        """
        report = []
        report.append("=" * 60)
        report.append("DATA VALIDATION REPORT")
        report.append("=" * 60)
        report.append(f"Status: {'PASSED' if validation_result.passed else 'FAILED'}")
        report.append(f"Quality Score: {validation_result.quality_score:.2%}")
        report.append("")

        if validation_result.errors:
            report.append("ERRORS:")
            for error in validation_result.errors:
                report.append(f"  ✗ {error}")
            report.append("")

        if validation_result.warnings:
            report.append("WARNINGS:")
            for warning in validation_result.warnings:
                report.append(f"  ⚠ {warning}")
            report.append("")

        report.append("CHECKS:")
        for check_name, check_passed in validation_result.checks.items():
            status = "✓" if check_passed else "✗"
            report.append(f"  {status} {check_name}")
        report.append("")

        if validation_result.statistics:
            report.append("STATISTICS:")
            for key, value in validation_result.statistics.items():
                if isinstance(value, (int, float)):
                    report.append(f"  {key}: {value:.4f}")
                elif isinstance(value, list) and len(value) <= 5:
                    report.append(f"  {key}: {value}")
            report.append("")

        report.append("=" * 60)

        return "\n".join(report)


def create_default_schema(feature_names: List[str]) -> DataSchema:
    """
    Create default schema for bond data

    Args:
        feature_names: List of feature names

    Returns:
        DataSchema with default settings
    """
    # Default ranges for common bond features
    feature_ranges = {
        "coupon_rate": (0.0, 0.20),  # 0-20%
        "time_to_maturity": (0.0, 50.0),  # 0-50 years
        "credit_rating_numeric": (0, 20),
        "price_to_par_ratio": (0.5, 1.5),
        "ytm": (0.0, 0.20),  # 0-20%
        "duration": (0.0, 30.0),
        "convexity": (0.0, 1000.0),
    }

    # Default types (all numeric for now)
    feature_types = {name: float for name in feature_names}

    return DataSchema(
        feature_names=feature_names,
        feature_types=feature_types,
        feature_ranges=feature_ranges,
        required_features=feature_names,
        allow_missing=False,
        max_missing_pct=0.05,
    )
