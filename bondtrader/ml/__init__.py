"""
Machine Learning modules for bond price adjustment

ML Adjuster Hierarchy:
- MLBondAdjuster: Basic ML adjuster with standard sklearn models
- EnhancedMLBondAdjuster: Enhanced with hyperparameter tuning and cross-validation
- AdvancedMLBondAdjuster: Advanced with deep learning, ensembles, and explainable AI
"""

from bondtrader.ml.automl import AutoMLBondAdjuster
from bondtrader.ml.bayesian_optimization import BayesianOptimizer
from bondtrader.ml.drift_detection import DriftDetector, ModelTuner
from bondtrader.ml.ml_adjuster import MLBondAdjuster
from bondtrader.ml.ml_adjuster_unified import MLBondAdjuster as UnifiedMLBondAdjuster
from bondtrader.ml.training_framework import TrainingConfig, UnifiedTrainingFramework


# Backward compatibility aliases for enhanced and advanced adjusters
def EnhancedMLBondAdjuster(model_type: str = "random_forest", valuator=None):
    """Backward compatible alias for enhanced ML adjuster"""
    import warnings

    warnings.warn(
        "EnhancedMLBondAdjuster is deprecated. Use MLBondAdjuster(feature_level='enhanced') instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return UnifiedMLBondAdjuster(model_type=model_type, feature_level="enhanced", valuator=valuator, use_ensemble=False)


def AdvancedMLBondAdjuster(valuator=None, use_ensemble: bool = True):
    """Backward compatible alias for advanced ML adjuster"""
    import warnings

    warnings.warn(
        "AdvancedMLBondAdjuster is deprecated. Use MLBondAdjuster(feature_level='advanced', use_ensemble=True) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return UnifiedMLBondAdjuster(
        model_type="random_forest", feature_level="advanced", valuator=valuator, use_ensemble=use_ensemble
    )


from bondtrader.ml.regime_models import RegimeDetector

# New MLOps modules
try:
    from bondtrader.ml.ab_testing import (
        ABTestConfig,
        ABTestFramework,
        ABTestResult,
        Variant,
        create_ab_test,
    )
    from bondtrader.ml.automated_retraining import (
        AutomatedRetrainingPipeline,
        RetrainingConfig,
        RetrainingResult,
        RetrainingTrigger,
        create_retraining_pipeline,
    )
    from bondtrader.ml.cicd_pipeline import (
        CICDResult,
        MLModelCICD,
        TestResult,
        TestStatus,
        ValidationGate,
        create_cicd_pipeline,
    )
    from bondtrader.ml.data_lineage import (
        DataLineageTracker,
        DatasetVersion,
        FeatureLineage,
        ModelLineage,
        get_lineage_tracker,
    )
    from bondtrader.ml.data_validation import DataSchema, DataValidator, ValidationResult, create_default_schema
    from bondtrader.ml.explainability import (
        Explanation,
        GlobalExplanation,
        ModelExplainer,
        create_model_explainer,
    )
    from bondtrader.ml.feature_store import FeatureStore, get_feature_store
    from bondtrader.ml.mlflow_tracking import MLflowTracker, track_training_run
    from bondtrader.ml.model_serving import (
        ModelServer,
        PredictionCache,
        PredictionRequest,
        PredictionResponse,
        create_model_server,
    )
    from bondtrader.ml.production_monitoring import ModelMonitor, MonitoringMetrics, PredictionRecord

    HAS_MLOPS = True
except ImportError:
    HAS_MLOPS = False

__all__ = [
    "MLBondAdjuster",
    "EnhancedMLBondAdjuster",
    "AdvancedMLBondAdjuster",
    "AutoMLBondAdjuster",
    "BayesianOptimizer",
    "DriftDetector",
    "ModelTuner",
    "RegimeDetector",
    "TrainingConfig",
    "UnifiedTrainingFramework",
]

if HAS_MLOPS:
    __all__.extend(
        [
            "MLflowTracker",
            "track_training_run",
            "DataValidator",
            "DataSchema",
            "ValidationResult",
            "create_default_schema",
            "FeatureStore",
            "get_feature_store",
            "ModelMonitor",
            "PredictionRecord",
            "MonitoringMetrics",
            "AutomatedRetrainingPipeline",
            "RetrainingConfig",
            "RetrainingTrigger",
            "RetrainingResult",
            "create_retraining_pipeline",
            "ABTestFramework",
            "ABTestConfig",
            "ABTestResult",
            "Variant",
            "create_ab_test",
            "ModelServer",
            "PredictionRequest",
            "PredictionResponse",
            "PredictionCache",
            "create_model_server",
            "ModelExplainer",
            "Explanation",
            "GlobalExplanation",
            "create_model_explainer",
            "DataLineageTracker",
            "DatasetVersion",
            "FeatureLineage",
            "ModelLineage",
            "get_lineage_tracker",
            "MLModelCICD",
            "ValidationGate",
            "TestResult",
            "TestStatus",
            "CICDResult",
            "create_cicd_pipeline",
        ]
    )
