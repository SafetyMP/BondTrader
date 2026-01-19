"""Machine Learning modules for bond price adjustment"""

from bondtrader.ml.automl import AutoMLBondAdjuster
from bondtrader.ml.bayesian_optimization import BayesianOptimizer
from bondtrader.ml.drift_detection import DriftDetector, ModelTuner
from bondtrader.ml.ml_adjuster import MLBondAdjuster
from bondtrader.ml.ml_adjuster_enhanced import EnhancedMLBondAdjuster
from bondtrader.ml.ml_advanced import AdvancedMLBondAdjuster
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
