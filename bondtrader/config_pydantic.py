"""
Pydantic-Enhanced Configuration (OPTIONAL)
Optional Pydantic validation for configuration management

NOTE: This module is optional and currently not used in the codebase.
The standard config.py module (using dataclasses) is the default.
This module is provided for users who want Pydantic validation.
To use it, you would need to manually import and use ConfigPydantic instead of Config.
"""

import os
from pathlib import Path
from typing import Dict, Optional

# Optional Pydantic for validation
try:
    from pydantic import BaseModel, Field, field_validator
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    BaseModel = object
    Field = None
    field_validator = None


if HAS_PYDANTIC:

    class ConfigPydantic(BaseModel):
        """Pydantic-validated configuration class for BondTrader"""

        # Default risk-free rate
        default_risk_free_rate: float = Field(
            default=float(os.getenv("DEFAULT_RFR", "0.03")), ge=0, le=1, description="Default risk-free rate"
        )

        # ML settings
        ml_model_type: str = Field(
            default=os.getenv("ML_MODEL_TYPE", "random_forest"),
            description="ML model type (random_forest, gradient_boosting, xgboost, lightgbm, catboost)",
        )
        ml_random_state: int = Field(default=int(os.getenv("ML_RANDOM_STATE", "42")), description="Random seed for ML")
        ml_test_size: float = Field(
            default=float(os.getenv("ML_TEST_SIZE", "0.2")), gt=0, lt=1, description="Test set size (0-1)"
        )

        # Training settings
        training_batch_size: int = Field(
            default=int(os.getenv("TRAINING_BATCH_SIZE", "100")), gt=0, description="Training batch size"
        )
        training_num_bonds: int = Field(
            default=int(os.getenv("TRAINING_NUM_BONDS", "5000")), gt=0, description="Number of bonds for training"
        )
        training_time_periods: int = Field(
            default=int(os.getenv("TRAINING_TIME_PERIODS", "60")), gt=0, description="Number of time periods"
        )

        # Evaluation settings
        evaluation_batch_size: int = Field(
            default=int(os.getenv("EVALUATION_BATCH_SIZE", "100")), gt=0, description="Evaluation batch size"
        )
        evaluation_use_parallel: bool = Field(
            default=os.getenv("EVALUATION_USE_PARALLEL", "true").lower() == "true",
            description="Use parallel evaluation",
        )

        # Paths
        model_dir: str = Field(default=os.getenv("MODEL_DIR", "trained_models"), description="Model directory")
        data_dir: str = Field(default=os.getenv("DATA_DIR", "training_data"), description="Data directory")
        evaluation_data_dir: str = Field(
            default=os.getenv("EVALUATION_DATA_DIR", "evaluation_data"), description="Evaluation data directory"
        )
        evaluation_results_dir: str = Field(
            default=os.getenv("EVALUATION_RESULTS_DIR", "evaluation_results"), description="Evaluation results directory"
        )
        logs_dir: str = Field(default=os.getenv("LOGS_DIR", "logs"), description="Logs directory")
        checkpoint_dir: str = Field(
            default=os.getenv("CHECKPOINT_DIR", "training_checkpoints"), description="Checkpoint directory"
        )

        # API keys (optional)
        fred_api_key: Optional[str] = Field(default=os.getenv("FRED_API_KEY", None), description="FRED API key")

        # Performance settings
        max_workers: Optional[int] = Field(
            default=int(os.getenv("MAX_WORKERS")) if os.getenv("MAX_WORKERS") else None,
            gt=0,
            description="Maximum workers for parallel processing",
        )

        # Logging settings
        log_level: str = Field(
            default=os.getenv("LOG_LEVEL", "INFO"),
            description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
        )
        log_file: str = Field(default=os.getenv("LOG_FILE", "bond_trading.log"), description="Log file path")

        # Arbitrage detection settings
        min_profit_threshold: float = Field(
            default=float(os.getenv("MIN_PROFIT_THRESHOLD", "0.01")), ge=0, description="Minimum profit threshold"
        )
        include_transaction_costs: bool = Field(
            default=os.getenv("INCLUDE_TRANSACTION_COSTS", "true").lower() == "true",
            description="Include transaction costs",
        )

        @field_validator("ml_model_type")
        @classmethod
        def validate_ml_model_type(cls, v: str) -> str:
            """Validate ML model type"""
            valid_types = ["random_forest", "gradient_boosting", "xgboost", "lightgbm", "catboost"]
            if v not in valid_types:
                raise ValueError(f"ml_model_type must be one of {valid_types}")
            return v

        @field_validator("log_level")
        @classmethod
        def validate_log_level(cls, v: str) -> str:
            """Validate log level"""
            valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            if v.upper() not in valid_levels:
                raise ValueError(f"log_level must be one of {valid_levels}")
            return v.upper()

        def model_post_init(self, __context):
            """Post-initialization: create directories"""
            for dir_path in [
                self.model_dir,
                self.data_dir,
                self.evaluation_data_dir,
                self.evaluation_results_dir,
                self.logs_dir,
                self.checkpoint_dir,
            ]:
                Path(dir_path).mkdir(parents=True, exist_ok=True)

        def to_dict(self) -> Dict:
            """Convert configuration to dictionary"""
            return self.model_dump()

        @classmethod
        def from_dict(cls, config_dict: Dict) -> "ConfigPydantic":
            """Create configuration from dictionary"""
            return cls(**config_dict)

else:
    # Fallback if Pydantic not available
    class ConfigPydantic:
        """Fallback ConfigPydantic (Pydantic not available)"""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Pydantic not installed. Install with: pip install pydantic. "
                "Or use the standard Config class from bondtrader.config"
            )
