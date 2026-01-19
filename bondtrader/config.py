"""
Configuration Management System
Centralized configuration with environment variable support and validation

NOTE: This is the standard configuration module used throughout the codebase.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class Config:
    """Main configuration class for BondTrader"""

    # Default risk-free rate
    default_risk_free_rate: float = float(os.getenv("DEFAULT_RFR", "0.03"))

    # ML settings
    ml_model_type: str = os.getenv("ML_MODEL_TYPE", "random_forest")
    ml_random_state: int = int(os.getenv("ML_RANDOM_STATE", "42"))
    ml_test_size: float = float(os.getenv("ML_TEST_SIZE", "0.2"))

    # Training settings
    training_batch_size: int = int(os.getenv("TRAINING_BATCH_SIZE", "100"))
    training_num_bonds: int = int(os.getenv("TRAINING_NUM_BONDS", "5000"))
    training_time_periods: int = int(os.getenv("TRAINING_TIME_PERIODS", "60"))

    # Evaluation settings
    evaluation_batch_size: int = int(os.getenv("EVALUATION_BATCH_SIZE", "100"))
    evaluation_use_parallel: bool = os.getenv("EVALUATION_USE_PARALLEL", "true").lower() == "true"

    # Paths
    model_dir: str = os.getenv("MODEL_DIR", "trained_models")
    data_dir: str = os.getenv("DATA_DIR", "training_data")
    evaluation_data_dir: str = os.getenv("EVALUATION_DATA_DIR", "evaluation_data")
    evaluation_results_dir: str = os.getenv("EVALUATION_RESULTS_DIR", "evaluation_results")
    logs_dir: str = os.getenv("LOGS_DIR", "logs")
    checkpoint_dir: str = os.getenv("CHECKPOINT_DIR", "training_checkpoints")

    # API keys (optional)
    fred_api_key: Optional[str] = os.getenv("FRED_API_KEY", None)
    finra_api_key: Optional[str] = os.getenv("FINRA_API_KEY", None)
    finra_api_password: Optional[str] = os.getenv("FINRA_API_PASSWORD", None)
    nasdaq_api_key: Optional[str] = os.getenv("NASDAQ_API_KEY", None)

    # Performance settings
    max_workers: Optional[int] = int(os.getenv("MAX_WORKERS")) if os.getenv("MAX_WORKERS") else None

    # Logging settings
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file: str = os.getenv("LOG_FILE", "bond_trading.log")

    # Arbitrage detection settings
    min_profit_threshold: float = float(os.getenv("MIN_PROFIT_THRESHOLD", "0.01"))
    include_transaction_costs: bool = os.getenv("INCLUDE_TRANSACTION_COSTS", "true").lower() == "true"

    def __post_init__(self):
        """Validate configuration values"""
        self.validate()

    def validate(self) -> None:
        """Validate configuration values"""
        if self.default_risk_free_rate < 0:
            raise ValueError("default_risk_free_rate must be non-negative")

        if not 0 < self.ml_test_size < 1:
            raise ValueError("ml_test_size must be between 0 and 1")

        if self.training_batch_size <= 0:
            raise ValueError("training_batch_size must be positive")

        if self.min_profit_threshold < 0:
            raise ValueError("min_profit_threshold must be non-negative")

        # Create directories if they don't exist
        for dir_path in [
            self.model_dir,
            self.data_dir,
            self.evaluation_data_dir,
            self.evaluation_results_dir,
            self.logs_dir,
            self.checkpoint_dir,
        ]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "default_risk_free_rate": self.default_risk_free_rate,
            "ml_model_type": self.ml_model_type,
            "ml_random_state": self.ml_random_state,
            "ml_test_size": self.ml_test_size,
            "training_batch_size": self.training_batch_size,
            "evaluation_batch_size": self.evaluation_batch_size,
            "model_dir": self.model_dir,
            "data_dir": self.data_dir,
            "min_profit_threshold": self.min_profit_threshold,
            "log_level": self.log_level,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create configuration from dictionary"""
        return cls(**config_dict)


# Global configuration instance
_config_instance: Optional[Config] = None


def get_config() -> Config:
    """Get global configuration instance (singleton pattern)"""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance


def set_config(config: Config) -> None:
    """Set global configuration instance"""
    global _config_instance
    _config_instance = config
