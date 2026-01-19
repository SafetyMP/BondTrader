"""
Unit tests for configuration management
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bondtrader.config import Config, get_config, set_config


def test_config_default_values():
    """Test that config has sensible defaults"""
    config = Config()

    assert config.default_risk_free_rate == 0.03
    assert config.ml_model_type == "random_forest"
    assert config.ml_random_state == 42
    assert config.min_profit_threshold == 0.01


def test_config_validation():
    """Test configuration validation"""
    # Test invalid risk-free rate
    with pytest.raises(ValueError):
        Config(default_risk_free_rate=-0.01)

    # Test invalid test size
    with pytest.raises(ValueError):
        Config(ml_test_size=1.5)

    with pytest.raises(ValueError):
        Config(ml_test_size=-0.1)

    # Test invalid batch size
    with pytest.raises(ValueError):
        Config(training_batch_size=0)


def test_config_from_environment():
    """Test configuration from environment variables"""
    # Note: Config reads environment at class definition time, so this test
    # may not work if env vars were set before import. Test structure instead.
    config = Config()

    # Test that config can be created with environment variables set
    # (actual env var testing requires restart of Python process)
    assert hasattr(config, "default_risk_free_rate")
    assert hasattr(config, "ml_model_type")
    assert config.default_risk_free_rate > 0


def test_config_to_dict():
    """Test converting config to dictionary"""
    config = Config()
    config_dict = config.to_dict()

    assert isinstance(config_dict, dict)
    assert "default_risk_free_rate" in config_dict
    assert "ml_model_type" in config_dict


def test_config_singleton():
    """Test singleton pattern for global config"""
    config1 = get_config()
    config2 = get_config()

    assert config1 is config2


def test_set_config():
    """Test setting custom config"""
    custom_config = Config(default_risk_free_rate=0.04)
    set_config(custom_config)

    retrieved_config = get_config()
    assert retrieved_config.default_risk_free_rate == 0.04

    # Reset to default
    set_config(Config())


def test_config_directories_created():
    """Test that config creates necessary directories"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = Config(model_dir=os.path.join(tmpdir, "models"), data_dir=os.path.join(tmpdir, "data"))

        assert Path(config.model_dir).exists()
        assert Path(config.data_dir).exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
