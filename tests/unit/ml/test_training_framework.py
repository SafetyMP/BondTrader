"""
Unit tests for training framework
"""

import pytest
from unittest.mock import MagicMock, patch

from bondtrader.ml.training_framework import UnifiedTrainingFramework, TrainingConfig


@pytest.mark.unit
class TestTrainingConfig:
    """Test TrainingConfig class"""

    def test_training_config_creation(self):
        """Test creating training config"""
        config = TrainingConfig(model_type="random_forest", feature_level="basic")
        assert config.model_type == "random_forest"
        assert config.feature_level == "basic"


@pytest.mark.unit
class TestUnifiedTrainingFramework:
    """Test UnifiedTrainingFramework class"""

    def test_training_framework_creation(self):
        """Test creating training framework"""
        with patch("bondtrader.ml.training_framework.get_container") as mock_container:
            mock_container.return_value.get_valuator.return_value = MagicMock()
            framework = UnifiedTrainingFramework(generate_new_dataset=False)
            assert framework is not None

    def test_training_framework_with_config(self):
        """Test training framework with config"""
        with patch("bondtrader.ml.training_framework.get_container") as mock_container:
            mock_container.return_value.get_valuator.return_value = MagicMock()
            config = TrainingConfig(model_type="random_forest")
            framework = UnifiedTrainingFramework(generate_new_dataset=False)
            assert framework is not None