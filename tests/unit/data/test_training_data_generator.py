"""
Unit tests for training data generator module
"""

import os
import sys

import pytest

pytestmark = pytest.mark.unit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bondtrader.data.training_data_generator import MarketRegime, TrainingDataGenerator


def test_training_data_generator_initialization():
    """Test training data generator initialization"""
    generator = TrainingDataGenerator(seed=42)
    assert generator.seed == 42
    assert generator.valuator is not None
    assert generator.base_generator is not None


def test_market_regime_initialization():
    """Test MarketRegime dataclass"""
    regime = MarketRegime(
        regime_name="Test",
        risk_free_rate=0.03,
        volatility_multiplier=1.0,
        credit_spread_base=0.0,
        liquidity_factor=1.0,
        market_sentiment=0.0,
    )

    assert regime.regime_name == "Test"
    assert regime.risk_free_rate == 0.03
    assert regime.volatility_multiplier == 1.0


def test_generate_comprehensive_dataset_basic():
    """Test basic dataset generation"""
    generator = TrainingDataGenerator(seed=42)

    # Use small dataset for testing
    dataset = generator.generate_comprehensive_dataset(
        total_bonds=100,  # Small for tests
        time_periods=5,  # Few periods
        bonds_per_period=10,
        train_split=0.7,
        validation_split=0.15,
        test_split=0.15,
    )

    assert "train" in dataset
    assert "validation" in dataset
    assert "test" in dataset
    assert "bonds" in dataset["train"]
    assert len(dataset["train"]["bonds"]) > 0


def test_generate_comprehensive_dataset_splits():
    """Test dataset generation with different splits"""
    generator = TrainingDataGenerator(seed=42)

    dataset = generator.generate_comprehensive_dataset(
        total_bonds=100,
        time_periods=5,
        bonds_per_period=10,
        train_split=0.6,
        validation_split=0.2,
        test_split=0.2,
    )

    # Verify splits are approximately correct
    train_size = len(dataset["train"]["bonds"])
    val_size = len(dataset["validation"]["bonds"])
    test_size = len(dataset["test"]["bonds"])
    total = train_size + val_size + test_size

    assert total > 0
    assert train_size / total >= 0.5  # At least 50% in train


def test_generate_comprehensive_dataset_invalid_splits():
    """Test dataset generation with invalid splits"""
    generator = TrainingDataGenerator(seed=42)

    with pytest.raises(ValueError, match="Splits must sum to 1.0"):
        generator.generate_comprehensive_dataset(
            total_bonds=100,
            time_periods=5,
            train_split=0.6,
            validation_split=0.3,
            test_split=0.2,  # Doesn't sum to 1.0
        )
