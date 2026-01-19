"""
Unit tests for ML bond adjuster module
"""

import os
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.unit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import sys

# Add parent directories to path for fixture imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fixtures.bond_factory import create_multiple_bonds, create_test_bond

from bondtrader.core.bond_models import Bond, BondType
from bondtrader.ml.ml_adjuster import MLBondAdjuster


@pytest.fixture
def ml_adjuster():
    """Create ML adjuster instance"""
    return MLBondAdjuster(model_type="random_forest")


@pytest.fixture
def training_bonds():
    """Create bonds for training (need at least 10)"""
    return create_multiple_bonds(count=15)


def test_ml_adjuster_initialization():
    """Test ML adjuster initialization"""
    adjuster = MLBondAdjuster()
    assert adjuster.model_type == "random_forest"
    assert adjuster.model is None
    assert not adjuster.is_trained


def test_ml_adjuster_initialization_with_type():
    """Test ML adjuster initialization with model type"""
    adjuster = MLBondAdjuster(model_type="gradient_boosting")
    assert adjuster.model_type == "gradient_boosting"


def test_ml_adjuster_train_success(training_bonds, ml_adjuster):
    """Test successful model training"""
    metrics = ml_adjuster.train(training_bonds, test_size=0.2, random_state=42)

    assert ml_adjuster.is_trained
    assert ml_adjuster.model is not None
    assert "train_r2" in metrics
    assert "test_r2" in metrics
    assert "train_mse" in metrics
    assert "test_mse" in metrics


def test_ml_adjuster_train_insufficient_bonds(ml_adjuster):
    """Test training with insufficient bonds"""
    bonds = create_multiple_bonds(count=5)

    with pytest.raises(ValueError, match="Need at least 10 bonds"):
        ml_adjuster.train(bonds)


def test_ml_adjuster_create_features(training_bonds, ml_adjuster):
    """Test feature creation"""
    fair_values = [ml_adjuster.valuator.calculate_fair_value(bond) for bond in training_bonds]
    features, feature_names = ml_adjuster._create_features(training_bonds, fair_values)

    assert isinstance(features, np.ndarray)
    assert len(features) == len(training_bonds)
    assert features.shape[1] > 0  # Has features


def test_ml_adjuster_create_targets(training_bonds, ml_adjuster):
    """Test target creation"""
    fair_values = [ml_adjuster.valuator.calculate_fair_value(bond) for bond in training_bonds]
    targets = ml_adjuster._create_targets(training_bonds, fair_values)

    assert isinstance(targets, np.ndarray)
    assert len(targets) == len(training_bonds)
    assert all(targets > 0)


def test_ml_adjuster_predict(training_bonds, ml_adjuster):
    """Test prediction after training"""
    ml_adjuster.train(training_bonds, test_size=0.2, random_state=42)

    test_bond = create_test_bond()
    result = ml_adjuster.predict_adjusted_value(test_bond)

    assert "theoretical_fair_value" in result
    assert "ml_adjusted_fair_value" in result
    assert "adjustment_factor" in result
    assert result["ml_adjusted_fair_value"] > 0


def test_ml_adjuster_predict_before_training(ml_adjuster):
    """Test prediction before training returns base fair value"""
    test_bond = create_test_bond()
    result = ml_adjuster.predict_adjusted_value(test_bond)

    # Should return basic fair value without error
    assert "theoretical_fair_value" in result
    assert result["adjustment_factor"] == 1.0
    assert result["ml_confidence"] == 0.0


def test_ml_adjuster_save_load(training_bonds, ml_adjuster, tmp_path):
    """Test saving and loading model"""
    import os

    ml_adjuster.train(training_bonds, test_size=0.2, random_state=42)

    # Change to tmp_path directory and use relative filename for security check
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        # Use just the filename (relative path) to avoid absolute path validation error
        model_filename = "test_model.pkl"
        ml_adjuster.save_model(model_filename)

        # Create new adjuster and load
        new_adjuster = MLBondAdjuster(model_type=ml_adjuster.model_type)
        new_adjuster.load_model(model_filename)

        assert new_adjuster.is_trained
        assert new_adjuster.model is not None

        # Test prediction works
        test_bond = create_test_bond()
        original_pred = ml_adjuster.predict_adjusted_value(test_bond)
        loaded_pred = new_adjuster.predict_adjusted_value(test_bond)

        assert (
            abs(original_pred["ml_adjusted_fair_value"] - loaded_pred["ml_adjusted_fair_value"])
            < 1e-6
        )
    finally:
        os.chdir(original_cwd)


def test_ml_adjuster_gradient_boosting(training_bonds):
    """Test gradient boosting model"""
    adjuster = MLBondAdjuster(model_type="gradient_boosting")
    metrics = adjuster.train(training_bonds, test_size=0.2, random_state=42)

    assert adjuster.is_trained
    assert "train_r2" in metrics


def test_ml_adjuster_invalid_model_type(training_bonds):
    """Test invalid model type during training"""
    adjuster = MLBondAdjuster(model_type="random_forest")
    adjuster.model_type = "invalid_type"

    with pytest.raises(ValueError, match="Model type 'invalid_type' not available"):
        adjuster.train(training_bonds)
