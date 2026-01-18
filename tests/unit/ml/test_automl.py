"""
Unit tests for AutoML bond adjuster module
"""

import os
import sys

import pytest

pytestmark = pytest.mark.unit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fixtures.bond_factory import create_multiple_bonds

from bondtrader.ml.automl import AutoMLBondAdjuster


@pytest.fixture
def automl_adjuster():
    """Create AutoML adjuster instance"""
    return AutoMLBondAdjuster()


@pytest.fixture
def training_bonds():
    """Create bonds for training (need at least 20)"""
    return create_multiple_bonds(count=25)


def test_automl_adjuster_initialization():
    """Test AutoML adjuster initialization"""
    adjuster = AutoMLBondAdjuster()
    assert adjuster.valuator is not None
    assert adjuster.best_model is None
    assert not adjuster.is_trained


def test_automated_model_selection_insufficient_bonds(automl_adjuster):
    """Test AutoML with insufficient bonds"""
    bonds = create_multiple_bonds(count=10)

    with pytest.raises(ValueError, match="Need at least 20 bonds"):
        automl_adjuster.automated_model_selection(bonds)


def test_automated_model_selection_basic(training_bonds, automl_adjuster):
    """Test automated model selection (may take time, so mark as slow)"""
    # Use limited candidate models and quick evaluation for tests
    result = automl_adjuster.automated_model_selection(
        training_bonds,
        candidate_models=["random_forest"],  # Test with one model only
        max_evaluation_time=10,  # Short timeout for tests
    )

    assert "best_model" in result
    assert "automl_success" in result
    assert result["best_model"] in ["random_forest", "gradient_boosting", "neural_network", "ensemble"]
