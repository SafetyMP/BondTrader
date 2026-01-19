"""
Unit tests for enhanced ML bond adjuster module
"""

import os
import sys

import pytest

pytestmark = pytest.mark.unit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fixtures.bond_factory import create_multiple_bonds

from bondtrader.ml.ml_adjuster_enhanced import EnhancedMLBondAdjuster


@pytest.fixture
def enhanced_ml_adjuster():
    """Create enhanced ML adjuster instance"""
    return EnhancedMLBondAdjuster(model_type="random_forest")


@pytest.fixture
def training_bonds():
    """Create bonds for training (need at least 10)"""
    return create_multiple_bonds(count=15)


def test_enhanced_ml_adjuster_initialization():
    """Test enhanced ML adjuster initialization"""
    adjuster = EnhancedMLBondAdjuster()
    assert adjuster.model_type == "random_forest"
    assert adjuster.model is None
    assert not adjuster.is_trained
    assert adjuster.feature_names == []


def test_enhanced_ml_adjuster_train_success(training_bonds, enhanced_ml_adjuster):
    """Test successful model training"""
    metrics = enhanced_ml_adjuster.train_with_tuning(training_bonds, test_size=0.2, random_state=42)

    assert enhanced_ml_adjuster.is_trained
    assert enhanced_ml_adjuster.model is not None
    assert len(enhanced_ml_adjuster.feature_names) > 0
    assert "train_r2" in metrics
    assert "test_r2" in metrics


def test_enhanced_ml_adjuster_create_enhanced_features(training_bonds, enhanced_ml_adjuster):
    """Test enhanced feature creation"""
    fair_values = [enhanced_ml_adjuster.valuator.calculate_fair_value(bond) for bond in training_bonds]
    features, feature_names = enhanced_ml_adjuster._create_enhanced_features(training_bonds, fair_values)

    assert len(feature_names) > 0
    assert features.shape[0] == len(training_bonds)
    assert features.shape[1] == len(feature_names)


def test_enhanced_ml_adjuster_predict(training_bonds, enhanced_ml_adjuster):
    """Test prediction after training"""
    enhanced_ml_adjuster.train_with_tuning(training_bonds, test_size=0.2, random_state=42)

    from fixtures.bond_factory import create_test_bond

    test_bond = create_test_bond()
    result = enhanced_ml_adjuster.predict_adjusted_value(test_bond)

    assert "theoretical_fair_value" in result
    assert "ml_adjusted_fair_value" in result
    assert result["ml_adjusted_fair_value"] > 0
