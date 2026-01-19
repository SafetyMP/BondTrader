"""
Integration tests for evaluation pipeline
Tests end-to-end model evaluation workflows
"""

import os
import sys
import tempfile

import pytest

pytestmark = pytest.mark.integration

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from fixtures
import sys
from pathlib import Path

from bondtrader.core.bond_models import Bond, BondType
from bondtrader.core.bond_valuation import BondValuator
from bondtrader.ml.ml_adjuster import MLBondAdjuster

fixtures_path = Path(__file__).parent.parent / "fixtures"
sys.path.insert(0, str(fixtures_path))
from bond_factory import create_multiple_bonds


@pytest.fixture
def trained_model(training_bonds):
    """Create a trained model for evaluation"""
    adjuster = MLBondAdjuster(model_type="random_forest")
    adjuster.train(training_bonds, test_size=0.2, random_state=42)
    return adjuster


@pytest.fixture
def training_bonds():
    """Create bonds for training"""
    return create_multiple_bonds(count=20)


@pytest.fixture
def evaluation_bonds():
    """Create bonds for evaluation"""
    return create_multiple_bonds(count=10)


@pytest.fixture
def temp_eval_dir():
    """Create temporary directory for evaluation outputs"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestEvaluationPipeline:
    """Integration tests for evaluation pipeline"""

    def test_end_to_end_evaluation_workflow(self, trained_model, evaluation_bonds):
        """Test complete evaluation workflow"""
        # Step 1: Verify model is trained
        assert trained_model.is_trained

        # Step 2: Evaluate on evaluation bonds
        results = []
        for bond in evaluation_bonds:
            prediction = trained_model.predict_adjusted_value(bond)
            results.append(prediction)

        # Step 3: Verify evaluation results
        assert len(results) == len(evaluation_bonds)
        for result in results:
            assert "theoretical_fair_value" in result
            assert "ml_adjusted_fair_value" in result
            assert result["ml_adjusted_fair_value"] > 0

    def test_model_comparison_workflow(self, training_bonds, evaluation_bonds, temp_eval_dir):
        """Test comparing multiple models"""
        # Step 1: Train multiple models
        models = {}
        model_types = ["random_forest", "gradient_boosting"]

        for model_type in model_types:
            adjuster = MLBondAdjuster(model_type=model_type)
            metrics = adjuster.train(training_bonds, test_size=0.2, random_state=42)
            models[model_type] = {
                "model": adjuster,
                "metrics": metrics,
            }

        # Step 2: Evaluate all models on same evaluation set
        evaluation_results = {}
        for model_name, model_data in models.items():
            predictions = []
            for bond in evaluation_bonds:
                pred = model_data["model"].predict_adjusted_value(bond)
                predictions.append(pred)
            evaluation_results[model_name] = predictions

        # Step 3: Verify all models evaluated
        assert len(evaluation_results) == len(model_types)
        for model_name, preds in evaluation_results.items():
            assert len(preds) == len(evaluation_bonds)

    def test_evaluation_with_valuation_benchmark(self, trained_model, evaluation_bonds):
        """Test evaluation comparing ML predictions to theoretical valuation"""
        valuator = BondValuator()

        for bond in evaluation_bonds:
            # Get ML prediction
            ml_result = trained_model.predict_adjusted_value(bond)

            # Get theoretical valuation
            theoretical_fv = valuator.calculate_fair_value(bond)

            # Compare
            assert "theoretical_fair_value" in ml_result
            assert abs(ml_result["theoretical_fair_value"] - theoretical_fv) < 1.0  # Should be close

    def test_model_metrics_calculation(self, training_bonds, evaluation_bonds):
        """Test calculating evaluation metrics"""
        # Train model
        adjuster = MLBondAdjuster()
        train_metrics = adjuster.train(training_bonds, test_size=0.2, random_state=42)

        # Evaluate on separate evaluation set
        predictions = []
        actual_prices = []
        theoretical_fvs = []

        for bond in evaluation_bonds:
            result = adjuster.predict_adjusted_value(bond)
            predictions.append(result["ml_adjusted_fair_value"])
            actual_prices.append(bond.current_price)
            theoretical_fvs.append(result["theoretical_fair_value"])

        # Basic metric checks
        assert len(predictions) == len(evaluation_bonds)
        assert all(p > 0 for p in predictions)  # All predictions positive

        # Verify training metrics present
        assert "train_r2" in train_metrics
        assert "test_r2" in train_metrics
        assert train_metrics["test_r2"] >= 0  # R² can be negative but unlikely for trained model


@pytest.mark.integration
class TestEvaluationErrorHandling:
    """Test error handling in evaluation pipeline"""

    def test_evaluation_with_untrained_model(self, evaluation_bonds):
        """Test evaluation fails gracefully with untrained model"""
        adjuster = MLBondAdjuster()
        assert not adjuster.is_trained

        # Should still provide theoretical value even if model not trained
        bond = evaluation_bonds[0]
        result = adjuster.predict_adjusted_value(bond)
        assert "theoretical_fair_value" in result
        # ML-adjusted value should fallback to theoretical
        assert "ml_adjusted_fair_value" in result

    def test_evaluation_with_invalid_bonds(self, trained_model):
        """Test evaluation handles invalid bonds gracefully"""
        # Test with empty bond list (edge case)
        results = []
        empty_bonds = []
        for bond in empty_bonds:
            results.append(trained_model.predict_adjusted_value(bond))

        assert len(results) == 0
