"""
Integration tests for training pipeline
Tests end-to-end model training workflows
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.fixtures.bond_factory import create_multiple_bonds

from bondtrader.core.bond_models import Bond, BondType
from bondtrader.data.training_data_generator import TrainingDataGenerator
from bondtrader.ml.ml_adjuster import MLBondAdjuster
from bondtrader.ml.ml_adjuster_enhanced import EnhancedMLBondAdjuster


@pytest.fixture
def training_bonds():
    """Create bonds for training (need at least 10)"""
    return create_multiple_bonds(count=20)


@pytest.fixture
def temp_training_dir():
    """Create temporary directory for training outputs"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestTrainingPipeline:
    """Integration tests for training pipeline"""

    def test_end_to_end_training_workflow(self, training_bonds, temp_training_dir):
        """Test complete training workflow from data to model"""
        # Step 1: Initialize trainer
        adjuster = MLBondAdjuster(model_type="random_forest")
        assert not adjuster.is_trained

        # Step 2: Train model
        metrics = adjuster.train(training_bonds, test_size=0.2, random_state=42)

        # Step 3: Verify training succeeded
        assert adjuster.is_trained
        assert adjuster.model is not None
        assert "train_r2" in metrics
        assert "test_r2" in metrics
        assert metrics["train_r2"] > 0  # Should have some predictive power

        # Step 4: Save model
        model_path = os.path.join(temp_training_dir, "test_model.joblib")
        adjuster.save_model(model_path)
        assert os.path.exists(model_path)

        # Step 5: Load model
        new_adjuster = MLBondAdjuster()
        new_adjuster.load_model(model_path)

        # Step 6: Verify loaded model works
        assert new_adjuster.is_trained
        assert new_adjuster.model is not None

        # Step 7: Test prediction with loaded model
        test_bond = training_bonds[0]
        prediction = new_adjuster.predict_adjusted_value(test_bond)
        assert "theoretical_fair_value" in prediction
        assert "ml_adjusted_fair_value" in prediction

    def test_enhanced_training_workflow(self, training_bonds, temp_training_dir):
        """Test enhanced ML training workflow with hyperparameter tuning"""
        # Step 1: Initialize enhanced trainer
        adjuster = EnhancedMLBondAdjuster(model_type="random_forest")
        assert not adjuster.is_trained

        # Step 2: Train with tuning
        metrics = adjuster.train_with_tuning(training_bonds, tune_hyperparameters=False, random_state=42)

        # Step 3: Verify training succeeded
        assert adjuster.is_trained
        assert "test_r2" in metrics
        assert adjuster.best_params is not None

        # Step 4: Save and reload
        model_path = os.path.join(temp_training_dir, "enhanced_model.joblib")
        adjuster.save_model(model_path)

        new_adjuster = EnhancedMLBondAdjuster()
        new_adjuster.load_model(model_path)
        assert new_adjuster.is_trained
        assert new_adjuster.best_params == adjuster.best_params

    def test_training_data_generation_pipeline(self, temp_training_dir):
        """Test training data generation pipeline"""
        # Step 1: Initialize generator
        generator = TrainingDataGenerator(seed=42)

        # Step 2: Generate training data
        dataset = generator.generate_training_dataset(num_bonds=50, num_periods=10)

        # Step 3: Verify dataset structure
        assert "train_bonds" in dataset
        assert "validation_bonds" in dataset
        assert "test_bonds" in dataset
        assert len(dataset["train_bonds"]) > 0

        # Step 4: Save dataset
        dataset_path = os.path.join(temp_training_dir, "training_dataset.joblib")
        generator.save_training_dataset(dataset, dataset_path)
        assert os.path.exists(dataset_path)

        # Step 5: Load dataset
        loaded_dataset = generator.load_training_dataset(dataset_path)
        assert len(loaded_dataset["train_bonds"]) == len(dataset["train_bonds"])

    def test_model_training_with_saved_dataset(self, temp_training_dir):
        """Test training model with pre-generated dataset"""
        # Step 1: Generate and save dataset
        generator = TrainingDataGenerator(seed=42)
        dataset = generator.generate_training_dataset(num_bonds=30, num_periods=5)
        dataset_path = os.path.join(temp_training_dir, "dataset.joblib")
        generator.save_training_dataset(dataset, dataset_path)

        # Step 2: Load dataset
        loaded_dataset = generator.load_training_dataset(dataset_path)
        train_bonds = loaded_dataset["train_bonds"]

        # Step 3: Train model with loaded bonds
        adjuster = MLBondAdjuster()
        metrics = adjuster.train(train_bonds, test_size=0.2, random_state=42)

        # Step 4: Verify training succeeded
        assert adjuster.is_trained
        assert "test_r2" in metrics

    def test_training_error_recovery(self, training_bonds):
        """Test training pipeline error recovery"""
        # Test with insufficient bonds (should fail gracefully)
        adjuster = MLBondAdjuster()
        with pytest.raises(ValueError, match="Need at least 10 bonds"):
            adjuster.train(training_bonds[:5])  # Only 5 bonds

        # Test with valid bonds (should succeed)
        metrics = adjuster.train(training_bonds[:15], test_size=0.2, random_state=42)
        assert adjuster.is_trained
        assert "test_r2" in metrics


@pytest.mark.integration
@pytest.mark.slow
class TestTrainingPerformance:
    """Performance tests for training pipeline"""

    def test_training_performance_with_large_dataset(self, temp_training_dir):
        """Test training performance with larger dataset"""
        # Generate larger dataset
        generator = TrainingDataGenerator(seed=42)
        dataset = generator.generate_training_dataset(num_bonds=100, num_periods=10)

        # Train model
        adjuster = MLBondAdjuster(model_type="random_forest")
        import time

        start_time = time.time()
        metrics = adjuster.train(dataset["train_bonds"], test_size=0.2, random_state=42)
        elapsed_time = time.time() - start_time

        # Should complete within reasonable time (adjust threshold as needed)
        assert elapsed_time < 60  # Should complete within 60 seconds
        assert adjuster.is_trained
        assert metrics["test_r2"] >= 0  # Should have reasonable performance
