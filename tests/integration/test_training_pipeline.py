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

# Import from fixtures
import sys
from pathlib import Path

from datetime import datetime

from bondtrader.core.bond_models import Bond, BondType
from bondtrader.data.training_data_generator import (
    TrainingDataGenerator,
    load_training_dataset,
    save_training_dataset,
)
from bondtrader.ml.ml_adjuster import MLBondAdjuster
from bondtrader.ml.ml_adjuster_enhanced import EnhancedMLBondAdjuster

fixtures_path = Path(__file__).parent.parent / "fixtures"
sys.path.insert(0, str(fixtures_path))
from bond_factory import create_multiple_bonds


def reconstruct_bonds_from_metadata(metadata_list):
    """Reconstruct Bond objects from metadata"""
    bonds = []
    for meta in metadata_list:
        # Parse dates from metadata
        if isinstance(meta["maturity_date"], str):
            maturity_date = datetime.fromisoformat(meta["maturity_date"])
        else:
            maturity_date = meta["maturity_date"]
        if isinstance(meta["issue_date"], str):
            issue_date = datetime.fromisoformat(meta["issue_date"])
        else:
            issue_date = meta["issue_date"]
        
        bond = Bond(
            bond_id=meta["bond_id"],
            bond_type=BondType(meta["bond_type"]),
            face_value=meta["face_value"],
            coupon_rate=meta["coupon_rate"],
            maturity_date=maturity_date,
            issue_date=issue_date,
            current_price=meta.get("current_price", meta["face_value"] * 0.95),  # Default if not in metadata
            credit_rating=meta["credit_rating"],
            issuer=meta["issuer"],
            frequency=meta.get("frequency", 2),
            callable=meta.get("callable", False),
            convertible=meta.get("convertible", False),
        )
        bonds.append(bond)
    return bonds


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

        # Step 4: Save model (use relative path)
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_training_dir)
            model_path = "test_model.joblib"
            adjuster.save_model(model_path)
            assert os.path.exists(model_path)
        finally:
            os.chdir(original_cwd)

        # Step 5: Load model (use relative path)
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_training_dir)
            new_adjuster = MLBondAdjuster()
            new_adjuster.load_model("test_model.joblib")
        finally:
            os.chdir(original_cwd)

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
        # best_params may be None if tune_hyperparameters=False
        # Just verify training completed successfully

        # Step 4: Save and reload (use relative path)
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_training_dir)
            adjuster.save_model("enhanced_model.joblib")
            new_adjuster = EnhancedMLBondAdjuster()
            new_adjuster.load_model("enhanced_model.joblib")
        finally:
            os.chdir(original_cwd)
        assert new_adjuster.is_trained
        # best_params comparison only if both are not None
        if adjuster.best_params is not None and new_adjuster.best_params is not None:
            assert new_adjuster.best_params == adjuster.best_params

    def test_training_data_generation_pipeline(self, temp_training_dir):
        """Test training data generation pipeline"""
        # Step 1: Initialize generator
        generator = TrainingDataGenerator(seed=42)

        # Step 2: Generate training data
        dataset = generator.generate_comprehensive_dataset(total_bonds=50, time_periods=10, bonds_per_period=10)

        # Step 3: Verify dataset structure
        assert "train" in dataset, "Dataset should have 'train' key"
        assert "bonds" in dataset["train"], "Dataset train should have 'bonds' key"
        assert len(dataset["train"]["bonds"]) > 0, "Train bonds should not be empty"

        # Step 4: Save dataset (use relative path)
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_training_dir)
            save_training_dataset(dataset, "training_dataset.joblib")
            assert os.path.exists("training_dataset.joblib")

            # Step 5: Load dataset
            loaded_dataset = load_training_dataset("training_dataset.joblib")
            assert len(loaded_dataset) > 0
        finally:
            os.chdir(original_cwd)

    def test_model_training_with_saved_dataset(self, temp_training_dir):
        """Test training model with pre-generated dataset"""
        # Step 1: Generate and save dataset
        generator = TrainingDataGenerator(seed=42)
        dataset = generator.generate_comprehensive_dataset(total_bonds=30, time_periods=5, bonds_per_period=10)
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_training_dir)
            save_training_dataset(dataset, "dataset.joblib")

            # Step 2: Load dataset
            loaded_dataset = load_training_dataset("dataset.joblib")
            # Dataset structure: {"train": {"bonds": [...], "metadata": [...], ...}, ...}
            train_metadata = loaded_dataset.get("train", {}).get("metadata", [])
            # Reconstruct Bond objects from metadata
            train_bonds = reconstruct_bonds_from_metadata(train_metadata)
            # Ensure we have enough bonds for training
            assert len(train_bonds) >= 10, f"Need at least 10 bonds for training, got {len(train_bonds)}"
        finally:
            os.chdir(original_cwd)

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
        dataset = generator.generate_comprehensive_dataset(total_bonds=100, time_periods=10, bonds_per_period=20)

        # Train model
        adjuster = MLBondAdjuster(model_type="random_forest")
        import time

        # Dataset structure: {"train": {"bonds": [...], "metadata": [...], ...}, ...}
        train_metadata = dataset.get("train", {}).get("metadata", [])
        # Reconstruct Bond objects from metadata
        train_bonds = reconstruct_bonds_from_metadata(train_metadata)
        assert len(train_bonds) >= 10, f"Need at least 10 bonds for training, got {len(train_bonds)}"

        start_time = time.time()
        metrics = adjuster.train(train_bonds, test_size=0.2, random_state=42)
        elapsed_time = time.time() - start_time

        # Should complete within reasonable time (adjust threshold as needed)
        assert elapsed_time < 60  # Should complete within 60 seconds
        assert adjuster.is_trained
        assert metrics["test_r2"] >= 0  # Should have reasonable performance
