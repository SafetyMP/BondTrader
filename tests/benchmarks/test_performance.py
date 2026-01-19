"""
Performance benchmarks for critical operations
Tests execution time and memory usage for key functions
"""

import os
import sys
import time
from typing import Dict, List

import pytest

pytestmark = pytest.mark.performance

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bondtrader.core.bond_models import Bond, BondType
from bondtrader.core.bond_valuation import BondValuator
from bondtrader.ml.ml_adjuster import MLBondAdjuster

# Import from fixtures
import sys
from pathlib import Path

fixtures_path = Path(__file__).parent.parent / "fixtures"
sys.path.insert(0, str(fixtures_path))
from bond_factory import create_multiple_bonds


class TestPerformanceBenchmarks:
    """Performance benchmarks for critical operations"""

    @pytest.fixture
    def benchmark_bonds(self):
        """Create bonds for performance testing"""
        return create_multiple_bonds(count=100)

    @pytest.fixture
    def valuator(self):
        """Create valuator for performance testing"""
        return BondValuator()

    def benchmark_bond_valuation(self, benchmark_bonds, valuator):
        """Benchmark bond valuation performance"""
        start_time = time.time()

        fair_values = [valuator.calculate_fair_value(bond) for bond in benchmark_bonds]

        elapsed_time = time.time() - start_time
        avg_time_per_bond = elapsed_time / len(benchmark_bonds)

        assert len(fair_values) == len(benchmark_bonds)
        assert avg_time_per_bond < 0.1  # Should be fast (<100ms per bond)

        return {
            "total_time": elapsed_time,
            "avg_time_per_bond": avg_time_per_bond,
            "bonds_processed": len(benchmark_bonds),
        }

    def benchmark_ytm_calculation(self, benchmark_bonds, valuator):
        """Benchmark YTM calculation performance"""
        start_time = time.time()

        ytms = [valuator.calculate_yield_to_maturity(bond) for bond in benchmark_bonds]

        elapsed_time = time.time() - start_time
        avg_time_per_bond = elapsed_time / len(benchmark_bonds)

        assert len(ytms) == len(benchmark_bonds)
        assert avg_time_per_bond < 0.05  # YTM should be fast (<50ms per bond)

        return {
            "total_time": elapsed_time,
            "avg_time_per_bond": avg_time_per_bond,
            "bonds_processed": len(benchmark_bonds),
        }

    def benchmark_ml_prediction(self, benchmark_bonds):
        """Benchmark ML model prediction performance"""
        # Train model first
        adjuster = MLBondAdjuster(model_type="random_forest")
        adjuster.train(benchmark_bonds[:50], test_size=0.2, random_state=42)

        # Benchmark prediction
        test_bonds = benchmark_bonds[50:]

        start_time = time.time()

        predictions = [adjuster.predict_adjusted_value(bond) for bond in test_bonds]

        elapsed_time = time.time() - start_time
        avg_time_per_prediction = elapsed_time / len(test_bonds)

        assert len(predictions) == len(test_bonds)
        assert avg_time_per_prediction < 0.01  # Predictions should be very fast (<10ms)

        return {
            "total_time": elapsed_time,
            "avg_time_per_prediction": avg_time_per_prediction,
            "predictions_made": len(test_bonds),
        }

    def benchmark_ml_training(self):
        """Benchmark ML model training performance"""
        training_bonds = create_multiple_bonds(count=100)

        start_time = time.time()

        adjuster = MLBondAdjuster(model_type="random_forest")
        metrics = adjuster.train(training_bonds, test_size=0.2, random_state=42)

        elapsed_time = time.time() - start_time

        assert adjuster.is_trained
        assert elapsed_time < 30  # Training should complete within 30 seconds for 100 bonds

        return {
            "total_time": elapsed_time,
            "bonds_trained": len(training_bonds),
            "metrics": metrics,
        }

    def benchmark_bulk_operations(self, benchmark_bonds, valuator):
        """Benchmark bulk valuation operations"""
        start_time = time.time()

        # Multiple operations
        fair_values = [valuator.calculate_fair_value(bond) for bond in benchmark_bonds]
        ytms = [valuator.calculate_yield_to_maturity(bond) for bond in benchmark_bonds]
        durations = [valuator.calculate_duration(bond, ytm) for bond, ytm in zip(benchmark_bonds, ytms)]

        elapsed_time = time.time() - start_time
        avg_time_per_bond = elapsed_time / len(benchmark_bonds)

        assert len(fair_values) == len(benchmark_bonds)
        assert avg_time_per_bond < 0.2  # All operations should complete within 200ms per bond

        return {
            "total_time": elapsed_time,
            "avg_time_per_bond": avg_time_per_bond,
            "bonds_processed": len(benchmark_bonds),
        }


@pytest.mark.performance
class TestPerformanceRegression:
    """Performance regression tests - ensure performance doesn't degrade"""

    def test_valuation_performance_regression(self):
        """Ensure valuation performance doesn't regress"""
        bonds = create_multiple_bonds(count=50)
        valuator = BondValuator()

        start_time = time.time()
        for bond in bonds:
            valuator.calculate_fair_value(bond)
        elapsed_time = time.time() - start_time

        # Should complete 50 valuations in less than 5 seconds
        assert elapsed_time < 5.0

    def test_ml_prediction_performance_regression(self):
        """Ensure ML prediction performance doesn't regress"""
        # Train model
        training_bonds = create_multiple_bonds(count=30)
        adjuster = MLBondAdjuster(model_type="random_forest")
        adjuster.train(training_bonds, test_size=0.2, random_state=42)

        # Test prediction speed
        test_bonds = create_multiple_bonds(count=10)

        start_time = time.time()
        for bond in test_bonds:
            adjuster.predict_adjusted_value(bond)
        elapsed_time = time.time() - start_time

        # Should complete 10 predictions in less than 1 second
        assert elapsed_time < 1.0


@pytest.mark.performance
@pytest.mark.slow
class TestPerformanceScalability:
    """Test performance with larger datasets"""

    def test_valuation_scalability(self):
        """Test valuation performance with larger datasets"""
        large_bonds = create_multiple_bonds(count=500)
        valuator = BondValuator()

        start_time = time.time()
        fair_values = [valuator.calculate_fair_value(bond) for bond in large_bonds]
        elapsed_time = time.time() - start_time

        assert len(fair_values) == len(large_bonds)
        # Should handle 500 bonds in reasonable time (<60 seconds)
        assert elapsed_time < 60.0

    def test_ml_training_scalability(self):
        """Test ML training performance with larger datasets"""
        large_training_bonds = create_multiple_bonds(count=200)

        start_time = time.time()

        adjuster = MLBondAdjuster(model_type="random_forest")
        metrics = adjuster.train(large_training_bonds, test_size=0.2, random_state=42)

        elapsed_time = time.time() - start_time

        assert adjuster.is_trained
        # Should train on 200 bonds in reasonable time (<120 seconds)
        assert elapsed_time < 120.0

        return {
            "total_time": elapsed_time,
            "bonds_trained": len(large_training_bonds),
            "metrics": metrics,
        }
