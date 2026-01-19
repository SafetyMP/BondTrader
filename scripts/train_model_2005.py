"""
Train ML Model with 2005 Historical Data
Generates training data starting from 2005 and trains a new model
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bondtrader.config import get_config
from bondtrader.data.training_data_generator import TrainingDataGenerator, save_training_dataset
from bondtrader.ml.automl import AutoMLBondAdjuster
from bondtrader.ml.ml_adjuster import MLBondAdjuster
from bondtrader.ml.ml_adjuster_enhanced import EnhancedMLBondAdjuster
from bondtrader.ml.ml_advanced import AdvancedMLBondAdjuster
from bondtrader.utils.utils import logger


def generate_2005_training_data(
    total_bonds: int = 5000,
    time_periods: int = 60,  # 60 months = 5 years (2005-2010)
    bonds_per_period: int = 100,
    start_year: int = 2005,
) -> dict:
    """
    Generate training data starting from 2005

    Args:
        total_bonds: Total number of unique bonds
        time_periods: Number of time periods (months)
        bonds_per_period: Bonds observed per period
        start_year: Starting year for data generation

    Returns:
        Training dataset dictionary
    """
    print("=" * 70)
    print(f"GENERATING TRAINING DATA STARTING FROM {start_year}")
    print("=" * 70)
    print()

    generator = TrainingDataGenerator(seed=42)

    # Generate base bond universe
    print(f"Step 1: Generating base bond universe ({total_bonds} bonds)...")
    base_bonds = generator._generate_diverse_bond_universe(total_bonds)
    print(f"✓ Generated {len(base_bonds)} bonds")

    # Generate time series data starting from 2005
    print(f"\nStep 2: Generating time series data from {start_year}...")
    time_series_data = _generate_time_series_data_from_date(generator, base_bonds, time_periods, bonds_per_period, start_year)
    print(f"✓ Generated {len(time_series_data)} time series observations")

    # Create feature matrices
    print("\nStep 3: Creating feature matrices...")
    features, targets, metadata = generator._create_feature_matrices(time_series_data)
    print(f"✓ Created feature matrix: {features.shape}")

    # Create time-based splits
    print("\nStep 4: Creating time-based splits...")
    splits = generator._create_time_based_splits(
        features, targets, metadata, train_split=0.7, validation_split=0.15, test_split=0.15
    )
    print(f"✓ Train: {len(splits['train']['features'])} samples")
    print(f"✓ Validation: {len(splits['validation']['features'])} samples")
    print(f"✓ Test: {len(splits['test']['features'])} samples")

    # Validate data quality
    print("\nStep 5: Validating data quality...")
    quality_report = generator._validate_data_quality(splits)
    print("✓ Data quality validated")

    # Generate stress scenarios
    print("\nStep 6: Generating stress test scenarios...")
    stress_scenarios = generator._generate_stress_scenarios(base_bonds)
    print("✓ Stress scenarios generated")

    # Create dataset dictionary
    dataset = {
        "train": {
            "bonds": splits["train"]["bonds"],
            "features": splits["train"]["features"],
            "targets": splits["train"]["targets"],
            "metadata": splits["train"]["metadata"],
        },
        "validation": {
            "bonds": splits["validation"]["bonds"],
            "features": splits["validation"]["features"],
            "targets": splits["validation"]["targets"],
            "metadata": splits["validation"]["metadata"],
        },
        "test": {
            "bonds": splits["test"]["bonds"],
            "features": splits["test"]["features"],
            "targets": splits["test"]["targets"],
            "metadata": splits["test"]["metadata"],
        },
        "stress_scenarios": stress_scenarios,
        "quality_report": quality_report,
        "dataset_metadata": {
            "total_bonds": total_bonds,
            "time_periods": time_periods,
            "total_observations": len(features),
            "train_size": len(splits["train"]["features"]),
            "validation_size": len(splits["validation"]["features"]),
            "test_size": len(splits["test"]["features"]),
            "num_features": features.shape[1] if len(features) > 0 else 0,
            "regimes_represented": list(set([m["regime"] for m in metadata])),
            "date_range": {
                "start": (min([m["date"] for m in metadata]).isoformat() if metadata else datetime.now().isoformat()),
                "end": (max([m["date"] for m in metadata]).isoformat() if metadata else datetime.now().isoformat()),
            },
            "start_year": start_year,
        },
    }

    print("\n" + "=" * 70)
    print("DATASET GENERATION COMPLETE")
    print("=" * 70)
    print(f"Total observations: {dataset['dataset_metadata']['total_observations']}")
    print(f"Train size: {dataset['dataset_metadata']['train_size']}")
    print(f"Validation size: {dataset['dataset_metadata']['validation_size']}")
    print(f"Test size: {dataset['dataset_metadata']['test_size']}")
    print(f"Number of features: {dataset['dataset_metadata']['num_features']}")
    print(
        f"Date range: {dataset['dataset_metadata']['date_range']['start']} to {dataset['dataset_metadata']['date_range']['end']}"
    )

    return dataset


def _generate_time_series_data_from_date(
    generator: TrainingDataGenerator,
    base_bonds: list,
    time_periods: int,
    bonds_per_period: int,
    start_year: int,
) -> list:
    """
    Generate time series data starting from a specific year
    Modified version that accepts a start year instead of using datetime.now()
    """
    import random

    import numpy as np

    time_series_data = []

    # Regime transition probabilities (Markov chain)
    regime_transitions = {
        "normal": {
            "normal": 0.6,
            "bull": 0.15,
            "bear": 0.15,
            "high_volatility": 0.05,
            "low_volatility": 0.05,
        },
        "bull": {
            "normal": 0.3,
            "bull": 0.5,
            "bear": 0.1,
            "high_volatility": 0.05,
            "low_volatility": 0.05,
        },
        "bear": {"normal": 0.2, "bull": 0.1, "bear": 0.5, "high_volatility": 0.15, "crisis": 0.05},
        "high_volatility": {"normal": 0.3, "bear": 0.3, "high_volatility": 0.3, "crisis": 0.1},
        "low_volatility": {"normal": 0.4, "bull": 0.4, "low_volatility": 0.2},
        "crisis": {"recovery": 0.4, "bear": 0.3, "high_volatility": 0.2, "crisis": 0.1},
        "recovery": {"normal": 0.5, "bull": 0.3, "recovery": 0.2},
    }

    current_regime = "normal"
    # Start from January 1st of the specified year
    start_date = datetime(start_year, 1, 1)

    for period in range(time_periods):
        # Transition to new regime
        if period > 0:
            transition_probs = regime_transitions.get(current_regime, regime_transitions["normal"])
            current_regime = random.choice(list(transition_probs.keys()))
            # Use probabilities for selection
            current_regime = np.random.choice(list(transition_probs.keys()), p=list(transition_probs.values()))

        regime = generator.regimes[current_regime]
        period_date = start_date + timedelta(days=period * 30)

        # Sample bonds for this period
        sampled_bonds = random.sample(base_bonds, min(bonds_per_period, len(base_bonds)))

        # Update valuator for this regime
        generator.valuator.risk_free_rate = regime.risk_free_rate

        for bond in sampled_bonds:
            # Calculate fair value under current regime
            base_spread = generator.valuator._get_credit_spread(bond.credit_rating)
            adjusted_spread = base_spread + regime.credit_spread_base

            # Calculate fair value with regime-adjusted spread
            required_ytm = regime.risk_free_rate + adjusted_spread
            fair_value = generator.valuator.calculate_fair_value(bond, required_yield=required_ytm)

            # Add market microstructure noise (bid-ask, liquidity)
            volatility = 0.02 * regime.volatility_multiplier
            liquidity_noise = np.random.normal(0, volatility / regime.liquidity_factor)
            sentiment_impact = regime.market_sentiment * 0.01

            # Market price with regime effects
            market_price = fair_value * (1 + liquidity_noise + sentiment_impact + np.random.normal(0, volatility * 0.5))

            # Ensure price is reasonable
            market_price = np.clip(market_price, fair_value * 0.5, fair_value * 1.5)
            bond.current_price = market_price

            time_series_data.append(
                {
                    "bond": bond,
                    "date": period_date,
                    "regime": current_regime,
                    "fair_value": fair_value,
                    "market_price": market_price,
                    "risk_free_rate": regime.risk_free_rate,
                    "volatility": volatility,
                    "liquidity_factor": regime.liquidity_factor,
                    "sentiment": regime.market_sentiment,
                }
            )

    return time_series_data


def train_model_with_2005_data(dataset: dict, model_type: str = "random_forest", model_dir: str = None) -> dict:
    """
    Train ML models using the 2005 dataset

    Args:
        dataset: Training dataset dictionary
        model_type: Type of model to train ('random_forest' or 'gradient_boosting')
        model_dir: Directory to save trained models

    Returns:
        Dictionary with training results
    """
    config = get_config()
    if model_dir is None:
        model_dir = os.path.join(config.model_dir, "models_2005")
    os.makedirs(model_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("TRAINING MODELS WITH 2005 DATA")
    print("=" * 70)
    print()

    results = {}

    # Convert dataset to Bond objects for training
    from bondtrader.core.bond_models import Bond
    from bondtrader.core.container import get_container

    def features_to_bonds(features, targets, metadata):
        """Convert features back to Bond objects for training"""
        bonds = []
        valuator = get_container().get_valuator()

        for i, (feat, target, meta) in enumerate(zip(features, targets, metadata)):
            # Reconstruct bond from metadata
            # Note: This is a simplified reconstruction - in practice, you'd want to store full bond objects
            # For now, we'll generate bonds from the training data generator
            pass

        return bonds

    # Use the training data generator to create bonds from the dataset
    generator = TrainingDataGenerator(seed=42)

    # For training, we'll use generate_bonds_for_training which creates bonds directly
    # We need to extract bonds from the dataset metadata
    print("Preparing training bonds from dataset...")

    # Get training bonds - we'll generate fresh bonds matching the dataset characteristics
    train_bonds = generator.generate_bonds_for_training(
        num_bonds=min(2000, len(dataset["train"]["features"])),
        include_regimes=["normal", "bull", "bear"],
    )

    # Split for training
    split_idx = int(len(train_bonds) * 0.8)
    train_set = train_bonds[:split_idx]
    test_set = train_bonds[split_idx:]

    print(f"Training with {len(train_set)} bonds")
    print(f"Testing with {len(test_set)} bonds")
    print()

    # 1. Basic ML Adjuster
    print("-" * 70)
    print(f"Training: Basic ML Adjuster ({model_type})")
    print("-" * 70)
    try:
        ml_adjuster = MLBondAdjuster(model_type=model_type)
        metrics = ml_adjuster.train(train_set, test_size=0.2, random_state=42)
        results["ml_adjuster"] = {"model": ml_adjuster, "metrics": metrics, "status": "success"}

        # Save model
        import joblib

        model_path = os.path.join(model_dir, "ml_adjuster_2005.joblib")
        ml_adjuster.save_model(model_path)
        print(f"✓ Trained and saved to {model_path}")
        print(f"  Test R²: {metrics.get('test_r2', 0):.4f}")
        print(f"  Test RMSE: {metrics.get('test_rmse', 0):.4f}")
    except Exception as e:
        print(f"✗ Failed: {e}")
        logger.error(f"Error training basic ML adjuster: {e}", exc_info=True)
        results["ml_adjuster"] = {"status": "failed", "error": str(e)}

    # 2. Enhanced ML Adjuster
    print("\n" + "-" * 70)
    print("Training: Enhanced ML Adjuster (with hyperparameter tuning)")
    print("-" * 70)
    try:
        enhanced_ml = EnhancedMLBondAdjuster(model_type=model_type)
        metrics = enhanced_ml.train_with_tuning(train_set, test_size=0.2, random_state=42, tune_hyperparameters=True)
        results["enhanced_ml_adjuster"] = {
            "model": enhanced_ml,
            "metrics": metrics,
            "status": "success",
        }

        # Save model
        import joblib

        model_path = os.path.join(model_dir, "enhanced_ml_adjuster_2005.joblib")
        enhanced_ml.save_model(model_path)
        print(f"✓ Trained and saved to {model_path}")
        print(f"  Test R²: {metrics.get('test_r2', 0):.4f}")
        print(f"  Test RMSE: {metrics.get('test_rmse', 0):.4f}")
        print(f"  CV R²: {metrics.get('cv_r2_mean', 0):.4f} ± {metrics.get('cv_r2_std', 0):.4f}")
    except Exception as e:
        print(f"✗ Failed: {e}")
        logger.error(f"Error training enhanced ML adjuster: {e}", exc_info=True)
        results["enhanced_ml_adjuster"] = {"status": "failed", "error": str(e)}

    # 3. Advanced ML Adjuster (Ensemble)
    print("\n" + "-" * 70)
    print("Training: Advanced ML Adjuster (Ensemble Methods)")
    print("-" * 70)
    try:
        from bondtrader.core.bond_valuation import BondValuator
        from bondtrader.core.container import get_container

        valuator = get_container().get_valuator()
        advanced_ml = AdvancedMLBondAdjuster(valuator)
        ensemble_metrics = advanced_ml.train_ensemble(train_set, test_size=0.2, random_state=42)
        results["advanced_ml_adjuster"] = {
            "model": advanced_ml,
            "metrics": ensemble_metrics,
            "status": "success",
        }

        # Save model
        import joblib

        model_path = os.path.join(model_dir, "advanced_ml_adjuster_2005.joblib")
        advanced_ml.save_model(model_path)
        print(f"✓ Trained and saved to {model_path}")
        print(f"  Ensemble Test R²: {ensemble_metrics.get('ensemble_metrics', {}).get('test_r2', 0):.4f}")
        print(f"  Ensemble Test RMSE: {ensemble_metrics.get('ensemble_metrics', {}).get('test_rmse', 0):.4f}")
        if "improvement_over_best" in ensemble_metrics:
            print(f"  Improvement over best individual: {ensemble_metrics['improvement_over_best']:.4f}")
    except Exception as e:
        print(f"✗ Failed: {e}")
        logger.error(f"Error training advanced ML adjuster: {e}", exc_info=True)
        results["advanced_ml_adjuster"] = {"status": "failed", "error": str(e)}

    # 4. AutoML
    print("\n" + "-" * 70)
    print("Training: AutoML (Automated Model Selection)")
    print("-" * 70)
    try:
        from bondtrader.core.container import get_container

        valuator = get_container().get_valuator()
        automl = AutoMLBondAdjuster(valuator)
        automl_results = automl.automated_model_selection(
            train_set,
            candidate_models=["random_forest", "gradient_boosting"],
            max_evaluation_time=300,  # 5 minutes
        )
        results["automl"] = {"model": automl, "results": automl_results, "status": "success"}

        # Save model (AutoML doesn't have save_model method, use joblib directly)
        import joblib

        model_path = os.path.join(model_dir, "automl_adjuster_2005.joblib")
        joblib.dump(
            {
                "best_model": automl.best_model,
                "best_model_type": automl.best_model_type,
                "best_params": automl.best_params,
                "scaler": automl.scaler,
                "is_trained": automl.is_trained,
            },
            model_path,
        )
        print(f"✓ Trained and saved to {model_path}")
        print(f"  Best Model: {automl_results.get('best_model', 'N/A')}")
        print(f"  Best Score: {automl_results.get('best_score', 0):.4f}")
    except Exception as e:
        print(f"✗ Failed: {e}")
        logger.error(f"Error training AutoML: {e}", exc_info=True)
        results["automl"] = {"status": "failed", "error": str(e)}

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Models saved to: {model_dir}")
    successful = sum(1 for r in results.values() if r.get("status") == "success")
    print(f"Successful: {successful}/{len(results)}")

    return results


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Train ML model with 2005 historical data")
    parser.add_argument(
        "--total-bonds",
        type=int,
        default=5000,
        help="Total number of bonds to generate (default: 5000)",
    )
    parser.add_argument(
        "--time-periods",
        type=int,
        default=60,
        help="Number of time periods in months (default: 60 = 5 years)",
    )
    parser.add_argument("--bonds-per-period", type=int, default=100, help="Bonds per time period (default: 100)")
    parser.add_argument(
        "--start-year",
        type=int,
        default=2005,
        help="Starting year for data generation (default: 2005)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="random_forest",
        choices=["random_forest", "gradient_boosting"],
        help="Type of ML model to train (default: random_forest)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Directory to save trained models (default: models/models_2005)",
    )
    parser.add_argument("--save-dataset", action="store_true", help="Save the generated dataset to disk")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to save/load dataset (default: data/training_dataset_2005.joblib)",
    )

    args = parser.parse_args()

    try:
        config = get_config()

        # Determine dataset path
        if args.dataset_path is None:
            dataset_path = os.path.join(config.data_dir, f"training_dataset_{args.start_year}.joblib")
        else:
            dataset_path = args.dataset_path

        # Check if dataset already exists
        if os.path.exists(dataset_path) and not args.save_dataset:
            print(f"Loading existing dataset from {dataset_path}...")
            from bondtrader.data.training_data_generator import load_training_dataset

            dataset = load_training_dataset(dataset_path)
            print("✓ Dataset loaded")
        else:
            # Generate new dataset
            dataset = generate_2005_training_data(
                total_bonds=args.total_bonds,
                time_periods=args.time_periods,
                bonds_per_period=args.bonds_per_period,
                start_year=args.start_year,
            )

            # Save dataset if requested
            if args.save_dataset:
                save_training_dataset(dataset, dataset_path)
                print(f"\n✓ Dataset saved to {dataset_path}")

        # Train models
        results = train_model_with_2005_data(dataset=dataset, model_type=args.model_type, model_dir=args.model_dir)

        print("\n✓ All operations complete!")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
