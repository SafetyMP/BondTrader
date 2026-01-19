"""
Further Tune Models with Additional Training Data
Generates additional dataset and uses it to fine-tune existing models
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from bondtrader.config import get_config
from bondtrader.core.bond_models import Bond
from bondtrader.data.training_data_generator import (
    TrainingDataGenerator,
    load_training_dataset,
    save_training_dataset,
)
from bondtrader.ml.automl import AutoMLBondAdjuster
from bondtrader.ml.ml_adjuster import MLBondAdjuster
from bondtrader.ml.ml_adjuster_enhanced import EnhancedMLBondAdjuster
from bondtrader.ml.ml_advanced import AdvancedMLBondAdjuster
from bondtrader.utils.utils import logger


def generate_additional_training_data(
    total_bonds: int = 5000,
    time_periods: int = 60,
    bonds_per_period: int = 100,
    start_year: int = 2010,
    seed: int = 123,  # Different seed for diversity
) -> dict:
    """
    Generate additional training data from a different time period

    Args:
        total_bonds: Total number of unique bonds
        time_periods: Number of time periods (months)
        bonds_per_period: Bonds observed per period
        start_year: Starting year for data generation
        seed: Random seed (different from original to ensure diversity)

    Returns:
        Training dataset dictionary
    """
    print("=" * 70)
    print(f"GENERATING ADDITIONAL TRAINING DATA STARTING FROM {start_year}")
    print("=" * 70)
    print()

    generator = TrainingDataGenerator(seed=seed)

    # Generate base bond universe
    print(f"Step 1: Generating base bond universe ({total_bonds} bonds)...")
    base_bonds = generator._generate_diverse_bond_universe(total_bonds)
    print(f"✓ Generated {len(base_bonds)} bonds")

    # Generate time series data
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
    print("ADDITIONAL DATASET GENERATION COMPLETE")
    print("=" * 70)
    print(f"Total observations: {dataset['dataset_metadata']['total_observations']}")
    print(f"Train size: {dataset['dataset_metadata']['train_size']}")
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
    """Generate time series data starting from a specific year"""
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
    start_date = datetime(start_year, 1, 1)

    for period in range(time_periods):
        if period > 0:
            transition_probs = regime_transitions.get(current_regime, regime_transitions["normal"])
            current_regime = np.random.choice(list(transition_probs.keys()), p=list(transition_probs.values()))

        regime = generator.regimes[current_regime]
        period_date = start_date + timedelta(days=period * 30)

        sampled_bonds = random.sample(base_bonds, min(bonds_per_period, len(base_bonds)))
        generator.valuator.risk_free_rate = regime.risk_free_rate

        for bond in sampled_bonds:
            base_spread = generator.valuator._get_credit_spread(bond.credit_rating)
            adjusted_spread = base_spread + regime.credit_spread_base
            required_ytm = regime.risk_free_rate + adjusted_spread
            fair_value = generator.valuator.calculate_fair_value(bond, required_yield=required_ytm)

            volatility = 0.02 * regime.volatility_multiplier
            liquidity_noise = np.random.normal(0, volatility / regime.liquidity_factor)
            sentiment_impact = regime.market_sentiment * 0.01

            market_price = fair_value * (1 + liquidity_noise + sentiment_impact + np.random.normal(0, volatility * 0.5))
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


def further_tune_models(
    existing_model_paths: dict,
    additional_bonds: List[Bond],
    model_dir: str = None,
    fine_tune: bool = True,
) -> dict:
    """
    Further tune existing models with additional training data

    Args:
        existing_model_paths: Dictionary with model paths {'ml_adjuster': path, 'enhanced_ml_adjuster': path}
        additional_bonds: Additional bonds for training
        model_dir: Directory to save tuned models
        fine_tune: If True, retrain with combined data; if False, just evaluate on new data

    Returns:
        Dictionary with tuning results
    """
    config = get_config()
    if model_dir is None:
        model_dir = os.path.join(config.model_dir, "models_tuned")
    os.makedirs(model_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("FURTHER TUNING MODELS WITH ADDITIONAL DATA")
    print("=" * 70)
    print(f"Additional training bonds: {len(additional_bonds)}")
    print()

    results = {}

    # Split additional data
    split_idx = int(len(additional_bonds) * 0.8)
    train_bonds = additional_bonds[:split_idx]
    test_bonds = additional_bonds[split_idx:]

    print(f"Training set: {len(train_bonds)} bonds")
    print(f"Test set: {len(test_bonds)} bonds")
    print()

    # 1. Tune Basic ML Adjuster
    if "ml_adjuster" in existing_model_paths and os.path.exists(existing_model_paths["ml_adjuster"]):
        print("-" * 70)
        print("Tuning: Basic ML Adjuster")
        print("-" * 70)
        try:
            # Load existing model
            import joblib

            existing_model = MLBondAdjuster()
            existing_model.load_model(existing_model_paths["ml_adjuster"])
            print(f"✓ Loaded existing model from {existing_model_paths['ml_adjuster']}")

            # Evaluate on new test data first
            from bondtrader.core.container import get_container

            valuator = get_container().get_valuator()
            test_predictions = []
            test_actuals = []

            for bond in test_bonds[:100]:  # Sample for evaluation
                try:
                    fair_value = valuator.calculate_fair_value(bond)
                    pred_result = existing_model.predict_adjusted_value(bond)
                    test_predictions.append(pred_result["adjustment_factor"])
                    test_actuals.append(bond.current_price / fair_value if fair_value > 0 else 1.0)
                except (ValueError, AttributeError, KeyError, TypeError) as e:
                    # Skip bonds that fail prediction - log if verbose
                    continue

            if test_predictions:
                from sklearn.metrics import mean_squared_error, r2_score

                old_r2 = r2_score(test_actuals, test_predictions)
                old_rmse = np.sqrt(mean_squared_error(test_actuals, test_predictions))
                print(f"  Performance on new data (before tuning):")
                print(f"    R²: {old_r2:.4f}")
                print(f"    RMSE: {old_rmse:.4f}")

            # Retrain with combined approach: load original training bonds and combine
            print("\n  Loading original training data...")
            try:
                # Try to load original 2005 dataset
                original_dataset_path = os.path.join(config.data_dir, "training_dataset_2005.joblib")
                if os.path.exists(original_dataset_path):
                    original_dataset = load_training_dataset(original_dataset_path)
                    original_generator = TrainingDataGenerator(seed=42)
                    original_bonds = original_generator.generate_bonds_for_training(
                        num_bonds=min(2000, len(original_dataset["train"]["features"])),
                        include_regimes=["normal", "bull", "bear"],
                    )
                    print(f"  ✓ Loaded {len(original_bonds)} original training bonds")
                    # Combine with new bonds
                    combined_bonds = original_bonds + train_bonds
                    print(f"  ✓ Combined dataset: {len(combined_bonds)} total bonds")
                else:
                    print("  ⚠ Original dataset not found, using only new data")
                    combined_bonds = train_bonds
            except Exception as e:
                print(f"  ⚠ Could not load original data: {e}, using only new data")
                combined_bonds = train_bonds

            print("\n  Retraining with combined data...")
            new_model = MLBondAdjuster(model_type=existing_model.model_type)
            metrics = new_model.train(combined_bonds, test_size=0.2, random_state=42)

            # Evaluate on new test data
            test_predictions_new = []
            test_actuals_new = []
            for bond in test_bonds[:100]:
                try:
                    fair_value = valuator.calculate_fair_value(bond)
                    pred_result = new_model.predict_adjusted_value(bond)
                    test_predictions_new.append(pred_result["adjustment_factor"])
                    test_actuals_new.append(bond.current_price / fair_value if fair_value > 0 else 1.0)
                except (ValueError, AttributeError, KeyError, TypeError) as e:
                    # Skip bonds that fail prediction - log if verbose
                    continue

            if test_predictions_new:
                new_r2 = r2_score(test_actuals_new, test_predictions_new)
                new_rmse = np.sqrt(mean_squared_error(test_actuals_new, test_predictions_new))
                print(f"\n  Performance on new data (after tuning):")
                print(f"    R²: {new_r2:.4f} (improvement: {new_r2 - old_r2:+.4f})")
                print(f"    RMSE: {new_rmse:.4f} (improvement: {old_rmse - new_rmse:+.4f})")

            results["ml_adjuster"] = {
                "model": new_model,
                "metrics": metrics,
                "status": "success",
                "improvement": {
                    "r2": new_r2 - old_r2 if test_predictions_new else 0,
                    "rmse": old_rmse - new_rmse if test_predictions_new else 0,
                },
            }

            # Save tuned model
            model_path = os.path.join(model_dir, "ml_adjuster_tuned.joblib")
            new_model.save_model(model_path)
            print(f"\n✓ Tuned model saved to {model_path}")

        except Exception as e:
            print(f"✗ Failed: {e}")
            logger.error(f"Error tuning basic ML adjuster: {e}", exc_info=True)
            results["ml_adjuster"] = {"status": "failed", "error": str(e)}

    # 2. Tune Enhanced ML Adjuster
    if "enhanced_ml_adjuster" in existing_model_paths and os.path.exists(existing_model_paths["enhanced_ml_adjuster"]):
        print("\n" + "-" * 70)
        print("Tuning: Enhanced ML Adjuster (with hyperparameter tuning)")
        print("-" * 70)
        try:
            # Load existing model
            import joblib

            existing_model = EnhancedMLBondAdjuster()
            existing_model.load_model(existing_model_paths["enhanced_ml_adjuster"])
            print(f"✓ Loaded existing model from {existing_model_paths['enhanced_ml_adjuster']}")

            # Evaluate on new test data first
            from bondtrader.core.container import get_container

            valuator = get_container().get_valuator()
            test_predictions = []
            test_actuals = []

            for bond in test_bonds[:100]:
                try:
                    fair_value = valuator.calculate_fair_value(bond)
                    pred_result = existing_model.predict_adjusted_value(bond)
                    test_predictions.append(pred_result["adjustment_factor"])
                    test_actuals.append(bond.current_price / fair_value if fair_value > 0 else 1.0)
                except (ValueError, AttributeError, KeyError, TypeError) as e:
                    # Skip bonds that fail prediction - log if verbose
                    continue

            if test_predictions:
                from sklearn.metrics import mean_squared_error, r2_score

                old_r2 = r2_score(test_actuals, test_predictions)
                old_rmse = np.sqrt(mean_squared_error(test_actuals, test_predictions))
                print(f"  Performance on new data (before tuning):")
                print(f"    R²: {old_r2:.4f}")
                print(f"    RMSE: {old_rmse:.4f}")

            # Retrain with combined data and hyperparameter tuning
            print("\n  Loading original training data...")
            try:
                # Try to load original 2005 dataset
                original_dataset_path = os.path.join(config.data_dir, "training_dataset_2005.joblib")
                if os.path.exists(original_dataset_path):
                    original_dataset = load_training_dataset(original_dataset_path)
                    original_generator = TrainingDataGenerator(seed=42)
                    original_bonds = original_generator.generate_bonds_for_training(
                        num_bonds=min(2000, len(original_dataset["train"]["features"])),
                        include_regimes=["normal", "bull", "bear"],
                    )
                    print(f"  ✓ Loaded {len(original_bonds)} original training bonds")
                    # Combine with new bonds
                    combined_bonds = original_bonds + train_bonds
                    print(f"  ✓ Combined dataset: {len(combined_bonds)} total bonds")
                else:
                    print("  ⚠ Original dataset not found, using only new data")
                    combined_bonds = train_bonds
            except Exception as e:
                print(f"  ⚠ Could not load original data: {e}, using only new data")
                combined_bonds = train_bonds

            print("\n  Retraining with combined data and hyperparameter tuning...")
            new_model = EnhancedMLBondAdjuster(model_type=existing_model.model_type)
            metrics = new_model.train_with_tuning(combined_bonds, test_size=0.2, random_state=42, tune_hyperparameters=True)

            # Evaluate on new test data
            test_predictions_new = []
            test_actuals_new = []
            for bond in test_bonds[:100]:
                try:
                    fair_value = valuator.calculate_fair_value(bond)
                    pred_result = new_model.predict_adjusted_value(bond)
                    test_predictions_new.append(pred_result["adjustment_factor"])
                    test_actuals_new.append(bond.current_price / fair_value if fair_value > 0 else 1.0)
                except (ValueError, AttributeError, KeyError, TypeError) as e:
                    # Skip bonds that fail prediction - log if verbose
                    continue

            if test_predictions_new:
                new_r2 = r2_score(test_actuals_new, test_predictions_new)
                new_rmse = np.sqrt(mean_squared_error(test_actuals_new, test_predictions_new))
                print(f"\n  Performance on new data (after tuning):")
                print(f"    R²: {new_r2:.4f} (improvement: {new_r2 - old_r2:+.4f})")
                print(f"    RMSE: {new_rmse:.4f} (improvement: {old_rmse - new_rmse:+.4f})")

            results["enhanced_ml_adjuster"] = {
                "model": new_model,
                "metrics": metrics,
                "status": "success",
                "improvement": {
                    "r2": new_r2 - old_r2 if test_predictions_new else 0,
                    "rmse": old_rmse - new_rmse if test_predictions_new else 0,
                },
            }

            # Save tuned model
            model_path = os.path.join(model_dir, "enhanced_ml_adjuster_tuned.joblib")
            new_model.save_model(model_path)
            print(f"\n✓ Tuned model saved to {model_path}")
            print(f"  Test R²: {metrics.get('test_r2', 0):.4f}")
            print(f"  CV R²: {metrics.get('cv_r2_mean', 0):.4f} ± {metrics.get('cv_r2_std', 0):.4f}")

        except Exception as e:
            print(f"✗ Failed: {e}")
            logger.error(f"Error tuning enhanced ML adjuster: {e}", exc_info=True)
            results["enhanced_ml_adjuster"] = {"status": "failed", "error": str(e)}

    # 3. Tune Advanced ML Adjuster (Ensemble)
    if "advanced_ml_adjuster" in existing_model_paths and os.path.exists(existing_model_paths["advanced_ml_adjuster"]):
        print("\n" + "-" * 70)
        print("Tuning: Advanced ML Adjuster (Ensemble Methods)")
        print("-" * 70)
        try:
            # Load existing model
            existing_model = AdvancedMLBondAdjuster(valuator)
            existing_model.load_model(existing_model_paths["advanced_ml_adjuster"])
            print(f"✓ Loaded existing model from {existing_model_paths['advanced_ml_adjuster']}")

            # Evaluate on new test data first
            from bondtrader.core.container import get_container

            valuator = get_container().get_valuator()
            test_predictions = []
            test_actuals = []

            for bond in test_bonds[:100]:
                try:
                    fair_value = valuator.calculate_fair_value(bond)
                    pred_result = existing_model.predict_adjusted_value(bond)
                    test_predictions.append(pred_result["adjustment_factor"])
                    test_actuals.append(bond.current_price / fair_value if fair_value > 0 else 1.0)
                except (ValueError, AttributeError, KeyError, TypeError) as e:
                    # Skip bonds that fail prediction - log if verbose
                    continue

            if test_predictions:
                from sklearn.metrics import mean_squared_error, r2_score

                old_r2 = r2_score(test_actuals, test_predictions)
                old_rmse = np.sqrt(mean_squared_error(test_actuals, test_predictions))
                print(f"  Performance on new data (before tuning):")
                print(f"    R²: {old_r2:.4f}")
                print(f"    RMSE: {old_rmse:.4f}")

            # Retrain with combined data
            print("\n  Loading original training data...")
            try:
                original_dataset_path = os.path.join(config.data_dir, "training_dataset_2005.joblib")
                if os.path.exists(original_dataset_path):
                    original_dataset = load_training_dataset(original_dataset_path)
                    original_generator = TrainingDataGenerator(seed=42)
                    original_bonds = original_generator.generate_bonds_for_training(
                        num_bonds=min(2000, len(original_dataset["train"]["features"])),
                        include_regimes=["normal", "bull", "bear"],
                    )
                    print(f"  ✓ Loaded {len(original_bonds)} original training bonds")
                    combined_bonds = original_bonds + train_bonds
                    print(f"  ✓ Combined dataset: {len(combined_bonds)} total bonds")
                else:
                    combined_bonds = train_bonds
            except Exception as e:
                print(f"  ⚠ Could not load original data: {e}, using only new data")
                combined_bonds = train_bonds

            print("\n  Retraining ensemble with combined data...")
            new_model = AdvancedMLBondAdjuster(valuator)
            ensemble_metrics = new_model.train_ensemble(combined_bonds, test_size=0.2, random_state=42)

            # Evaluate on new test data
            test_predictions_new = []
            test_actuals_new = []
            for bond in test_bonds[:100]:
                try:
                    fair_value = valuator.calculate_fair_value(bond)
                    pred_result = new_model.predict_adjusted_value(bond)
                    test_predictions_new.append(pred_result["adjustment_factor"])
                    test_actuals_new.append(bond.current_price / fair_value if fair_value > 0 else 1.0)
                except (ValueError, AttributeError, KeyError, TypeError) as e:
                    # Skip bonds that fail prediction - log if verbose
                    continue

            if test_predictions_new:
                new_r2 = r2_score(test_actuals_new, test_predictions_new)
                new_rmse = np.sqrt(mean_squared_error(test_actuals_new, test_predictions_new))
                print(f"\n  Performance on new data (after tuning):")
                print(f"    R²: {new_r2:.4f} (improvement: {new_r2 - old_r2:+.4f})")
                print(f"    RMSE: {new_rmse:.4f} (improvement: {old_rmse - new_rmse:+.4f})")

            results["advanced_ml_adjuster"] = {
                "model": new_model,
                "metrics": ensemble_metrics,
                "status": "success",
                "improvement": {
                    "r2": new_r2 - old_r2 if test_predictions_new else 0,
                    "rmse": old_rmse - new_rmse if test_predictions_new else 0,
                },
            }

            # Save tuned model
            model_path = os.path.join(model_dir, "advanced_ml_adjuster_tuned.joblib")
            new_model.save_model(model_path)
            print(f"\n✓ Tuned model saved to {model_path}")
            print(f"  Ensemble Test R²: {ensemble_metrics.get('ensemble_metrics', {}).get('test_r2', 0):.4f}")

        except Exception as e:
            print(f"✗ Failed: {e}")
            logger.error(f"Error tuning advanced ML adjuster: {e}", exc_info=True)
            results["advanced_ml_adjuster"] = {"status": "failed", "error": str(e)}

    # 4. Tune AutoML
    if "automl" in existing_model_paths and os.path.exists(existing_model_paths["automl"]):
        print("\n" + "-" * 70)
        print("Tuning: AutoML (Automated Model Selection)")
        print("-" * 70)
        try:
            # Load existing model
            import joblib

            existing_data = joblib.load(existing_model_paths["automl"])
            existing_model = AutoMLBondAdjuster(valuator)
            if isinstance(existing_data, dict):
                existing_model.best_model = existing_data.get("best_model")
                existing_model.best_model_type = existing_data.get("best_model_type")
                existing_model.scaler = existing_data.get("scaler")
                existing_model.is_trained = existing_data.get("is_trained", False)
            else:
                existing_model = existing_data
            print(f"✓ Loaded existing model from {existing_model_paths['automl']}")

            # Evaluate on new test data first
            from bondtrader.core.container import get_container

            valuator = get_container().get_valuator()
            test_predictions = []
            test_actuals = []

            for bond in test_bonds[:100]:
                try:
                    fair_value = valuator.calculate_fair_value(bond)
                    pred_result = existing_model.predict_adjusted_value(bond)
                    test_predictions.append(pred_result["adjustment_factor"])
                    test_actuals.append(bond.current_price / fair_value if fair_value > 0 else 1.0)
                except (ValueError, AttributeError, KeyError, TypeError) as e:
                    # Skip bonds that fail prediction - log if verbose
                    continue

            if test_predictions:
                from sklearn.metrics import mean_squared_error, r2_score

                old_r2 = r2_score(test_actuals, test_predictions)
                old_rmse = np.sqrt(mean_squared_error(test_actuals, test_predictions))
                print(f"  Performance on new data (before tuning):")
                print(f"    R²: {old_r2:.4f}")
                print(f"    RMSE: {old_rmse:.4f}")

            # Retrain with combined data
            print("\n  Loading original training data...")
            try:
                original_dataset_path = os.path.join(config.data_dir, "training_dataset_2005.joblib")
                if os.path.exists(original_dataset_path):
                    original_dataset = load_training_dataset(original_dataset_path)
                    original_generator = TrainingDataGenerator(seed=42)
                    original_bonds = original_generator.generate_bonds_for_training(
                        num_bonds=min(2000, len(original_dataset["train"]["features"])),
                        include_regimes=["normal", "bull", "bear"],
                    )
                    print(f"  ✓ Loaded {len(original_bonds)} original training bonds")
                    combined_bonds = original_bonds + train_bonds
                    print(f"  ✓ Combined dataset: {len(combined_bonds)} total bonds")
                else:
                    combined_bonds = train_bonds
            except Exception as e:
                print(f"  ⚠ Could not load original data: {e}, using only new data")
                combined_bonds = train_bonds

            print("\n  Retraining AutoML with combined data...")
            new_model = AutoMLBondAdjuster(valuator)
            automl_results = new_model.automated_model_selection(
                combined_bonds,
                candidate_models=["random_forest", "gradient_boosting"],
                max_evaluation_time=300,
            )

            # Evaluate on new test data
            test_predictions_new = []
            test_actuals_new = []
            for bond in test_bonds[:100]:
                try:
                    fair_value = valuator.calculate_fair_value(bond)
                    pred_result = new_model.predict_adjusted_value(bond)
                    test_predictions_new.append(pred_result["adjustment_factor"])
                    test_actuals_new.append(bond.current_price / fair_value if fair_value > 0 else 1.0)
                except (ValueError, AttributeError, KeyError, TypeError) as e:
                    # Skip bonds that fail prediction - log if verbose
                    continue

            if test_predictions_new:
                new_r2 = r2_score(test_actuals_new, test_predictions_new)
                new_rmse = np.sqrt(mean_squared_error(test_actuals_new, test_predictions_new))
                print(f"\n  Performance on new data (after tuning):")
                print(f"    R²: {new_r2:.4f} (improvement: {new_r2 - old_r2:+.4f})")
                print(f"    RMSE: {new_rmse:.4f} (improvement: {old_rmse - new_rmse:+.4f})")

            results["automl"] = {
                "model": new_model,
                "results": automl_results,
                "status": "success",
                "improvement": {
                    "r2": new_r2 - old_r2 if test_predictions_new else 0,
                    "rmse": old_rmse - new_rmse if test_predictions_new else 0,
                },
            }

            # Save tuned model
            model_path = os.path.join(model_dir, "automl_adjuster_tuned.joblib")
            joblib.dump(
                {
                    "best_model": new_model.best_model,
                    "best_model_type": new_model.best_model_type,
                    "best_params": new_model.best_params,
                    "scaler": new_model.scaler,
                    "is_trained": new_model.is_trained,
                },
                model_path,
            )
            print(f"\n✓ Tuned model saved to {model_path}")
            print(f"  Best Model: {automl_results.get('best_model', 'N/A')}")
            print(f"  Best Score: {automl_results.get('best_score', 0):.4f}")

        except Exception as e:
            print(f"✗ Failed: {e}")
            logger.error(f"Error tuning AutoML: {e}", exc_info=True)
            results["automl"] = {"status": "failed", "error": str(e)}

    print("\n" + "=" * 70)
    print("TUNING COMPLETE")
    print("=" * 70)
    print(f"Tuned models saved to: {model_dir}")
    successful = sum(1 for r in results.values() if r.get("status") == "success")
    print(f"Successful: {successful}/{len(results)}")

    return results


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Further tune models with additional training data")
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
        help="Number of time periods in months (default: 60)",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2010,
        help="Starting year for additional data (default: 2010)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Directory to save tuned models (default: models/models_tuned)",
    )
    parser.add_argument(
        "--existing-models-dir",
        type=str,
        default="trained_models/models_2005",
        help="Directory containing existing models (default: trained_models/models_2005)",
    )
    parser.add_argument("--save-dataset", action="store_true", help="Save the generated additional dataset")

    args = parser.parse_args()

    try:
        config = get_config()

        # Generate additional training data
        additional_dataset = generate_additional_training_data(
            total_bonds=args.total_bonds,
            time_periods=args.time_periods,
            bonds_per_period=100,
            start_year=args.start_year,
            seed=123,  # Different seed for diversity
        )

        # Save dataset if requested
        if args.save_dataset:
            dataset_path = os.path.join(config.data_dir, f"training_dataset_{args.start_year}.joblib")
            save_training_dataset(additional_dataset, dataset_path)
            print(f"\n✓ Additional dataset saved to {dataset_path}")

        # Generate bonds for training
        print("\n" + "=" * 70)
        print("PREPARING TRAINING BONDS")
        print("=" * 70)
        generator = TrainingDataGenerator(seed=123)
        training_bonds = generator.generate_bonds_for_training(
            num_bonds=min(3000, len(additional_dataset["train"]["features"])),
            include_regimes=["normal", "bull", "bear", "high_volatility"],
        )
        print(f"✓ Generated {len(training_bonds)} training bonds")

        # Find existing models
        existing_model_paths = {}
        ml_path = os.path.join(args.existing_models_dir, "ml_adjuster_2005.joblib")
        enhanced_path = os.path.join(args.existing_models_dir, "enhanced_ml_adjuster_2005.joblib")
        advanced_path = os.path.join(args.existing_models_dir, "advanced_ml_adjuster_2005.joblib")
        automl_path = os.path.join(args.existing_models_dir, "automl_adjuster_2005.joblib")

        if os.path.exists(ml_path):
            existing_model_paths["ml_adjuster"] = ml_path
        if os.path.exists(enhanced_path):
            existing_model_paths["enhanced_ml_adjuster"] = enhanced_path
        if os.path.exists(advanced_path):
            existing_model_paths["advanced_ml_adjuster"] = advanced_path
        if os.path.exists(automl_path):
            existing_model_paths["automl"] = automl_path

        if not existing_model_paths:
            print(f"\n⚠ Warning: No existing models found in {args.existing_models_dir}")
            print("Will train new models from scratch...")

        # Further tune models
        results = further_tune_models(
            existing_model_paths=existing_model_paths,
            additional_bonds=training_bonds,
            model_dir=args.model_dir,
            fine_tune=True,
        )

        print("\n✓ All operations complete!")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
