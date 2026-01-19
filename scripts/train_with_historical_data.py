"""
Train Models with Historical Data
Loads historical bond data from CSV files and trains all models
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env file if it exists
try:
    from dotenv import load_dotenv

    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    # python-dotenv not installed, try to load .env manually
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()

from bondtrader.config import get_config
from bondtrader.core.bond_models import Bond, BondType
from bondtrader.data.training_data_generator import TrainingDataGenerator, save_training_dataset
from bondtrader.ml.automl import AutoMLBondAdjuster
from bondtrader.ml.ml_adjuster import MLBondAdjuster
from bondtrader.ml.ml_adjuster_enhanced import EnhancedMLBondAdjuster
from bondtrader.ml.ml_advanced import AdvancedMLBondAdjuster
from bondtrader.utils.utils import logger


def load_bonds_from_csv(csv_path: str) -> List[Bond]:
    """
    Load Bond objects from CSV file created by fetch_historical_data.py

    Args:
        csv_path: Path to CSV file with bond data

    Returns:
        List of Bond objects
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    print(f"Loading bonds from {csv_path}...")
    df = pd.read_csv(csv_path)

    bonds = []
    for idx, row in df.iterrows():
        try:
            # Parse dates
            maturity_date = pd.to_datetime(row["maturity_date"]).to_pydatetime()
            issue_date = pd.to_datetime(row["issue_date"]).to_pydatetime()

            # Parse bond type
            bond_type_str = row.get("bond_type", "Fixed Rate")
            from bondtrader.utils.constants import BOND_TYPE_STRING_MAP

            bond_type_name = BOND_TYPE_STRING_MAP.get(bond_type_str, "FIXED_RATE")
            bond_type = getattr(BondType, bond_type_name, BondType.FIXED_RATE)

            bond = Bond(
                bond_id=row["bond_id"],
                bond_type=bond_type,
                face_value=float(row["face_value"]),
                coupon_rate=float(row["coupon_rate"]),
                maturity_date=maturity_date,
                issue_date=issue_date,
                current_price=float(row["current_price"]),
                credit_rating=str(row.get("credit_rating", "BBB")),
                issuer=str(row.get("issuer", "Unknown")),
                frequency=int(row.get("frequency", 2)),
                callable=bool(row.get("callable", False)),
                convertible=bool(row.get("convertible", False)),
            )
            bonds.append(bond)
        except Exception as e:
            logger.warning(f"Error loading bond {idx} from CSV: {e}")
            continue

    print(f"Loaded {len(bonds)} bonds from CSV")
    return bonds


def create_training_dataset_from_bonds(
    train_bonds: List[Bond],
    validation_bonds: List[Bond],
    test_bonds: List[Bond],
    output_path: str = None,
) -> dict:
    """
    Create training dataset format from Bond objects

    Args:
        train_bonds: Training bond objects
        validation_bonds: Validation bond objects
        test_bonds: Test bond objects
        output_path: Path to save the dataset (defaults to config.data_dir/historical_training_dataset.joblib)

    Returns:
        Dataset dictionary in the format expected by training scripts
    """
    # Get configuration
    config = get_config()
    if output_path is None:
        output_path = os.path.join(config.data_dir, "historical_training_dataset.joblib")

    print("Creating training dataset from bonds...")

    generator = TrainingDataGenerator(seed=config.ml_random_state)

    # Convert bonds to feature matrices
    def bonds_to_features(bonds: List[Bond]) -> tuple:
        """Convert bonds to features, targets, and metadata"""
        features = []
        targets = []
        metadata = []

        for bond in bonds:
            try:
                char = bond.get_bond_characteristics()

                # Calculate YTM, duration, convexity
                from bondtrader.core.container import get_container

                valuator = get_container().get_valuator()
                ytm = valuator.calculate_yield_to_maturity(bond)
                duration = valuator.calculate_duration(bond, ytm)
                convexity = valuator.calculate_convexity(bond, ytm)

                # Calculate fair value
                fair_value = valuator.calculate_fair_value(bond)

                # Feature vector (matching training_data_generator format)
                feature_vector = [
                    char["coupon_rate"],
                    char["time_to_maturity"],
                    char["credit_rating_numeric"],
                    char["current_price"] / char["face_value"],  # Price to par
                    char["years_since_issue"],
                    char["frequency"],
                    char["callable"],
                    char["convertible"],
                    ytm * 100,
                    duration,
                    convexity,
                    bond.current_price / fair_value if fair_value > 0 else 1.0,
                    bond.face_value,
                    duration / (1 + ytm) if ytm > 0 else duration,
                    ytm - valuator.risk_free_rate,
                    valuator.risk_free_rate * 100,
                    0.02 * 100,  # Volatility (simplified)
                    1.0,  # Liquidity factor
                    0.0,  # Sentiment
                    datetime.now().month / 12.0,
                    datetime.now().year - 2020,
                    # Regime indicators (simplified - all normal)
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    # Polynomial features
                    char["coupon_rate"] ** 2,
                    char["time_to_maturity"] ** 2,
                    duration**2,
                    char["coupon_rate"] * char["time_to_maturity"],
                    char["coupon_rate"] * duration,
                    char["time_to_maturity"] * duration,
                ]

                features.append(feature_vector)

                # Target: adjustment factor (market_price / fair_value)
                target = bond.current_price / fair_value if fair_value > 0 else 1.0
                targets.append(target)

                metadata.append(
                    {
                        "bond_id": bond.bond_id,
                        "date": datetime.now(),
                        "regime": "normal",
                        "credit_rating": bond.credit_rating,
                        "bond_type": bond.bond_type.value,
                        "issuer": bond.issuer,
                        "coupon_rate": bond.coupon_rate,
                        "face_value": bond.face_value,
                        "maturity_date": bond.maturity_date,
                        "issue_date": bond.issue_date,
                        "frequency": bond.frequency,
                        "callable": bond.callable,
                        "convertible": bond.convertible,
                    }
                )
            except Exception as e:
                logger.warning(f"Error processing bond {bond.bond_id}: {e}")
                continue

        import numpy as np

        return np.array(features), np.array(targets), metadata

    train_features, train_targets, train_metadata = bonds_to_features(train_bonds)
    val_features, val_targets, val_metadata = bonds_to_features(validation_bonds)
    test_features, test_targets, test_metadata = bonds_to_features(test_bonds)

    dataset = {
        "train": {
            "bonds": [b.bond_id for b in train_bonds],
            "features": train_features,
            "targets": train_targets,
            "metadata": train_metadata,
        },
        "validation": {
            "bonds": [b.bond_id for b in validation_bonds],
            "features": val_features,
            "targets": val_targets,
            "metadata": val_metadata,
        },
        "test": {
            "bonds": [b.bond_id for b in test_bonds],
            "features": test_features,
            "targets": test_targets,
            "metadata": test_metadata,
        },
        "stress_scenarios": {},  # Empty for now
        "quality_report": {
            "train": {
                "n_samples": len(train_features),
                "n_features": train_features.shape[1] if len(train_features) > 0 else 0,
                "missing_values": 0,
                "infinite_values": 0,
                "target_range": (
                    (train_targets.min(), train_targets.max()) if len(train_targets) > 0 else (0, 1)
                ),
                "target_mean": train_targets.mean() if len(train_targets) > 0 else 0,
            },
            "validation": {
                "n_samples": len(val_features),
                "n_features": val_features.shape[1] if len(val_features) > 0 else 0,
                "missing_values": 0,
                "infinite_values": 0,
                "target_range": (
                    (val_targets.min(), val_targets.max()) if len(val_targets) > 0 else (0, 1)
                ),
                "target_mean": val_targets.mean() if len(val_targets) > 0 else 0,
            },
            "test": {
                "n_samples": len(test_features),
                "n_features": test_features.shape[1] if len(test_features) > 0 else 0,
                "missing_values": 0,
                "infinite_values": 0,
                "target_range": (
                    (test_targets.min(), test_targets.max()) if len(test_targets) > 0 else (0, 1)
                ),
                "target_mean": test_targets.mean() if len(test_targets) > 0 else 0,
            },
        },
        "dataset_metadata": {
            "total_bonds": len(train_bonds) + len(validation_bonds) + len(test_bonds),
            "time_periods": 1,
            "total_observations": len(train_features) + len(val_features) + len(test_features),
            "train_size": len(train_features),
            "validation_size": len(val_features),
            "test_size": len(test_features),
            "num_features": train_features.shape[1] if len(train_features) > 0 else 0,
            "regimes_represented": ["normal"],
            "date_range": {"start": datetime.now().isoformat(), "end": datetime.now().isoformat()},
        },
    }

    # Save dataset
    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True
    )
    save_training_dataset(dataset, output_path)
    print(f"Saved training dataset to {output_path}")

    return dataset


def train_models_with_bonds(bonds: List[Bond], model_dir: str = None):
    """
    Train all models using bond data

    Args:
        bonds: List of Bond objects for training
        model_dir: Directory to save trained models (defaults to config.model_dir)
    """
    # Get configuration
    config = get_config()
    if model_dir is None:
        model_dir = config.model_dir
    os.makedirs(model_dir, exist_ok=True)

    print("=" * 60)
    print("TRAINING MODELS WITH HISTORICAL DATA")
    print("=" * 60)
    print(f"Training with {len(bonds)} bonds")
    print()

    results = {}

    # Split data
    split_idx = int(len(bonds) * 0.8)
    train_bonds = bonds[:split_idx]
    test_bonds = bonds[split_idx:]

    print(f"Train: {len(train_bonds)} bonds")
    print(f"Test: {len(test_bonds)} bonds")
    print()

    # Get configuration for ML settings
    config = get_config()

    # 1. Basic ML Adjuster
    print("-" * 60)
    print(f"Training: Basic ML Adjuster ({config.ml_model_type})")
    print("-" * 60)
    try:
        ml_adjuster = MLBondAdjuster(model_type=config.ml_model_type)
        metrics = ml_adjuster.train(
            train_bonds, test_size=config.ml_test_size, random_state=config.ml_random_state
        )
        results["ml_adjuster"] = {"model": ml_adjuster, "metrics": metrics, "status": "success"}

        # Save model
        import joblib

        joblib.dump(ml_adjuster, os.path.join(model_dir, "ml_adjuster.joblib"))
        print(f"✓ Trained and saved. Test R²: {metrics.get('test_r2', 0):.4f}")
    except Exception as e:
        print(f"✗ Failed: {e}")
        results["ml_adjuster"] = {"status": "failed", "error": str(e)}

    # 2. Enhanced ML Adjuster
    print("\n" + "-" * 60)
    print("Training: Enhanced ML Adjuster (with hyperparameter tuning)")
    print("-" * 60)
    try:
        enhanced_ml = EnhancedMLBondAdjuster(model_type=config.ml_model_type)
        metrics = enhanced_ml.train_with_tuning(
            train_bonds,
            test_size=config.ml_test_size,
            random_state=config.ml_random_state,
            tune_hyperparameters=True,
        )
        results["enhanced_ml_adjuster"] = {
            "model": enhanced_ml,
            "metrics": metrics,
            "status": "success",
        }

        # Save model
        import joblib

        joblib.dump(enhanced_ml, os.path.join(model_dir, "enhanced_ml_adjuster.joblib"))
        print(f"✓ Trained and saved. Test R²: {metrics.get('test_r2', 0):.4f}")
    except Exception as e:
        print(f"✗ Failed: {e}")
        results["enhanced_ml_adjuster"] = {"status": "failed", "error": str(e)}

    # 3. Advanced ML Adjuster (Ensemble)
    print("\n" + "-" * 60)
    print("Training: Advanced ML Adjuster (Ensemble)")
    print("-" * 60)
    try:
        advanced_ml = AdvancedMLBondAdjuster()
        ensemble_result = advanced_ml.train_ensemble(
            train_bonds, test_size=config.ml_test_size, random_state=config.ml_random_state
        )
        results["advanced_ml_adjuster"] = {
            "model": advanced_ml,
            "metrics": ensemble_result,
            "status": "success",
        }

        # Save model
        import joblib

        joblib.dump(advanced_ml, os.path.join(model_dir, "advanced_ml_adjuster.joblib"))
        print(
            f"✓ Trained and saved. Ensemble Test R²: {ensemble_result.get('ensemble_metrics', {}).get('test_r2', 0):.4f}"
        )
    except Exception as e:
        print(f"✗ Failed: {e}")
        results["advanced_ml_adjuster"] = {"status": "failed", "error": str(e)}

    # 4. AutoML
    print("\n" + "-" * 60)
    print("Training: AutoML (Automated Model Selection)")
    print("-" * 60)
    try:
        automl = AutoMLBondAdjuster()
        automl.automated_model_selection(
            train_bonds,
            candidate_models=["random_forest", "gradient_boosting"],
            max_evaluation_time=300,
        )
        results["automl"] = {
            "model": automl,
            "best_model_type": automl.best_model_type,
            "status": "success",
        }

        # Save model
        import joblib

        joblib.dump(automl, os.path.join(model_dir, "automl_adjuster.joblib"))
        print(f"✓ Trained and saved. Best model: {automl.best_model_type}")
    except Exception as e:
        print(f"✗ Failed: {e}")
        results["automl"] = {"status": "failed", "error": str(e)}

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Models saved to: {model_dir}")
    print(
        f"Successful: {sum(1 for r in results.values() if r.get('status') == 'success')}/{len(results)}"
    )

    return results


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Train models with historical bond data")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="historical_data",
        help="Directory containing historical data CSV files (default: historical_data)",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default=None,
        help="Specific training CSV file (default: train_bonds_2010_2020.csv)",
    )
    parser.add_argument(
        "--eval-file",
        type=str,
        default=None,
        help="Specific evaluation CSV file (default: eval_bonds_2010_2020.csv)",
    )
    parser.add_argument(
        "--all-file",
        type=str,
        default=None,
        help="Use all bonds from a single CSV file (default: all_bonds_2010_2020.csv)",
    )
    # Get config for default values
    config = get_config()

    parser.add_argument(
        "--model-dir",
        type=str,
        default=config.model_dir,
        help=f"Directory to save trained models (default: {config.model_dir})",
    )
    parser.add_argument(
        "--create-dataset",
        action="store_true",
        help="Create training dataset file for use with train_all_models.py",
    )

    args = parser.parse_args()

    try:
        # Determine which files to use
        if args.all_file:
            csv_path = (
                os.path.join(args.data_dir, args.all_file)
                if not os.path.isabs(args.all_file)
                else args.all_file
            )
            print(f"Loading all bonds from: {csv_path}")
            all_bonds = load_bonds_from_csv(csv_path)

            # Split into train/eval
            split_idx = int(len(all_bonds) * 0.7)
            train_bonds = all_bonds[:split_idx]
            eval_bonds = all_bonds[split_idx:]
        else:
            train_file = args.train_file or "train_bonds_2010_2020.csv"
            eval_file = args.eval_file or "eval_bonds_2010_2020.csv"

            train_path = (
                os.path.join(args.data_dir, train_file)
                if not os.path.isabs(train_file)
                else train_file
            )
            eval_path = (
                os.path.join(args.data_dir, eval_file)
                if not os.path.isabs(eval_file)
                else eval_file
            )

            print(f"Loading training bonds from: {train_path}")
            train_bonds = load_bonds_from_csv(train_path)

            print(f"Loading evaluation bonds from: {eval_path}")
            eval_bonds = load_bonds_from_csv(eval_path) if os.path.exists(eval_path) else []

        if not train_bonds:
            print("ERROR: No training bonds loaded!")
            sys.exit(1)

        print(f"\nLoaded {len(train_bonds)} training bonds")
        if eval_bonds:
            print(f"Loaded {len(eval_bonds)} evaluation bonds")

        # Get config for default paths
        config = get_config()

        # Create dataset if requested
        if args.create_dataset:
            default_dataset_path = os.path.join(
                config.data_dir, "historical_training_dataset.joblib"
            )
            dataset = create_training_dataset_from_bonds(
                train_bonds=train_bonds,
                validation_bonds=(
                    eval_bonds[: len(eval_bonds) // 2]
                    if eval_bonds
                    else train_bonds[: len(train_bonds) // 10]
                ),
                test_bonds=(
                    eval_bonds[len(eval_bonds) // 2 :]
                    if eval_bonds
                    else train_bonds[len(train_bonds) // 10 : len(train_bonds) // 5]
                ),
                output_path=default_dataset_path,
            )
            print("\nDataset created! You can now use train_all_models.py with:")
            print(f"  python scripts/train_all_models.py --dataset-path {default_dataset_path}")

        # Train models
        all_train_bonds = train_bonds + eval_bonds  # Use all for training
        results = train_models_with_bonds(all_train_bonds, model_dir=args.model_dir)

        print("\n✓ Training complete!")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
