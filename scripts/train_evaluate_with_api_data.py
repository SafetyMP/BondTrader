"""
Train and Evaluate Models Using API Data
Fetches bond data from FRED and FINRA APIs for 2016-2017, 2018, and 2025
Trains on 2016-2017, fine-tunes on 2018, and predicts on 2025
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
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
from bondtrader.core.container import get_container
from bondtrader.data.market_data import FINRADataProvider, FREDDataProvider, MarketDataManager
from bondtrader.ml.automl import AutoMLBondAdjuster
from bondtrader.ml.ml_adjuster import MLBondAdjuster
from bondtrader.ml.ml_adjuster_enhanced import EnhancedMLBondAdjuster
from bondtrader.ml.ml_advanced import AdvancedMLBondAdjuster
from bondtrader.utils.utils import logger


def fetch_bonds_from_api(start_date: datetime, end_date: datetime, data_type: str = "treasury") -> List[Bond]:
    """
    Fetch bond data from APIs (FRED for Treasury, FINRA for corporate)

    Args:
        start_date: Start date for data
        end_date: End date for data
        data_type: 'treasury' or 'corporate'

    Returns:
        List of Bond objects
    """
    manager = MarketDataManager()
    bonds = []

    if data_type == "treasury":
        # Fetch Treasury data from FRED
        print(f"Fetching Treasury data from FRED ({start_date.date()} to {end_date.date()})...")
        treasury_data = manager.fetch_historical_treasury_data(
            start_date=start_date,
            end_date=end_date,
            maturities=["GS1", "GS2", "GS5", "GS10", "GS30"],
        )

        if treasury_data is None or treasury_data.empty:
            print("No Treasury data from FRED, generating synthetic data...")
            return _generate_synthetic_bonds(start_date, end_date, BondType.TREASURY, "US Treasury")

        # Convert Treasury data to Bond objects
        for date, row in treasury_data.iterrows():
            for maturity_col in treasury_data.columns:
                if pd.isna(row[maturity_col]):
                    continue

                yield_rate = row[maturity_col]
                try:
                    maturity_years = int(maturity_col.replace("GS", ""))
                except ValueError:
                    continue

                coupon_rate = yield_rate * 100
                issue_date = date - pd.Timedelta(days=365 * maturity_years // 2)
                maturity_date = date + pd.Timedelta(days=365 * maturity_years)

                bond_id = f"TREASURY-{date.strftime('%Y%m%d')}-{maturity_years}YR"

                try:
                    bond = Bond(
                        bond_id=bond_id,
                        bond_type=BondType.TREASURY,
                        face_value=1000.0,
                        coupon_rate=coupon_rate,
                        maturity_date=maturity_date.to_pydatetime(),
                        issue_date=issue_date.to_pydatetime(),
                        current_price=1000.0,
                        credit_rating="AAA",
                        issuer="US Treasury",
                        frequency=2,
                        callable=False,
                        convertible=False,
                    )
                    bonds.append(bond)
                except Exception as e:
                    logger.warning(f"Error creating bond {bond_id}: {e}")
                    continue

    elif data_type == "corporate":
        # Fetch corporate bond data from FINRA
        print(f"Fetching corporate bond data from FINRA ({start_date.date()} to {end_date.date()})...")
        finra = FINRADataProvider()
        finra_data = finra.fetch_historical_bond_data(start_date=start_date, end_date=end_date)

        if finra_data is None or finra_data.empty:
            print("No FINRA data available, generating synthetic corporate bonds...")
            return _generate_synthetic_bonds(start_date, end_date, BondType.CORPORATE, "Corporate Issuer")

        # Convert FINRA data to Bond objects
        for idx, row in finra_data.iterrows():
            try:
                bond_id = row.get("cusip", f"FINRA-{idx}")
                execution_date = pd.to_datetime(row.get("executionDate", row.get("date", start_date)))
                price = float(row.get("price", row.get("tradePrice", 1000.0)))
                yield_rate = float(row.get("yield", row.get("yieldToMaturity", 0.05)))

                # Estimate maturity from yield or use default
                maturity_years = int(row.get("maturityYears", 10))
                issue_date = execution_date - pd.Timedelta(days=365 * maturity_years // 2)
                maturity_date = execution_date + pd.Timedelta(days=365 * maturity_years)

                coupon_rate = yield_rate * 100
                credit_rating = str(row.get("creditRating", "BBB"))
                issuer = str(row.get("issuer", "Corporate Issuer"))

                bond = Bond(
                    bond_id=bond_id,
                    bond_type=BondType.CORPORATE,
                    face_value=1000.0,
                    coupon_rate=coupon_rate,
                    maturity_date=maturity_date.to_pydatetime(),
                    issue_date=issue_date.to_pydatetime(),
                    current_price=price,
                    credit_rating=credit_rating,
                    issuer=issuer,
                    frequency=2,
                    callable=bool(row.get("callable", False)),
                    convertible=bool(row.get("convertible", False)),
                )
                bonds.append(bond)
            except Exception as e:
                logger.warning(f"Error creating bond from FINRA data {idx}: {e}")
                continue

    print(f"Fetched {len(bonds)} bonds from API")
    return bonds


def _generate_synthetic_bonds(start_date: datetime, end_date: datetime, bond_type: BondType, issuer: str) -> List[Bond]:
    """Generate synthetic bond data when API is unavailable"""
    import random

    print(f"Generating synthetic {bond_type.value} bonds...")
    bonds = []
    current_date = start_date

    # Generate monthly data points
    while current_date <= end_date:
        base_rate = 0.02 + random.uniform(0, 0.03)

        for maturity_years in [1, 2, 5, 10, 30]:
            yield_rate = base_rate + (maturity_years / 30) * 0.01
            coupon_rate = yield_rate * 100

            issue_date = current_date - pd.Timedelta(days=365 * maturity_years // 2)
            maturity_date = current_date + pd.Timedelta(days=365 * maturity_years)

            current_price = 1000.0 * (1 + random.uniform(-0.05, 0.05))
            credit_rating = random.choice(["AAA", "AA", "A", "BBB", "BB"]) if bond_type == BondType.CORPORATE else "AAA"

            bond_id = f"SYNTH-{bond_type.value.upper()}-{current_date.strftime('%Y%m%d')}-{maturity_years}YR"

            try:
                bond = Bond(
                    bond_id=bond_id,
                    bond_type=bond_type,
                    face_value=1000.0,
                    coupon_rate=coupon_rate,
                    maturity_date=maturity_date.to_pydatetime(),
                    issue_date=issue_date.to_pydatetime(),
                    current_price=current_price,
                    credit_rating=credit_rating,
                    issuer=issuer,
                    frequency=2,
                    callable=False,
                    convertible=False,
                )
                bonds.append(bond)
            except Exception as e:
                logger.warning(f"Error creating synthetic bond {bond_id}: {e}")
                continue

        # Move to next month
        if current_date.month == 12:
            current_date = current_date.replace(year=current_date.year + 1, month=1)
        else:
            current_date = current_date.replace(month=current_date.month + 1)

    return bonds


def train_models(bonds: List[Bond], model_dir: str) -> Dict:
    """Train all models on provided bonds"""
    container = get_container()
    config = container.config
    valuator = container.get_valuator()  # Use shared valuator instance

    os.makedirs(model_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("TRAINING MODELS")
    print("=" * 60)
    print(f"Training with {len(bonds)} bonds")

    results = {}

    # Split data
    split_idx = int(len(bonds) * 0.8)
    train_bonds = bonds[:split_idx]
    test_bonds = bonds[split_idx:]

    print(f"Train: {len(train_bonds)} bonds, Test: {len(test_bonds)} bonds\n")

    # 1. Basic ML Adjuster
    print("-" * 60)
    print("Training: Basic ML Adjuster")
    print("-" * 60)
    try:
        # Pass valuator from container (shared instance)
        ml_adjuster = MLBondAdjuster(model_type=config.ml_model_type, valuator=valuator)
        metrics = ml_adjuster.train(train_bonds, test_size=0.2, random_state=config.ml_random_state)
        results["ml_adjuster"] = {"model": ml_adjuster, "metrics": metrics, "status": "success"}
        joblib.dump(ml_adjuster, os.path.join(model_dir, "ml_adjuster.joblib"))
        print(f"✓ Trained. Test R²: {metrics.get('test_r2', 0):.4f}")
    except Exception as e:
        print(f"✗ Failed: {e}")
        results["ml_adjuster"] = {"status": "failed", "error": str(e)}

    # 2. Enhanced ML Adjuster
    print("\n" + "-" * 60)
    print("Training: Enhanced ML Adjuster")
    print("-" * 60)
    try:
        # Pass valuator from container (shared instance)
        enhanced_ml = EnhancedMLBondAdjuster(model_type=config.ml_model_type, valuator=valuator)
        metrics = enhanced_ml.train_with_tuning(
            train_bonds,
            test_size=0.2,
            random_state=config.ml_random_state,
            tune_hyperparameters=True,
        )
        results["enhanced_ml_adjuster"] = {
            "model": enhanced_ml,
            "metrics": metrics,
            "status": "success",
        }
        joblib.dump(enhanced_ml, os.path.join(model_dir, "enhanced_ml_adjuster.joblib"))
        print(f"✓ Trained. Test R²: {metrics.get('test_r2', 0):.4f}")
    except Exception as e:
        print(f"✗ Failed: {e}")
        results["enhanced_ml_adjuster"] = {"status": "failed", "error": str(e)}

    # 3. Advanced ML Adjuster (Ensemble)
    print("\n" + "-" * 60)
    print("Training: Advanced ML Adjuster (Ensemble)")
    print("-" * 60)
    try:
        # Pass valuator from container (shared instance)
        advanced_ml = AdvancedMLBondAdjuster(valuator=valuator)
        ensemble_result = advanced_ml.train_ensemble(train_bonds, test_size=0.2, random_state=config.ml_random_state)
        results["advanced_ml_adjuster"] = {
            "model": advanced_ml,
            "metrics": ensemble_result,
            "status": "success",
        }
        joblib.dump(advanced_ml, os.path.join(model_dir, "advanced_ml_adjuster.joblib"))
        print(f"✓ Trained. Ensemble Test R²: {ensemble_result.get('ensemble_metrics', {}).get('test_r2', 0):.4f}")
    except Exception as e:
        print(f"✗ Failed: {e}")
        results["advanced_ml_adjuster"] = {"status": "failed", "error": str(e)}

    # 4. AutoML
    print("\n" + "-" * 60)
    print("Training: AutoML")
    print("-" * 60)
    try:
        # Pass valuator from container (shared instance)
        automl = AutoMLBondAdjuster(valuator=valuator)
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
        joblib.dump(automl, os.path.join(model_dir, "automl_adjuster.joblib"))
        print(f"✓ Trained. Best model: {automl.best_model_type}")
    except Exception as e:
        print(f"✗ Failed: {e}")
        results["automl"] = {"status": "failed", "error": str(e)}

    return results


def fine_tune_models(models: Dict, fine_tune_bonds: List[Bond], model_dir: str) -> Dict:
    """Fine-tune models on new data"""
    print("\n" + "=" * 60)
    print("FINE-TUNING MODELS")
    print("=" * 60)
    print(f"Fine-tuning with {len(fine_tune_bonds)} bonds\n")

    fine_tuned_results = {}

    for model_name, model_data in models.items():
        if model_data.get("status") != "success" or "model" not in model_data:
            continue

        print("-" * 60)
        print(f"Fine-tuning: {model_name}")
        print("-" * 60)

        try:
            model = model_data["model"]

            # Fine-tune by continuing training on new data
            if hasattr(model, "train_with_tuning"):
                # Enhanced model - use train_with_tuning
                metrics = model.train_with_tuning(fine_tune_bonds, test_size=0.2, random_state=42, tune_hyperparameters=False)
            elif hasattr(model, "train"):
                # Basic model - retrain with combined data
                # Get original training bonds if available, otherwise just use fine-tune data
                metrics = model.train(fine_tune_bonds, test_size=0.2, random_state=42)
            elif hasattr(model, "train_ensemble"):
                # Advanced model - retrain ensemble
                metrics = model.train_ensemble(fine_tune_bonds, test_size=0.2, random_state=42)
            else:
                print(f"  ⚠ Model {model_name} doesn't support fine-tuning")
                continue

            # Save fine-tuned model
            joblib.dump(model, os.path.join(model_dir, f"{model_name}_fine_tuned.joblib"))
            fine_tuned_results[model_name] = {
                "model": model,
                "metrics": metrics,
                "status": "success",
            }
            print(
                f"✓ Fine-tuned. Test R²: {metrics.get('test_r2', metrics.get('ensemble_metrics', {}).get('test_r2', 0)):.4f}"
            )

        except Exception as e:
            print(f"✗ Fine-tuning failed: {e}")
            fine_tuned_results[model_name] = {"status": "failed", "error": str(e)}

    return fine_tuned_results


def make_predictions(models: Dict, prediction_bonds: List[Bond], output_path: str) -> pd.DataFrame:
    """Make predictions on bonds using all models"""
    print("\n" + "=" * 60)
    print("MAKING PREDICTIONS")
    print("=" * 60)
    print(f"Predicting on {len(prediction_bonds)} bonds\n")

    container = get_container()
    bond_service = container.get_bond_service()  # Use service layer for valuations
    predictions = []

    for bond in prediction_bonds:
        bond_pred = {
            "bond_id": bond.bond_id,
            "bond_type": bond.bond_type.value,
            "issuer": bond.issuer,
            "credit_rating": bond.credit_rating,
            "coupon_rate": bond.coupon_rate,
            "current_price": bond.current_price,
            "face_value": bond.face_value,
            "maturity_date": bond.maturity_date.isoformat(),
        }

        # Use service layer for valuation (includes audit logging, metrics)
        valuation_result = bond_service.calculate_valuation_for_bond(bond)
        if valuation_result.is_ok():
            bond_pred["theoretical_fair_value"] = valuation_result.value["fair_value"]
        else:
            bond_pred["theoretical_fair_value"] = None

        # Get predictions from each model
        for model_name, model_data in models.items():
            if model_data.get("status") != "success" or "model" not in model_data:
                continue

            try:
                model = model_data["model"]
                if hasattr(model, "predict_adjusted_value"):
                    pred = model.predict_adjusted_value(bond)
                    bond_pred[f"{model_name}_predicted_value"] = pred.get(
                        "ml_adjusted_fair_value", pred.get("ml_adjusted_value", None)
                    )
                    bond_pred[f"{model_name}_adjustment_factor"] = pred.get("adjustment_factor", None)
                elif hasattr(model, "predict"):
                    # Direct prediction - use service layer
                    valuation_result = bond_service.calculate_valuation_for_bond(bond)
                    if valuation_result.is_ok():
                        bond_pred[f"{model_name}_predicted_value"] = valuation_result.value["fair_value"]
                    else:
                        bond_pred[f"{model_name}_predicted_value"] = None
            except Exception as e:
                logger.warning(f"Error predicting with {model_name} for {bond.bond_id}: {e}")
                bond_pred[f"{model_name}_predicted_value"] = None

        predictions.append(bond_pred)

    df = pd.DataFrame(predictions)

    # Save predictions
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✓ Saved predictions to {output_path}")

    return df


def main():
    """Main execution function"""
    config = get_config()

    # Define date ranges
    train_start = datetime(2016, 1, 1)
    train_end = datetime(2017, 12, 31)
    fine_tune_start = datetime(2018, 1, 1)
    fine_tune_end = datetime(2018, 12, 31)
    predict_start = datetime(2025, 1, 1)
    predict_end = datetime(2025, 12, 31)

    # Create output directories
    model_dir = os.path.join(config.model_dir, "api_trained_models")
    predictions_dir = os.path.join(config.data_dir, "predictions")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)

    print("=" * 60)
    print("BOND MODEL TRAINING AND EVALUATION WITH API DATA")
    print("=" * 60)
    print(f"\nTraining Period: {train_start.date()} to {train_end.date()}")
    print(f"Fine-tuning Period: {fine_tune_start.date()} to {fine_tune_end.date()}")
    print(f"Prediction Period: {predict_start.date()} to {predict_end.date()}\n")

    # Step 1: Fetch training data (2016-2017)
    print("STEP 1: Fetching training data (2016-2017)")
    train_bonds = fetch_bonds_from_api(train_start, train_end, data_type="treasury")
    if not train_bonds:
        print("ERROR: No training bonds fetched!")
        sys.exit(1)

    # Step 2: Train models
    print("\nSTEP 2: Training models on 2016-2017 data")
    trained_models = train_models(train_bonds, model_dir)

    # Step 3: Fetch fine-tuning data (2018)
    print("\nSTEP 3: Fetching fine-tuning data (2018)")
    fine_tune_bonds = fetch_bonds_from_api(fine_tune_start, fine_tune_end, data_type="treasury")
    if not fine_tune_bonds:
        print("WARNING: No fine-tuning bonds fetched, skipping fine-tuning")
        fine_tuned_models = trained_models
    else:
        # Step 4: Fine-tune models
        print("\nSTEP 4: Fine-tuning models on 2018 data")
        fine_tuned_models = fine_tune_models(trained_models, fine_tune_bonds, model_dir)
        # Use fine-tuned models if available, otherwise use original
        for model_name in fine_tuned_models:
            if fine_tuned_models[model_name].get("status") == "success":
                trained_models[model_name] = fine_tuned_models[model_name]

    # Step 5: Fetch prediction data (2025)
    print("\nSTEP 5: Fetching prediction data (2025)")
    prediction_bonds = fetch_bonds_from_api(predict_start, predict_end, data_type="treasury")
    if not prediction_bonds:
        print("ERROR: No prediction bonds fetched!")
        sys.exit(1)

    # Step 6: Make predictions
    print("\nSTEP 6: Making predictions on 2025 data")
    predictions_path = os.path.join(predictions_dir, "2025_predictions.csv")
    predictions_df = make_predictions(trained_models, prediction_bonds, predictions_path)

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Models saved to: {model_dir}")
    print(f"Predictions saved to: {predictions_path}")
    print(f"\nTo view results in Streamlit, run:")
    print(f"  streamlit run scripts/streamlit_predictions_dashboard.py")


if __name__ == "__main__":
    main()
