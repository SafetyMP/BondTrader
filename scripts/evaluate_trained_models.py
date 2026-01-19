"""
Evaluate Trained Models
Evaluates all trained models on evaluation dataset
"""

import os
import sys
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bondtrader.config import get_config
from bondtrader.core.bond_models import Bond
from bondtrader.core.bond_valuation import BondValuator
from scripts.train_with_historical_data import load_bonds_from_csv


def evaluate_model_on_bonds(model, bonds: List[Bond], model_name: str) -> Dict:
    """
    Evaluate a trained model on a set of bonds

    Args:
        model: Trained model object
        bonds: List of Bond objects for evaluation
        model_name: Name of the model

    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\nEvaluating {model_name}...")

    valuator = BondValuator()
    predictions = []
    actuals = []
    fair_values = []

    for bond in bonds:
        try:
            # Get actual market price
            actual_price = bond.current_price

            # Calculate fair value
            fair_value = valuator.calculate_fair_value(bond)
            fair_values.append(fair_value)

            # Get model prediction
            if hasattr(model, "predict_adjusted_value"):
                pred_result = model.predict_adjusted_value(bond)
                # Extract prediction value
                if isinstance(pred_result, dict):
                    pred_price = (
                        pred_result.get("ml_adjusted_value")
                        or pred_result.get("ml_adjusted_fair_value")
                        or pred_result.get("predicted_value")
                    )
                else:
                    pred_price = pred_result
            elif hasattr(model, "predict"):
                # Direct prediction (if model has predict method)
                pred_price = model.predict([bond])[0] if hasattr(model, "predict") else fair_value
            else:
                pred_price = fair_value

            if pred_price is None or np.isnan(pred_price) or np.isinf(pred_price):
                pred_price = fair_value

            predictions.append(pred_price)
            actuals.append(actual_price)

        except Exception as e:
            # Skip bonds that cause errors
            continue

    if len(predictions) == 0:
        return {"error": "No valid predictions"}

    predictions = np.array(predictions)
    actuals = np.array(actuals)
    fair_values = np.array(fair_values)

    # Calculate metrics
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)

    # Calculate percentage errors
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100

    # Compare to fair value baseline
    baseline_mse = mean_squared_error(actuals, fair_values)
    baseline_rmse = np.sqrt(baseline_mse)
    baseline_r2 = r2_score(actuals, fair_values)

    metrics = {
        "n_samples": len(predictions),
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": mape,
        "baseline_rmse": baseline_rmse,
        "baseline_r2": baseline_r2,
        "improvement_over_baseline": baseline_rmse - rmse,
        "r2_improvement": r2 - baseline_r2,
    }

    print(f"  Samples: {metrics['n_samples']}")
    print(f"  R² Score: {metrics['r2']:.4f} (Baseline: {metrics['baseline_r2']:.4f})")
    print(f"  RMSE: ${metrics['rmse']:.2f} (Baseline: ${metrics['baseline_rmse']:.2f})")
    print(f"  MAE: ${metrics['mae']:.2f}")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    if metrics["improvement_over_baseline"] > 0:
        print(f"  ✓ Improvement over baseline: ${metrics['improvement_over_baseline']:.2f}")

    return metrics


def main():
    """Main evaluation function"""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument(
        "--eval-file",
        type=str,
        default="historical_data/eval_bonds_2010_2020.csv",
        help="Evaluation CSV file (default: historical_data/eval_bonds_2010_2020.csv)",
    )
    # Get config for default values
    config = get_config()

    parser.add_argument(
        "--model-dir", type=str, default=config.model_dir, help=f"Directory with trained models (default: {config.model_dir})"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    # Load evaluation bonds
    print(f"\nLoading evaluation bonds from {args.eval_file}...")
    if not os.path.exists(args.eval_file):
        print(f"ERROR: Evaluation file not found: {args.eval_file}")
        print("Please run: python scripts/fetch_historical_data.py --start-year 2010 --end-year 2020")
        sys.exit(1)

    eval_bonds = load_bonds_from_csv(args.eval_file)
    print(f"Loaded {len(eval_bonds)} evaluation bonds")

    # Load and evaluate each model
    model_files = {
        "Basic ML Adjuster": "ml_adjuster.joblib",
        "Enhanced ML Adjuster": "enhanced_ml_adjuster.joblib",
        "Advanced ML Adjuster": "advanced_ml_adjuster.joblib",
        "AutoML": "automl_adjuster.joblib",
    }

    results = {}

    for model_name, model_file in model_files.items():
        model_path = os.path.join(args.model_dir, model_file)

        if not os.path.exists(model_path):
            print(f"\n⚠ Skipping {model_name}: Model file not found at {model_path}")
            continue

        try:
            print(f"\n{'='*70}")
            print(f"Loading {model_name}...")
            model = joblib.load(model_path)

            # Evaluate model
            metrics = evaluate_model_on_bonds(model, eval_bonds, model_name)
            results[model_name] = metrics

        except Exception as e:
            print(f"✗ Error evaluating {model_name}: {e}")
            import traceback

            traceback.print_exc()
            results[model_name] = {"error": str(e)}

    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    print(f"\n{'Model':<30} {'R² Score':<12} {'RMSE':<12} {'MAE':<12} {'MAPE':<12}")
    print("-" * 70)

    for model_name, metrics in results.items():
        if "error" not in metrics:
            print(
                f"{model_name:<30} {metrics['r2']:<12.4f} ${metrics['rmse']:<11.2f} ${metrics['mae']:<11.2f} {metrics['mape']:<11.2f}%"
            )
        else:
            print(f"{model_name:<30} {'ERROR':<12}")

    # Find best model
    valid_results = {k: v for k, v in results.items() if "error" not in v}
    if valid_results:
        best_model = max(valid_results.keys(), key=lambda k: valid_results[k]["r2"])
        best_metrics = valid_results[best_model]
        print(f"\n✓ Best Model: {best_model}")
        print(f"  R² Score: {best_metrics['r2']:.4f}")
        print(f"  RMSE: ${best_metrics['rmse']:.2f}")
        print(f"  Improvement over baseline: ${best_metrics['improvement_over_baseline']:.2f}")

    print("\n" + "=" * 70)
    print("Evaluation complete!")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
