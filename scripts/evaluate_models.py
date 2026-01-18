"""
Model Evaluation Script
Comprehensive evaluation of all trained models using evaluation dataset

Follows financial industry best practices for model validation
"""

import os
import sys
from datetime import datetime
from typing import Dict

from train_all_models import ModelTrainer

from bondtrader.data.evaluation_dataset_generator import (
    EvaluationDatasetGenerator,
    load_evaluation_dataset,
    save_evaluation_dataset,
)


def evaluate_all_models(
    model_results_path: str = None,
    evaluation_dataset_path: str = None,
    generate_new_evaluation: bool = False,
    save_results: bool = True,
) -> Dict:
    """
    Evaluate all trained models on evaluation dataset

    Args:
        model_results_path: Path to trained model results (if None, trains new models)
        evaluation_dataset_path: Path to evaluation dataset (if None, generates new)
        generate_new_evaluation: Force generation of new evaluation dataset
        save_results: Whether to save evaluation results

    Returns:
        Dictionary with evaluation results for all models
    """
    print("=" * 70)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("Following Financial Industry Best Practices")
    print("=" * 70)

    # Step 1: Load or generate evaluation dataset
    if generate_new_evaluation or evaluation_dataset_path is None or not os.path.exists(evaluation_dataset_path):
        print("\n[Step 1/4] Generating evaluation dataset...")
        eval_generator = EvaluationDatasetGenerator(seed=42)
        evaluation_dataset = eval_generator.generate_evaluation_dataset(
            num_bonds=2000, scenarios=None, include_benchmarks=True, point_in_time=True  # All scenarios
        )

        os.makedirs("evaluation_data", exist_ok=True)
        save_evaluation_dataset(evaluation_dataset, "evaluation_data/evaluation_dataset.joblib")
        evaluation_dataset_path = "evaluation_data/evaluation_dataset.joblib"
    else:
        print(f"\n[Step 1/4] Loading evaluation dataset from {evaluation_dataset_path}...")
        evaluation_dataset = load_evaluation_dataset(evaluation_dataset_path)

    # Step 2: Load trained models
    print("\n[Step 2/4] Loading trained models...")
    if model_results_path is None or not os.path.exists(model_results_path):
        print("  No trained models found. Training models first...")
        trainer = ModelTrainer(dataset_path="training_data/training_dataset.joblib", generate_new=False)
        model_results = trainer.train_all_models()
        trainer.save_models(model_results)
    else:
        # Load model results (simplified - in practice, load from trained_models/)
        print("  Models should be loaded from trained_models/ directory")
        # For this example, we'll assume models are trained fresh
        trainer = ModelTrainer(dataset_path="training_data/training_dataset.joblib", generate_new=False)
        model_results = trainer.train_all_models()

    # Step 3: Evaluate models
    print("\n[Step 3/4] Evaluating models on evaluation dataset...")
    eval_generator = EvaluationDatasetGenerator()

    all_evaluation_results = {}

    # Evaluate each ML model
    ml_model_names = ["ml_adjuster", "enhanced_ml_adjuster", "advanced_ml_adjuster", "automl"]

    for model_name in ml_model_names:
        if model_name not in model_results:
            print(f"  Skipping {model_name} (not found)")
            continue

        model_data = model_results[model_name]
        if model_data.get("status") != "success" or "model" not in model_data:
            print(f"  Skipping {model_name} (not successfully trained)")
            continue

        print(f"\n  Evaluating {model_name}...")
        try:
            model = model_data["model"]
            eval_results = eval_generator.evaluate_model(
                model=model, evaluation_dataset=evaluation_dataset, scenario_name=None  # All scenarios
            )
            all_evaluation_results[model_name] = eval_results
            print(f"    ✓ Evaluation complete")
        except Exception as e:
            print(f"    ✗ Evaluation failed: {e}")
            all_evaluation_results[model_name] = {"error": str(e)}

    # Step 4: Generate evaluation report
    print("\n[Step 4/4] Generating evaluation report...")
    evaluation_report = generate_evaluation_report(all_evaluation_results, evaluation_dataset)

    if save_results:
        print("\nSaving evaluation results...")
        import joblib

        os.makedirs("evaluation_results", exist_ok=True)
        results_path = f'evaluation_results/evaluation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.joblib'
        joblib.dump(
            {
                "evaluation_results": all_evaluation_results,
                "evaluation_report": evaluation_report,
                "evaluation_dataset_metadata": evaluation_dataset["metadata"],
                "timestamp": datetime.now().isoformat(),
            },
            results_path,
        )
        print(f"  ✓ Results saved to {results_path}")

    # Print summary
    print_evaluation_summary(evaluation_report)

    return {"evaluation_results": all_evaluation_results, "evaluation_report": evaluation_report}


def generate_evaluation_report(evaluation_results: Dict, evaluation_dataset: Dict) -> Dict:
    """Generate comprehensive evaluation report"""
    report = {
        "summary": {},
        "scenario_performance": {},
        "benchmark_comparison": {},
        "best_performing_models": {},
        "warnings": [],
    }

    # Summary statistics
    model_names = list(evaluation_results.keys())
    report["summary"]["models_evaluated"] = len(model_names)
    report["summary"]["scenarios"] = list(evaluation_dataset["scenarios"].keys())

    # Performance by scenario
    scenarios = list(evaluation_dataset["scenarios"].keys())
    for scenario in scenarios:
        if scenario == "benchmarks":
            continue

        scenario_perf = {}
        for model_name, model_results in evaluation_results.items():
            if "error" in model_results:
                continue

            if scenario in model_results:
                metrics = model_results[scenario]
                scenario_perf[model_name] = {
                    "r2_score": metrics.r2_score,
                    "rmse": metrics.root_mean_squared_error,
                    "mae": metrics.mean_absolute_error,
                    "drift_score": metrics.drift_vs_consensus.drift_score,
                    "correlation": metrics.drift_vs_consensus.correlation,
                }

        report["scenario_performance"][scenario] = scenario_perf

    # Benchmark comparison
    benchmark_comparison = {}
    for model_name, model_results in evaluation_results.items():
        if "error" in model_results:
            continue

        # Aggregate across scenarios (use normal_market as primary)
        if "normal_market" in model_results:
            metrics = model_results["normal_market"]
            benchmark_comparison[model_name] = {
                "vs_bloomberg": {
                    "drift": metrics.drift_vs_bloomberg.drift_score,
                    "correlation": metrics.drift_vs_bloomberg.correlation,
                },
                "vs_aladdin": {
                    "drift": metrics.drift_vs_aladdin.drift_score,
                    "correlation": metrics.drift_vs_aladdin.correlation,
                },
                "vs_goldman": {
                    "drift": metrics.drift_vs_goldman.drift_score,
                    "correlation": metrics.drift_vs_goldman.correlation,
                },
                "vs_jpmorgan": {
                    "drift": metrics.drift_vs_jpmorgan.drift_score,
                    "correlation": metrics.drift_vs_jpmorgan.correlation,
                },
                "vs_consensus": {
                    "drift": metrics.drift_vs_consensus.drift_score,
                    "correlation": metrics.drift_vs_consensus.correlation,
                },
            }

    report["benchmark_comparison"] = benchmark_comparison

    # Identify best performing models
    if "normal_market" in report["scenario_performance"]:
        normal_perf = report["scenario_performance"]["normal_market"]

        # Best R²
        best_r2 = max(normal_perf.items(), key=lambda x: x[1]["r2_score"])
        report["best_performing_models"]["highest_r2"] = {"model": best_r2[0], "r2_score": best_r2[1]["r2_score"]}

        # Lowest drift
        best_drift = min(normal_perf.items(), key=lambda x: x[1]["drift_score"])
        report["best_performing_models"]["lowest_drift"] = {
            "model": best_drift[0],
            "drift_score": best_drift[1]["drift_score"],
        }

        # Highest correlation
        best_corr = max(normal_perf.items(), key=lambda x: x[1]["correlation"])
        report["best_performing_models"]["highest_correlation"] = {
            "model": best_corr[0],
            "correlation": best_corr[1]["correlation"],
        }

    # Generate warnings
    for model_name, model_results in evaluation_results.items():
        if "error" in model_results:
            report["warnings"].append(f"{model_name}: Evaluation failed - {model_results['error']}")
            continue

        # Check for poor performance
        if "normal_market" in model_results:
            metrics = model_results["normal_market"]
            if metrics.r2_score < 0.70:
                report["warnings"].append(f"{model_name}: Low R² score ({metrics.r2_score:.4f})")
            if metrics.drift_vs_consensus.drift_score > 0.10:
                report["warnings"].append(f"{model_name}: High drift score ({metrics.drift_vs_consensus.drift_score:.4f})")
            if metrics.drift_vs_consensus.correlation < 0.85:
                report["warnings"].append(
                    f"{model_name}: Low benchmark correlation ({metrics.drift_vs_consensus.correlation:.4f})"
                )

    return report


def print_evaluation_summary(evaluation_report: Dict):
    """Print human-readable evaluation summary"""
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    print(f"\nModels Evaluated: {evaluation_report['summary']['models_evaluated']}")

    # Best performing models
    if "best_performing_models" in evaluation_report:
        print("\nBest Performing Models:")
        best = evaluation_report["best_performing_models"]
        if "highest_r2" in best:
            print(f"  Highest R²: {best['highest_r2']['model']} ({best['highest_r2']['r2_score']:.4f})")
        if "lowest_drift" in best:
            print(f"  Lowest Drift: {best['lowest_drift']['model']} ({best['lowest_drift']['drift_score']:.4f})")
        if "highest_correlation" in best:
            print(
                f"  Highest Correlation: {best['highest_correlation']['model']} ({best['highest_correlation']['correlation']:.4f})"
            )

    # Scenario performance
    if "scenario_performance" in evaluation_report:
        print("\nPerformance by Scenario:")
        for scenario, perf in evaluation_report["scenario_performance"].items():
            print(f"\n  {scenario}:")
            for model_name, metrics in perf.items():
                print(f"    {model_name}:")
                print(f"      R²: {metrics['r2_score']:.4f}")
                print(f"      RMSE: {metrics['rmse']:.2f}")
                print(f"      Drift: {metrics['drift_score']:.4f}")

    # Warnings
    if evaluation_report.get("warnings"):
        print("\n⚠️  Warnings:")
        for warning in evaluation_report["warnings"]:
            print(f"  - {warning}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Run comprehensive evaluation
    results = evaluate_all_models(
        generate_new_evaluation=False, save_results=True  # Set to True to regenerate evaluation dataset
    )

    print("\n✓ Model evaluation complete!")
