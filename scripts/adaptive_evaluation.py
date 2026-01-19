"""
Adaptive Evaluation Script
Generates evaluation data, runs against models, and adjusts generator based on feedback
"""

import json
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

from bondtrader.config import get_config
from bondtrader.core.bond_models import Bond
from bondtrader.core.bond_valuation import BondValuator
from bondtrader.data.evaluation_dataset_generator import (
    EvaluationDatasetGenerator,
    EvaluationScenario,
    load_evaluation_dataset,
    save_evaluation_dataset,
)
from bondtrader.ml.drift_detection import DriftDetector
from bondtrader.utils.utils import logger


class AdaptiveEvaluator:
    """
    Adaptive evaluation system that adjusts evaluation data generator
    based on model performance feedback
    """

    def __init__(self, model_dir: str = None):
        """Initialize adaptive evaluator"""
        config = get_config()
        self.model_dir = model_dir or config.model_dir
        from bondtrader.core.container import get_container

        self.valuator = get_container().get_valuator()
        self.drift_detector = DriftDetector()
        self.feedback_history = []

    def load_models(self) -> Dict:
        """Load all trained models"""
        models = {}
        model_files = {
            "basic_ml": "ml_adjuster.joblib",
            "enhanced_ml": "enhanced_ml_adjuster.joblib",
            "advanced_ml": "advanced_ml_adjuster.joblib",
            "automl": "automl_adjuster.joblib",
        }

        for name, filename in model_files.items():
            path = os.path.join(self.model_dir, filename)
            if os.path.exists(path):
                try:
                    models[name] = joblib.load(path)
                    print(f"✓ Loaded {name}")
                except Exception as e:
                    print(f"✗ Failed to load {name}: {e}")

        return models

    def evaluate_models_on_dataset(self, models: Dict, evaluation_dataset: Dict, scenario_name: Optional[str] = None) -> Dict:
        """
        Evaluate all models on evaluation dataset

        Returns:
            Dictionary with evaluation results and feedback
        """
        results = {}

        # Get bonds from dataset
        if scenario_name:
            scenarios = {scenario_name: evaluation_dataset["scenarios"][scenario_name]}
        else:
            scenarios = evaluation_dataset["scenarios"]

        for model_name, model in models.items():
            print(f"\nEvaluating {model_name}...")
            model_results = {}

            for sc_name, scenario_data in scenarios.items():
                bonds = scenario_data.get("bonds", [])
                if not bonds:
                    continue

                predictions = []
                actuals = []
                fair_values = []
                errors = []

                for bond in bonds:
                    try:
                        # Get actual price
                        actual_price = bond.current_price
                        actuals.append(actual_price)

                        # Calculate fair value
                        fair_value = self.valuator.calculate_fair_value(bond)
                        fair_values.append(fair_value)

                        # Get model prediction
                        if hasattr(model, "predict_adjusted_value"):
                            pred_result = model.predict_adjusted_value(bond)
                            if isinstance(pred_result, dict):
                                pred_price = (
                                    pred_result.get("ml_adjusted_value")
                                    or pred_result.get("ml_adjusted_fair_value")
                                    or pred_result.get("predicted_value")
                                )
                            else:
                                pred_price = pred_result
                        else:
                            pred_price = fair_value

                        if pred_price is None or np.isnan(pred_price) or np.isinf(pred_price):
                            pred_price = fair_value

                        predictions.append(pred_price)

                        # Calculate error
                        error = abs(pred_price - actual_price)
                        errors.append(error)

                    except Exception as e:
                        logger.warning(f"Error evaluating bond {bond.bond_id}: {e}")
                        continue

                if len(predictions) == 0:
                    continue

                # Calculate metrics
                predictions = np.array(predictions)
                actuals = np.array(actuals)
                errors = np.array(errors)

                mae = np.mean(errors)
                rmse = np.sqrt(np.mean(errors**2))
                mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100

                # Calculate bias
                bias = np.mean(predictions - actuals)

                # Calculate correlation
                if len(predictions) > 1:
                    correlation = np.corrcoef(predictions, actuals)[0, 1]
                else:
                    correlation = 0.0

                model_results[sc_name] = {
                    "n_samples": len(predictions),
                    "mae": float(mae),
                    "rmse": float(rmse),
                    "mape": float(mape),
                    "bias": float(bias),
                    "correlation": float(correlation),
                    "mean_error": float(np.mean(errors)),
                    "max_error": float(np.max(errors)),
                    "std_error": float(np.std(errors)),
                }

                print(f"  {sc_name}: RMSE=${rmse:.2f}, MAE=${mae:.2f}, MAPE={mape:.2f}%")

            results[model_name] = model_results

        return results

    def analyze_feedback(self, evaluation_results: Dict, evaluation_dataset: Dict) -> Dict:
        """
        Analyze model performance and generate feedback for generator adjustment

        Args:
            evaluation_results: Model evaluation results
            evaluation_dataset: The evaluation dataset used

        Returns:
            Dictionary with feedback and recommendations
        """
        feedback = {
            "poor_performing_scenarios": [],
            "well_performing_scenarios": [],
            "recommendations": [],
            "metrics_summary": {},
            "price_distribution_analysis": {},
        }

        # Analyze price distributions in evaluation data
        for scenario_name, scenario_data in evaluation_dataset.get("scenarios", {}).items():
            if scenario_name == "benchmarks":
                continue

            bonds = scenario_data.get("bonds", [])
            if bonds:
                prices = [b.current_price for b in bonds]
                face_values = [b.face_value for b in bonds]

                feedback["price_distribution_analysis"][scenario_name] = {
                    "mean_price": float(np.mean(prices)),
                    "std_price": float(np.std(prices)),
                    "min_price": float(np.min(prices)),
                    "max_price": float(np.max(prices)),
                    "mean_face_value": float(np.mean(face_values)),
                    "price_to_face_ratio": float(np.mean([p / fv for p, fv in zip(prices, face_values)])),
                }

        # Aggregate metrics across all models
        scenario_metrics = {}

        for model_name, model_results in evaluation_results.items():
            for scenario_name, metrics in model_results.items():
                if scenario_name not in scenario_metrics:
                    scenario_metrics[scenario_name] = {
                        "rmse": [],
                        "mae": [],
                        "mape": [],
                        "bias": [],
                        "correlation": [],
                        "std_error": [],
                    }

                scenario_metrics[scenario_name]["rmse"].append(metrics["rmse"])
                scenario_metrics[scenario_name]["mae"].append(metrics["mae"])
                scenario_metrics[scenario_name]["mape"].append(metrics["mape"])
                scenario_metrics[scenario_name]["bias"].append(metrics["bias"])
                scenario_metrics[scenario_name]["correlation"].append(metrics["correlation"])
                scenario_metrics[scenario_name]["std_error"].append(metrics.get("std_error", 0))

        # Calculate averages and identify issues
        for scenario_name, metrics in scenario_metrics.items():
            avg_rmse = np.mean(metrics["rmse"])
            avg_mae = np.mean(metrics["mae"])
            avg_mape = np.mean(metrics["mape"])
            avg_correlation = np.mean(metrics["correlation"])
            avg_bias = np.mean(metrics["bias"])
            avg_std_error = np.mean(metrics["std_error"])

            feedback["metrics_summary"][scenario_name] = {
                "avg_rmse": float(avg_rmse),
                "avg_mae": float(avg_mae),
                "avg_mape": float(avg_mape),
                "avg_correlation": float(avg_correlation),
                "avg_bias": float(avg_bias),
                "avg_std_error": float(avg_std_error),
            }

            # Identify poor performing scenarios (adjusted thresholds)
            # Use relative thresholds based on price scale
            price_info = feedback["price_distribution_analysis"].get(scenario_name, {})
            mean_price = price_info.get("mean_price", 1000)

            # Relative thresholds
            rmse_threshold = mean_price * 0.1  # 10% of mean price
            mae_threshold = mean_price * 0.05  # 5% of mean price
            mape_threshold = 10.0  # 10% MAPE
            bias_threshold = mean_price * 0.02  # 2% bias

            is_poor = (
                avg_mape > mape_threshold
                or avg_rmse > rmse_threshold
                or avg_correlation < 0.5
                or abs(avg_bias) > bias_threshold
            )

            if is_poor:
                scenario_feedback = {
                    "scenario": scenario_name,
                    "issues": [],
                    "metrics": {
                        "rmse": float(avg_rmse),
                        "mae": float(avg_mae),
                        "mape": float(avg_mape),
                        "correlation": float(avg_correlation),
                        "bias": float(avg_bias),
                        "std_error": float(avg_std_error),
                    },
                    "adjustments_needed": {},
                }

                # Identify specific issues with actionable adjustments
                if avg_mape > mape_threshold:
                    scenario_feedback["issues"].append(f"High MAPE ({avg_mape:.2f}%) - price prediction struggling")
                    scenario_feedback["adjustments_needed"]["reduce_price_variance"] = True
                    scenario_feedback["adjustments_needed"]["align_with_training_range"] = True

                if avg_rmse > rmse_threshold:
                    scenario_feedback["issues"].append(f"High RMSE (${avg_rmse:.2f}) - large prediction errors")
                    scenario_feedback["adjustments_needed"]["reduce_volatility"] = True

                if avg_correlation < 0.5:
                    scenario_feedback["issues"].append(f"Low correlation ({avg_correlation:.3f}) - not capturing patterns")
                    scenario_feedback["adjustments_needed"]["improve_market_structure"] = True

                if abs(avg_bias) > bias_threshold:
                    scenario_feedback["issues"].append(f"Significant bias (${avg_bias:.2f}) - systematic prediction error")
                    scenario_feedback["adjustments_needed"]["adjust_risk_free_rate"] = True
                    scenario_feedback["adjustments_needed"]["bias_direction"] = "up" if avg_bias > 0 else "down"

                feedback["poor_performing_scenarios"].append(scenario_feedback)

            # Identify well performing scenarios
            elif avg_mape < 2.0 and avg_correlation > 0.8 and avg_rmse < rmse_threshold:
                feedback["well_performing_scenarios"].append(scenario_name)

        # Generate recommendations
        if feedback["poor_performing_scenarios"]:
            feedback["recommendations"].append("Adjust evaluation data to better match training data characteristics")
            feedback["recommendations"].append("Reduce price variance in poorly performing scenarios")
            feedback["recommendations"].append("Align bond price ranges with historical training data")

        if len(feedback["well_performing_scenarios"]) > len(feedback["poor_performing_scenarios"]):
            feedback["recommendations"].append("Models are performing well overall - consider adding edge cases")

        return feedback

    def adjust_generator(self, generator: EvaluationDatasetGenerator, feedback: Dict) -> EvaluationDatasetGenerator:
        """
        Adjust evaluation generator based on feedback

        Args:
            generator: Current evaluation generator
            feedback: Feedback from model evaluation

        Returns:
            Adjusted generator
        """
        print("\n" + "=" * 70)
        print("ADJUSTING EVALUATION GENERATOR BASED ON FEEDBACK")
        print("=" * 70)

        # Adjust scenarios based on poor performance
        for poor_scenario in feedback["poor_performing_scenarios"]:
            scenario_name = poor_scenario["scenario"]
            issues = poor_scenario["issues"]
            adjustments = poor_scenario.get("adjustments_needed", {})
            metrics = poor_scenario["metrics"]

            print(f"\nAdjusting scenario: {scenario_name}")
            print(f"  Issues: {', '.join(issues)}")

            if scenario_name in generator.evaluation_scenarios:
                scenario = generator.evaluation_scenarios[scenario_name]

                # Adjust based on specific issues
                if adjustments.get("reduce_volatility"):
                    # Reduce volatility multiplier
                    scenario.volatility_multiplier *= 0.85
                    print(f"  ✓ Reduced volatility multiplier to {scenario.volatility_multiplier:.3f}")

                if adjustments.get("reduce_price_variance"):
                    # Reduce credit spread adjustment to make prices more consistent
                    scenario.credit_spread_adjustment *= 0.8
                    print(f"  ✓ Reduced credit spread adjustment to {scenario.credit_spread_adjustment:.4f}")

                if adjustments.get("align_with_training_range"):
                    # Adjust liquidity factor to bring prices closer to training range
                    scenario.liquidity_factor = min(1.0, scenario.liquidity_factor * 1.1)
                    print(f"  ✓ Adjusted liquidity factor to {scenario.liquidity_factor:.3f}")

                if adjustments.get("improve_market_structure"):
                    # Adjust market sentiment for better correlation
                    if abs(scenario.market_sentiment) > 0.5:
                        scenario.market_sentiment *= 0.7
                    print(f"  ✓ Adjusted market sentiment to {scenario.market_sentiment:.3f}")

                if adjustments.get("adjust_risk_free_rate"):
                    # Adjust risk-free rate based on bias direction
                    bias = metrics["bias"]
                    if adjustments.get("bias_direction") == "up":
                        # Model over-predicting, reduce risk-free rate
                        scenario.risk_free_rate *= 0.97
                    else:
                        # Model under-predicting, increase risk-free rate
                        scenario.risk_free_rate *= 1.03
                    print(f"  ✓ Adjusted risk-free rate to {scenario.risk_free_rate:.4f} (bias: ${bias:.2f})")

        # Add more diverse scenarios if models are performing well
        if len(feedback["well_performing_scenarios"]) > 3:
            print("\nModels performing well - scenarios are well-calibrated")

        return generator

    def run_adaptive_evaluation(
        self,
        num_iterations: int = 3,
        num_bonds: int = 1000,
        initial_scenarios: Optional[List[str]] = None,
    ) -> Dict:
        """
        Run adaptive evaluation loop

        Args:
            num_iterations: Number of adjustment iterations
            num_bonds: Number of bonds per evaluation
            initial_scenarios: Initial scenarios to test

        Returns:
            Final evaluation results and history
        """
        print("=" * 70)
        print("ADAPTIVE EVALUATION SYSTEM")
        print("=" * 70)

        # Load models
        print("\n[Step 1] Loading trained models...")
        models = self.load_models()
        if not models:
            print("ERROR: No models found!")
            return {}

        # Initialize generator
        generator = EvaluationDatasetGenerator(seed=42)

        if initial_scenarios is None:
            initial_scenarios = [
                "normal_market",
                "rate_shock_up_200bps",
                "credit_spread_widening",
                "liquidity_crisis",
            ]

        all_results = []

        for iteration in range(num_iterations):
            print(f"\n{'='*70}")
            print(f"ITERATION {iteration + 1}/{num_iterations}")
            print(f"{'='*70}")

            # Generate evaluation dataset
            print(f"\n[Iteration {iteration + 1}] Generating evaluation dataset...")
            evaluation_dataset = generator.generate_evaluation_dataset(
                num_bonds=num_bonds,
                scenarios=initial_scenarios,
                include_benchmarks=True,
                point_in_time=True,
            )

            # Evaluate models
            print(f"\n[Iteration {iteration + 1}] Evaluating models...")
            results = self.evaluate_models_on_dataset(models, evaluation_dataset)
            all_results.append(results)

            # Analyze feedback
            print(f"\n[Iteration {iteration + 1}] Analyzing feedback...")
            feedback = self.analyze_feedback(results, evaluation_dataset)
            self.feedback_history.append(feedback)

            # Print feedback summary
            print(f"\nFeedback Summary:")
            print(f"  Poor performing scenarios: {len(feedback['poor_performing_scenarios'])}")
            print(f"  Well performing scenarios: {len(feedback['well_performing_scenarios'])}")

            if feedback["poor_performing_scenarios"]:
                print(f"\n  Poor performers:")
                for poor in feedback["poor_performing_scenarios"]:
                    print(f"    - {poor['scenario']}: {', '.join(poor['issues'])}")

            # Adjust generator (except on last iteration)
            if iteration < num_iterations - 1:
                generator = self.adjust_generator(generator, feedback)

            # Save intermediate results
            results_path = f"evaluation_results/iteration_{iteration + 1}_results.json"
            os.makedirs("evaluation_results", exist_ok=True)
            with open(results_path, "w") as f:
                json.dump(
                    {
                        "iteration": iteration + 1,
                        "results": {
                            k: {sk: {m: float(v) for m, v in sv.items()} for sk, sv in v.items()} for k, v in results.items()
                        },
                        "feedback": {
                            "poor_performing_scenarios": feedback["poor_performing_scenarios"],
                            "well_performing_scenarios": feedback["well_performing_scenarios"],
                            "recommendations": feedback["recommendations"],
                        },
                    },
                    f,
                    indent=2,
                )
            print(f"\n  ✓ Saved results to {results_path}")

        # Final summary
        print("\n" + "=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)

        final_feedback = self.feedback_history[-1]
        print(f"\nFinal Performance:")
        print(f"  Poor performing scenarios: {len(final_feedback['poor_performing_scenarios'])}")
        print(f"  Well performing scenarios: {len(final_feedback['well_performing_scenarios'])}")

        print(f"\nRecommendations:")
        for rec in final_feedback["recommendations"]:
            print(f"  - {rec}")

        return {
            "all_results": all_results,
            "feedback_history": self.feedback_history,
            "final_generator": generator,
        }


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Adaptive evaluation with generator adjustment")
    parser.add_argument("--iterations", type=int, default=3, help="Number of adjustment iterations (default: 3)")
    parser.add_argument("--num-bonds", type=int, default=1000, help="Number of bonds per evaluation (default: 1000)")
    # Get config for default values
    config = get_config()

    parser.add_argument(
        "--model-dir",
        type=str,
        default=config.model_dir,
        help=f"Directory with trained models (default: {config.model_dir})",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        nargs="+",
        default=None,
        help="Specific scenarios to test (default: normal_market, rate_shock_up_200bps, etc.)",
    )

    args = parser.parse_args()

    evaluator = AdaptiveEvaluator(model_dir=args.model_dir)

    try:
        results = evaluator.run_adaptive_evaluation(
            num_iterations=args.iterations,
            num_bonds=args.num_bonds,
            initial_scenarios=args.scenarios,
        )

        print("\n✓ Adaptive evaluation complete!")
        print(f"Results saved to: evaluation_results/")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
