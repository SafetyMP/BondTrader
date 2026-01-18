"""
Comprehensive Model Evaluation and Scoring System
Evaluates all models in the codebase using the evaluation dataset generator
and creates performance scores based on multiple metrics.

Scoring System:
- Combines multiple evaluation metrics into a single score
- Weights metrics based on financial industry best practices
- Considers prediction accuracy, drift, correlation, and stress test performance
- Generates detailed performance reports
"""

import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

    def tqdm(iterable=None, desc=None, total=None, **kwargs):
        return iterable if iterable is not None else range(total or 0)


from bondtrader.core.bond_models import Bond
from bondtrader.core.bond_valuation import BondValuator
from bondtrader.data.evaluation_dataset_generator import (
    EvaluationDatasetGenerator,
    EvaluationMetrics,
    load_evaluation_dataset,
    save_evaluation_dataset,
)
from bondtrader.ml.automl import AutoMLBondAdjuster
from bondtrader.ml.ml_adjuster import MLBondAdjuster
from bondtrader.ml.ml_adjuster_enhanced import EnhancedMLBondAdjuster
from bondtrader.ml.ml_advanced import AdvancedMLBondAdjuster


class ModelPerformanceScorer:
    """
    Calculates comprehensive performance scores for models based on evaluation metrics.

    Scoring Components:
    1. Prediction Accuracy (40%): R², RMSE, MAPE
    2. Benchmark Drift (30%): Drift scores vs industry benchmarks
    3. Correlation (15%): Correlation with benchmarks and actuals
    4. Stress Test Performance (15%): Performance under stress scenarios
    """

    def __init__(self):
        """Initialize scorer"""
        # Scoring weights (must sum to 1.0)
        self.weights = {"prediction_accuracy": 0.40, "benchmark_drift": 0.30, "correlation": 0.15, "stress_performance": 0.15}

    def calculate_model_score(
        self, evaluation_results: Dict[str, EvaluationMetrics], scenario_type_weights: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        Calculate comprehensive score for a model across all scenarios

        Args:
            evaluation_results: Dictionary mapping scenario names to EvaluationMetrics
            scenario_type_weights: Optional weights for different scenario types

        Returns:
            Dictionary with overall score and component scores
        """
        if scenario_type_weights is None:
            scenario_type_weights = {"normal": 0.30, "stress": 0.35, "crisis": 0.20, "sensitivity": 0.15}

        # Calculate component scores for each scenario
        scenario_scores = {}

        for scenario_name, metrics in evaluation_results.items():
            # Prediction accuracy component (0-100)
            accuracy_score = self._calculate_accuracy_score(metrics)

            # Benchmark drift component (0-100, higher is better)
            drift_score = self._calculate_drift_score(metrics)

            # Correlation component (0-100)
            correlation_score = self._calculate_correlation_score(metrics)

            # Overall scenario score
            scenario_score = (
                accuracy_score * self.weights["prediction_accuracy"]
                + drift_score * self.weights["benchmark_drift"]
                + correlation_score * self.weights["correlation"]
            )

            scenario_scores[scenario_name] = {
                "accuracy_score": accuracy_score,
                "drift_score": drift_score,
                "correlation_score": correlation_score,
                "scenario_score": scenario_score,
            }

        # Weighted average across scenarios
        weighted_scenario_score = 0.0
        total_weight = 0.0

        # Group scenarios by type
        stress_scenarios = []
        normal_scenarios = []
        crisis_scenarios = []
        sensitivity_scenarios = []

        for scenario_name, metrics in evaluation_results.items():
            # Infer scenario type from name (simplified)
            if "normal" in scenario_name.lower() or "recovery" in scenario_name.lower():
                normal_scenarios.append(scenario_name)
                weight = scenario_type_weights.get("normal", 0.30)
            elif "crisis" in scenario_name.lower() or "crash" in scenario_name.lower():
                crisis_scenarios.append(scenario_name)
                weight = scenario_type_weights.get("crisis", 0.20)
            elif "stress" in scenario_name.lower() or "shock" in scenario_name.lower():
                stress_scenarios.append(scenario_name)
                weight = scenario_type_weights.get("stress", 0.35)
            else:
                sensitivity_scenarios.append(scenario_name)
                weight = scenario_type_weights.get("sensitivity", 0.15)

            weighted_scenario_score += scenario_scores[scenario_name]["scenario_score"] * weight
            total_weight += weight

        if total_weight > 0:
            overall_score = weighted_scenario_score / total_weight
        else:
            overall_score = 0.0

        # Stress test performance (average across stress and crisis scenarios)
        stress_performance = 0.0
        stress_count = 0
        for scenario_name in stress_scenarios + crisis_scenarios:
            if scenario_name in scenario_scores:
                stress_performance += scenario_scores[scenario_name]["scenario_score"]
                stress_count += 1

        if stress_count > 0:
            stress_performance = stress_performance / stress_count
        else:
            stress_performance = overall_score  # Fallback

        # Final weighted score
        final_score = (
            overall_score * (1.0 - self.weights["stress_performance"])
            + stress_performance * self.weights["stress_performance"]
        )

        return {
            "overall_score": final_score,
            "accuracy_component": self._aggregate_component_scores(
                scenario_scores, "accuracy_score", scenario_type_weights, evaluation_results
            ),
            "drift_component": self._aggregate_component_scores(
                scenario_scores, "drift_score", scenario_type_weights, evaluation_results
            ),
            "correlation_component": self._aggregate_component_scores(
                scenario_scores, "correlation_score", scenario_type_weights, evaluation_results
            ),
            "stress_performance": stress_performance,
            "scenario_scores": scenario_scores,
            "scenario_breakdown": {
                "normal_scenarios": len(normal_scenarios),
                "stress_scenarios": len(stress_scenarios),
                "crisis_scenarios": len(crisis_scenarios),
                "sensitivity_scenarios": len(sensitivity_scenarios),
            },
        }

    def _calculate_accuracy_score(self, metrics: EvaluationMetrics) -> float:
        """Calculate accuracy score (0-100) based on R², RMSE, MAPE"""
        # R² score (0-1, higher is better) -> convert to 0-50 scale
        r2_score = max(0.0, min(1.0, metrics.r2_score)) * 50

        # RMSE penalty (normalized, lower is better) -> 0-30 scale
        # Assume reasonable RMSE is < 10% of average price
        # Use price_drift_score as proxy for normalized RMSE
        rmse_score = max(0.0, 30.0 * (1.0 - min(1.0, metrics.price_drift_score)))

        # MAPE score (lower is better) -> 0-20 scale
        # Good MAPE is < 5%, acceptable is < 10%
        mape_normalized = min(1.0, metrics.mean_absolute_percentage_error / 20.0)
        mape_score = max(0.0, 20.0 * (1.0 - mape_normalized))

        return r2_score + rmse_score + mape_score

    def _calculate_drift_score(self, metrics: EvaluationMetrics) -> float:
        """Calculate drift score (0-100, higher is better)"""
        # Use consensus drift as primary metric
        consensus_drift = metrics.drift_vs_consensus.drift_score

        # Drift score: 0 (no drift) = 100, 1 (max drift) = 0
        drift_score = 100.0 * (1.0 - consensus_drift)

        # Bonus for low drift across all benchmarks
        avg_drift = (
            metrics.drift_vs_bloomberg.drift_score
            + metrics.drift_vs_aladdin.drift_score
            + metrics.drift_vs_goldman.drift_score
            + metrics.drift_vs_jpmorgan.drift_score
        ) / 4.0

        # Bonus if drift is consistently low
        if avg_drift < 0.1:
            drift_score *= 1.1  # 10% bonus

        return min(100.0, drift_score)

    def _calculate_correlation_score(self, metrics: EvaluationMetrics) -> float:
        """Calculate correlation score (0-100)"""
        # Primary: consensus correlation
        consensus_corr = max(0.0, metrics.drift_vs_consensus.correlation)

        # Return correlation with actual prices
        return_corr = max(0.0, metrics.return_correlation)

        # Weighted average
        correlation_score = (consensus_corr * 0.7 + return_corr * 0.3) * 100

        return correlation_score

    def _aggregate_component_scores(
        self,
        scenario_scores: Dict,
        component_key: str,
        scenario_weights: Dict[str, float],
        evaluation_results: Dict[str, EvaluationMetrics],
    ) -> float:
        """Aggregate component scores across scenarios"""
        total_score = 0.0
        total_weight = 0.0

        for scenario_name, scores in scenario_scores.items():
            # Get weight for scenario type
            if "normal" in scenario_name.lower() or "recovery" in scenario_name.lower():
                weight = scenario_weights.get("normal", 0.30)
            elif "crisis" in scenario_name.lower() or "crash" in scenario_name.lower():
                weight = scenario_weights.get("crisis", 0.20)
            elif "stress" in scenario_name.lower() or "shock" in scenario_name.lower():
                weight = scenario_weights.get("stress", 0.35)
            else:
                weight = scenario_weights.get("sensitivity", 0.15)

            total_score += scores[component_key] * weight
            total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0


class ModelEvaluator:
    """
    Comprehensive model evaluator that loads models, generates evaluation datasets,
    evaluates models, and creates performance scores.
    """

    def __init__(self, model_dir: str = "trained_models", evaluation_data_dir: str = "evaluation_data"):
        """Initialize evaluator"""
        self.model_dir = model_dir
        self.evaluation_data_dir = evaluation_data_dir
        self.scorer = ModelPerformanceScorer()
        self.valuator = BondValuator()

        os.makedirs(evaluation_data_dir, exist_ok=True)
        os.makedirs("evaluation_results", exist_ok=True)

    def load_model(self, model_name: str) -> Optional[object]:
        """Load a trained model from disk"""
        model_path = os.path.join(self.model_dir, f"{model_name}.joblib")

        if not os.path.exists(model_path):
            return None

        try:
            data = joblib.load(model_path)

            # Determine model type and instantiate
            if model_name == "ml_adjuster":
                # Check if it's already a model object or dict
                if hasattr(data, "predict_adjusted_value"):
                    return data
                model = MLBondAdjuster()
                model.load_model(model_path)
                return model
            elif model_name == "enhanced_ml_adjuster":
                # Check if it's already a model object
                if hasattr(data, "predict_adjusted_value"):
                    return data
                model = EnhancedMLBondAdjuster()
                model.load_model(model_path)
                return model
            elif model_name == "advanced_ml_adjuster":
                # Advanced model is likely saved as the full model object
                if hasattr(data, "predict_adjusted_value"):
                    return data  # Already a model object
                elif isinstance(data, dict):
                    # Try to reconstruct if it's a dict with model components
                    if "ensemble_model" in data or "models" in data:
                        model = AdvancedMLBondAdjuster(self.valuator)
                        # Restore attributes if possible
                        for key, value in data.items():
                            if hasattr(model, key):
                                setattr(model, key, value)
                        if data.get("is_trained", False):
                            model.is_trained = True
                        return model
                    elif "model" in data:
                        return data["model"]
                # Fallback: return as-is
                return data
            elif model_name == "automl":
                # AutoML is likely saved as the full model object
                if hasattr(data, "predict_adjusted_value"):
                    return data
                elif isinstance(data, dict):
                    # Try to reconstruct
                    if "best_model" in data or "is_trained" in data:
                        model = AutoMLBondAdjuster(self.valuator)
                        # Restore attributes
                        for key, value in data.items():
                            if hasattr(model, key):
                                setattr(model, key, value)
                        if data.get("is_trained", False):
                            model.is_trained = True
                        return model
                    elif "model" in data:
                        return data["model"]
                # Fallback: return as-is
                return data
            else:
                # Generic loading
                return data

        except Exception as e:
            print(f"  Error loading {model_name}: {e}")
            import traceback

            traceback.print_exc()
            return None

    def load_all_models(self) -> Dict[str, object]:
        """Load all available trained models"""
        print("\n[Loading Models]")
        models = {}

        model_names = ["ml_adjuster", "enhanced_ml_adjuster", "advanced_ml_adjuster", "automl"]

        for model_name in model_names:
            model = self.load_model(model_name)
            if model is not None:
                models[model_name] = model
                print(f"  ✓ Loaded {model_name}")
            else:
                print(f"  ✗ {model_name} not found")

        return models

    def generate_or_load_evaluation_dataset(
        self, generate_new: bool = False, num_bonds: int = 1000  # Reduced for faster execution
    ) -> Dict:
        """Generate or load evaluation dataset"""
        eval_path = os.path.join(self.evaluation_data_dir, "evaluation_dataset.joblib")

        if generate_new or not os.path.exists(eval_path):
            print("\n[Generating Evaluation Dataset]")
            print("  This may take a few minutes...")
            generator = EvaluationDatasetGenerator(seed=42)
            # Use fewer scenarios for faster execution
            evaluation_dataset = generator.generate_evaluation_dataset(
                num_bonds=num_bonds,
                scenarios=["normal_market", "rate_shock_up_200bps", "credit_spread_widening"],  # Reduced scenarios
                include_benchmarks=True,
                point_in_time=True,
            )
            save_evaluation_dataset(evaluation_dataset, eval_path)
            print(f"  ✓ Dataset saved to {eval_path}")
        else:
            print(f"\n[Loading Evaluation Dataset from {eval_path}]")
            evaluation_dataset = load_evaluation_dataset(eval_path)

            # Restore EvaluationScenario objects from dictionaries
            from datetime import datetime

            from bondtrader.data.evaluation_dataset_generator import EvaluationScenario

            # Check if we need to restore bonds (if evaluation_bonds is available)
            evaluation_bonds = evaluation_dataset.get("evaluation_bonds", [])

            for scenario_name, scenario_data in evaluation_dataset["scenarios"].items():
                if scenario_name == "benchmarks":
                    continue

                if not isinstance(scenario_data, dict):
                    continue

                # Restore scenario object
                if "scenario" in scenario_data:
                    if isinstance(scenario_data["scenario"], dict):
                        # Restore EvaluationScenario from dict
                        scenario_dict = scenario_data["scenario"]
                        scenario_data["scenario"] = EvaluationScenario(**scenario_dict)

                # Restore bonds if missing
                if "bonds" not in scenario_data or not scenario_data["bonds"]:
                    # Regenerate bonds from saved data
                    if evaluation_bonds:
                        # Use evaluation_bonds if available
                        num_bonds = scenario_data.get("num_bonds", len(evaluation_bonds))
                        bonds_to_use = evaluation_bonds[:num_bonds]

                        # Update bond prices to match actual_prices from scenario
                        restored_bonds = []
                        actual_prices = scenario_data.get("actual_prices", [])

                        for i, bond in enumerate(bonds_to_use):
                            if i < len(actual_prices):
                                # Create bond copy with scenario price
                                from bondtrader.core.bond_models import Bond

                                restored_bond = Bond(
                                    bond_id=bond.bond_id,
                                    bond_type=bond.bond_type,
                                    face_value=bond.face_value,
                                    coupon_rate=bond.coupon_rate,
                                    maturity_date=bond.maturity_date,
                                    issue_date=bond.issue_date,
                                    current_price=float(actual_prices[i]),
                                    credit_rating=bond.credit_rating,
                                    issuer=bond.issuer,
                                    frequency=bond.frequency,
                                    callable=bond.callable,
                                    convertible=bond.convertible,
                                )
                                restored_bonds.append(restored_bond)

                        scenario_data["bonds"] = restored_bonds
                    else:
                        # Fallback: regenerate evaluation dataset if bonds are missing
                        print(f"  ⚠️  Bonds missing for {scenario_name}, regenerating dataset...")
                        generator = EvaluationDatasetGenerator(seed=42)
                        evaluation_dataset = generator.generate_evaluation_dataset(
                            num_bonds=1000, scenarios=[scenario_name], include_benchmarks=True, point_in_time=True
                        )
                        break

            print(f"  ✓ Dataset loaded and restored")

        return evaluation_dataset

    def evaluate_all_models(
        self,
        models: Dict[str, object],
        evaluation_dataset: Dict,
        save_results: bool = True,
        use_parallel: bool = True,
        max_workers: Optional[int] = None,
        batch_size: int = 100,
    ) -> Dict:
        """
        Evaluate all models on evaluation dataset and calculate scores (optimized)

        Args:
            models: Dictionary of model names to model objects
            evaluation_dataset: Evaluation dataset
            save_results: Whether to save results to disk
            use_parallel: Whether to evaluate models in parallel
            max_workers: Maximum parallel workers (None = auto-detect)
            batch_size: Batch size for bond processing

        Returns:
            Dictionary with evaluation results and scores for all models
        """
        print("\n[Evaluating Models]")
        eval_generator = EvaluationDatasetGenerator()

        all_results = {}
        start_time = time.time()

        if use_parallel and len(models) > 1:
            # Parallel model evaluation
            all_results = self._evaluate_models_parallel(models, evaluation_dataset, eval_generator, max_workers, batch_size)
        else:
            # Sequential evaluation with progress tracking
            model_items = list(models.items())
            model_iter = tqdm(model_items, desc="Evaluating models", disable=not TQDM_AVAILABLE)

            for model_name, model in model_iter:
                if TQDM_AVAILABLE:
                    model_iter.set_description(f"Evaluating {model_name}")

                try:
                    model_start = time.time()
                    # Evaluate model on all scenarios
                    evaluation_results = eval_generator.evaluate_model(
                        model=model,
                        evaluation_dataset=evaluation_dataset,
                        scenario_name=None,  # All scenarios
                        batch_size=batch_size,
                        use_parallel=use_parallel,
                        max_workers=max_workers,
                    )

                    # Calculate performance score
                    score_result = self.scorer.calculate_model_score(evaluation_results)

                    model_time = time.time() - model_start

                    all_results[model_name] = {
                        "evaluation_results": evaluation_results,
                        "performance_score": score_result,
                        "status": "success",
                        "evaluation_time": model_time,
                    }

                    if TQDM_AVAILABLE:
                        model_iter.set_postfix({"score": f"{score_result['overall_score']:.1f}", "time": f"{model_time:.1f}s"})
                    else:
                        print(f"    ✓ Overall Score: {score_result['overall_score']:.2f}/100")
                        print(f"    ✓ Accuracy Component: {score_result['accuracy_component']:.2f}/100")
                        print(f"    ✓ Drift Component: {score_result['drift_component']:.2f}/100")
                        print(f"    ✓ Time: {model_time:.1f}s")

                except Exception as e:
                    print(f"    ✗ Evaluation failed: {e}")
                    import traceback

                    print(f"    Full error traceback:")
                    traceback.print_exc()
                    all_results[model_name] = {"status": "failed", "error": str(e), "traceback": traceback.format_exc()}

        total_time = time.time() - start_time
        print(f"\n[Evaluation Complete] Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")

        # Save results
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_path = f"evaluation_results/model_scores_{timestamp}.joblib"

            # Convert to serializable format
            serializable_results = self._make_serializable(all_results)

            joblib.dump(
                {
                    "evaluation_results": serializable_results,
                    "timestamp": datetime.now().isoformat(),
                    "scorer_weights": self.scorer.weights,
                    "total_evaluation_time": total_time,
                },
                results_path,
            )

            print(f"\n[Results Saved]")
            print(f"  ✓ Saved to {results_path}")

        return all_results

    def _evaluate_models_parallel(
        self,
        models: Dict[str, object],
        evaluation_dataset: Dict,
        eval_generator: "EvaluationDatasetGenerator",
        max_workers: Optional[int],
        batch_size: int,
    ) -> Dict:
        """Evaluate multiple models in parallel"""
        if max_workers is None:
            max_workers = min(len(models), mp.cpu_count(), 4)  # Limit to 4 for memory

        def evaluate_single_model(model_name, model):
            """Evaluate a single model"""
            try:
                start_time = time.time()
                evaluation_results = eval_generator.evaluate_model(
                    model=model,
                    evaluation_dataset=evaluation_dataset,
                    scenario_name=None,
                    batch_size=batch_size,
                    use_parallel=True,
                    max_workers=max_workers,
                )

                score_result = self.scorer.calculate_model_score(evaluation_results)
                eval_time = time.time() - start_time

                return {
                    "model_name": model_name,
                    "status": "success",
                    "evaluation_results": evaluation_results,
                    "performance_score": score_result,
                    "evaluation_time": eval_time,
                }
            except Exception as e:
                return {"model_name": model_name, "status": "failed", "error": str(e)}

        all_results = {}

        # Use ThreadPoolExecutor for I/O-bound operations (model predictions)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(evaluate_single_model, name, model): name for name, model in models.items()}

            # Process completed evaluations with progress bar
            completed = tqdm(as_completed(futures), total=len(futures), desc="Parallel evaluation", disable=not TQDM_AVAILABLE)

            for future in completed:
                result = future.result()
                model_name = result["model_name"]
                all_results[model_name] = {k: v for k, v in result.items() if k != "model_name"}

                if result.get("status") == "success" and TQDM_AVAILABLE:
                    score = result["performance_score"]["overall_score"]
                    eval_time = result.get("evaluation_time", 0)
                    completed.set_postfix({f"{model_name}": f"{score:.1f} ({eval_time:.1f}s)"})

        return all_results

    def _make_serializable(self, results: Dict) -> Dict:
        """Convert results to serializable format"""
        serializable = {}

        for model_name, model_data in results.items():
            if model_data.get("status") == "failed":
                serializable[model_name] = model_data
                continue

            serializable_model = {"status": "success", "performance_score": model_data["performance_score"]}

            # Convert evaluation results
            eval_results_serialized = {}
            for scenario_name, metrics in model_data["evaluation_results"].items():
                eval_results_serialized[scenario_name] = metrics.to_dict()

            serializable_model["evaluation_results"] = eval_results_serialized
            serializable[model_name] = serializable_model

        return serializable

    def generate_scoring_report(self, results: Dict, output_file: Optional[str] = None) -> Dict:
        """
        Generate comprehensive scoring report

        Returns:
            Dictionary with report data
        """
        print("\n" + "=" * 80)
        print("MODEL PERFORMANCE SCORING REPORT")
        print("=" * 80)

        report = {"summary": {}, "model_rankings": [], "detailed_scores": {}, "best_performers": {}, "warnings": []}

        # Collect all model scores
        model_scores = {}
        for model_name, model_data in results.items():
            if model_data.get("status") == "success":
                score = model_data["performance_score"]["overall_score"]
                model_scores[model_name] = score

        # Rank models
        ranked_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)

        report["model_rankings"] = [
            {"model": name, "score": score, "rank": i + 1} for i, (name, score) in enumerate(ranked_models)
        ]

        # Summary statistics
        if model_scores:
            report["summary"] = {
                "total_models": len(model_scores),
                "average_score": np.mean(list(model_scores.values())),
                "best_score": max(model_scores.values()),
                "worst_score": min(model_scores.values()),
                "score_std": np.std(list(model_scores.values())),
            }

        # Detailed scores
        for model_name, model_data in results.items():
            if model_data.get("status") == "success":
                score_result = model_data["performance_score"]
                report["detailed_scores"][model_name] = {
                    "overall_score": score_result["overall_score"],
                    "accuracy_component": score_result["accuracy_component"],
                    "drift_component": score_result["drift_component"],
                    "correlation_component": score_result["correlation_component"],
                    "stress_performance": score_result["stress_performance"],
                }

        # Best performers
        if ranked_models:
            best_model = ranked_models[0]
            report["best_performers"] = {"overall_best": best_model[0], "best_score": best_model[1]}

        # Warnings
        for model_name, model_data in results.items():
            if model_data.get("status") == "failed":
                report["warnings"].append(f"{model_name}: Evaluation failed")
            elif model_data.get("status") == "success":
                score = model_data["performance_score"]["overall_score"]
                if score < 50:
                    report["warnings"].append(f"{model_name}: Low performance score ({score:.2f}/100)")
                if score < 70:
                    report["warnings"].append(f"{model_name}: Performance below industry standards ({score:.2f}/100)")

        # Print report
        self._print_report(report)

        # Save report
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"evaluation_results/scoring_report_{timestamp}.json"

        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n[Report Saved]")
        print(f"  ✓ Saved to {output_file}")

        return report

    def _print_report(self, report: Dict):
        """Print human-readable report"""
        # Summary
        if report["summary"]:
            summary = report["summary"]
            print(f"\nSUMMARY")
            print(f"  Total Models Evaluated: {summary['total_models']}")
            print(f"  Average Score: {summary['average_score']:.2f}/100")
            print(f"  Best Score: {summary['best_score']:.2f}/100")
            print(f"  Score Std Dev: {summary['score_std']:.2f}")

        # Rankings
        print(f"\nMODEL RANKINGS (by Overall Score)")
        print(f"{'Rank':<6} {'Model':<30} {'Score':<10} {'Grade':<10}")
        print("-" * 60)

        for ranking in report["model_rankings"]:
            score = ranking["score"]
            grade = self._score_to_grade(score)
            print(f"{ranking['rank']:<6} {ranking['model']:<30} {score:<10.2f} {grade:<10}")

        # Detailed scores
        print(f"\nDETAILED SCORES")
        for model_name, scores in report["detailed_scores"].items():
            print(f"\n  {model_name}:")
            print(f"    Overall Score: {scores['overall_score']:.2f}/100")
            print(f"    - Accuracy Component: {scores['accuracy_component']:.2f}/100")
            print(f"    - Drift Component: {scores['drift_component']:.2f}/100")
            print(f"    - Correlation Component: {scores['correlation_component']:.2f}/100")
            print(f"    - Stress Performance: {scores['stress_performance']:.2f}/100")

        # Best performers
        if report["best_performers"]:
            best = report["best_performers"]
            print(f"\nBEST PERFORMING MODEL")
            print(f"  Model: {best['overall_best']}")
            print(f"  Score: {best['best_score']:.2f}/100")
            print(f"  Grade: {self._score_to_grade(best['best_score'])}")

        # Warnings
        if report["warnings"]:
            print(f"\n⚠️  WARNINGS")
            for warning in report["warnings"]:
                print(f"  - {warning}")

        print("\n" + "=" * 80)

    def _score_to_grade(self, score: float) -> str:
        """Convert score to letter grade"""
        if score >= 90:
            return "A+ (Excellent)"
        elif score >= 85:
            return "A (Very Good)"
        elif score >= 80:
            return "A- (Good)"
        elif score >= 75:
            return "B+ (Above Average)"
        elif score >= 70:
            return "B (Average)"
        elif score >= 65:
            return "B- (Below Average)"
        elif score >= 60:
            return "C+ (Fair)"
        elif score >= 50:
            return "C (Poor)"
        else:
            return "F (Failing)"


def main():
    """Main evaluation function"""
    print("=" * 80)
    print("COMPREHENSIVE MODEL EVALUATION AND SCORING SYSTEM")
    print("=" * 80)

    # Initialize evaluator
    evaluator = ModelEvaluator(model_dir="trained_models", evaluation_data_dir="evaluation_data")

    # Step 1: Load models
    models = evaluator.load_all_models()

    if not models:
        print("\n⚠️  No models found. Please train models first using train_all_models.py")
        return

    # Step 2: Generate or load evaluation dataset
    evaluation_dataset = evaluator.generate_or_load_evaluation_dataset(
        generate_new=True, num_bonds=1000  # Generate new dataset  # Reduced for faster execution
    )

    # Step 3: Evaluate all models
    results = evaluator.evaluate_all_models(models=models, evaluation_dataset=evaluation_dataset, save_results=True)

    # Step 4: Generate scoring report
    report = evaluator.generate_scoring_report(results)

    print("\n✓ Evaluation and scoring complete!")

    return results, report


if __name__ == "__main__":
    results, report = main()
