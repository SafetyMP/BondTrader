"""
Drift Detection and Benchmark Comparison Module
Compares model outputs against expected outputs from leading financial firms
to detect and minimize drift.

Industry Benchmarks:
- Bloomberg Terminal: Sophisticated pricing models with market data integration
- BlackRock Aladdin: Advanced risk-adjusted valuation
- Goldman Sachs: Proprietary credit-adjusted models
- JPMorgan: Liquidity-adjusted pricing models
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from bondtrader.core.bond_models import Bond
from bondtrader.core.bond_valuation import BondValuator
from bondtrader.utils.utils import logger


@dataclass
class DriftMetrics:
    """Drift metrics for a model compared to benchmark"""

    mean_absolute_error: float
    mean_squared_error: float
    root_mean_squared_error: float
    mean_absolute_percentage_error: float
    max_absolute_error: float
    drift_score: float  # 0-1, where 0 is no drift, 1 is maximum drift
    correlation: float
    bias: float  # Systematic bias (mean error)
    variance_ratio: float  # Variance of errors / variance of benchmark


@dataclass
class BenchmarkOutput:
    """Output from benchmark model"""

    bond_id: str
    predicted_value: float
    confidence: float
    methodology: str


class BenchmarkGenerator:
    """
    Generates benchmark predictions simulating leading financial firms' models

    This simulates what Bloomberg, Aladdin, Goldman Sachs models would produce
    using more sophisticated valuation methods as proxies
    """

    def __init__(self, valuator: BondValuator = None):
        """
        Initialize benchmark generator

        Args:
            valuator: Optional BondValuator instance. If None, gets from container.
        """
        if valuator is None:
            from bondtrader.core.container import get_container

            self.valuator = get_container().get_valuator()
        else:
            self.valuator = valuator

        # Industry-specific adjustment factors (based on research)
        # These simulate how different firms adjust standard valuations
        self.bloomberg_factor = 1.002  # Bloomberg tends to be slightly conservative
        self.aladdin_factor = 0.998  # Aladdin adjusts for liquidity premium
        self.goldman_factor = 1.001  # Goldman includes credit research premium
        self.jpmorgan_factor = 0.999  # JPMorgan includes transaction costs

        # Volatility and uncertainty adjustments
        self.market_volatility_factor = 1.01  # Accounts for market microstructure

    def generate_bloomberg_benchmark(self, bond: Bond) -> BenchmarkOutput:
        """
        Generate Bloomberg Terminal-style benchmark
        Bloomberg uses sophisticated pricing with market data integration
        """
        # Calculate base fair value
        base_fair_value = self.valuator.calculate_fair_value(bond)

        # Bloomberg adjustments:
        # 1. Credit spread adjustment (more granular)
        credit_spread = self._get_enhanced_credit_spread(bond)
        required_yield = self.valuator.risk_free_rate + credit_spread

        # 2. Market microstructure adjustment
        liquidity_adjustment = self._calculate_liquidity_adjustment(bond)

        # 3. Duration-based volatility adjustment
        ytm = self.valuator.calculate_yield_to_maturity(bond)
        duration = self.valuator.calculate_duration(bond, ytm)
        volatility_adjustment = 1.0 + (duration * 0.0001)  # Duration impact

        # Bloomberg benchmark value
        benchmark_value = base_fair_value * self.bloomberg_factor * liquidity_adjustment * volatility_adjustment

        # Confidence based on bond characteristics
        confidence = self._calculate_confidence(bond, base_fair_value)

        return BenchmarkOutput(
            bond_id=bond.bond_id,
            predicted_value=benchmark_value,
            confidence=confidence,
            methodology="bloomberg",
        )

    def generate_aladdin_benchmark(self, bond: Bond) -> BenchmarkOutput:
        """
        Generate BlackRock Aladdin-style benchmark
        Aladdin emphasizes risk-adjusted valuation and liquidity
        """
        base_fair_value = self.valuator.calculate_fair_value(bond)

        # Aladdin adjustments:
        # 1. Risk-adjusted discount rate
        risk_adjustment = self._calculate_risk_adjustment(bond)

        # 2. Liquidity premium (Aladdin is strong on liquidity metrics)
        liquidity_premium = self._calculate_liquidity_premium(bond)

        # 3. Portfolio-level adjustment (simulated)
        portfolio_adjustment = 0.999  # Slight discount for portfolio context

        # Aladdin benchmark value
        benchmark_value = base_fair_value * self.aladdin_factor * risk_adjustment * liquidity_premium * portfolio_adjustment

        confidence = self._calculate_confidence(bond, base_fair_value)

        return BenchmarkOutput(
            bond_id=bond.bond_id,
            predicted_value=benchmark_value,
            confidence=confidence,
            methodology="aladdin",
        )

    def generate_goldman_benchmark(self, bond: Bond) -> BenchmarkOutput:
        """
        Generate Goldman Sachs-style benchmark
        Goldman uses proprietary credit research and market intelligence
        """
        base_fair_value = self.valuator.calculate_fair_value(bond)

        # Goldman adjustments:
        # 1. Credit research premium (Goldman has strong credit research)
        credit_premium = self._calculate_credit_research_premium(bond)

        # 2. Market intelligence adjustment
        market_intelligence = 1.0005  # Slight premium for proprietary insights

        # 3. Structured product expertise (if applicable)
        structure_adjustment = 1.0
        if bond.callable or bond.convertible:
            structure_adjustment = 1.001  # Premium for structured product expertise

        # Goldman benchmark value
        benchmark_value = base_fair_value * self.goldman_factor * credit_premium * market_intelligence * structure_adjustment

        confidence = self._calculate_confidence(bond, base_fair_value)

        return BenchmarkOutput(
            bond_id=bond.bond_id,
            predicted_value=benchmark_value,
            confidence=confidence,
            methodology="goldman",
        )

    def generate_jpmorgan_benchmark(self, bond: Bond) -> BenchmarkOutput:
        """
        Generate JPMorgan-style benchmark
        JPMorgan emphasizes transaction costs and execution
        """
        base_fair_value = self.valuator.calculate_fair_value(bond)

        # JPMorgan adjustments:
        # 1. Transaction cost adjustment
        transaction_cost = self._calculate_transaction_cost_adjustment(bond)

        # 2. Execution risk adjustment
        execution_risk = 0.9995  # Slight discount for execution risk

        # 3. Liquidity cost
        liquidity_cost = self._calculate_liquidity_cost(bond)

        # JPMorgan benchmark value
        benchmark_value = base_fair_value * self.jpmorgan_factor * transaction_cost * execution_risk * liquidity_cost

        confidence = self._calculate_confidence(bond, base_fair_value)

        return BenchmarkOutput(
            bond_id=bond.bond_id,
            predicted_value=benchmark_value,
            confidence=confidence,
            methodology="jpmorgan",
        )

    def generate_consensus_benchmark(self, bond: Bond) -> BenchmarkOutput:
        """
        Generate consensus benchmark (average of leading firms)
        This is the most robust benchmark for drift detection
        """
        benchmarks = [
            self.generate_bloomberg_benchmark(bond),
            self.generate_aladdin_benchmark(bond),
            self.generate_goldman_benchmark(bond),
            self.generate_jpmorgan_benchmark(bond),
        ]

        # Weighted average (can weight by confidence)
        weights = [b.confidence for b in benchmarks]
        total_weight = sum(weights)

        if total_weight > 0:
            consensus_value = sum(b.predicted_value * w for b, w in zip(benchmarks, weights)) / total_weight
            avg_confidence = np.mean([b.confidence for b in benchmarks])
        else:
            consensus_value = np.mean([b.predicted_value for b in benchmarks])
            avg_confidence = 0.8

        return BenchmarkOutput(
            bond_id=bond.bond_id,
            predicted_value=consensus_value,
            confidence=avg_confidence,
            methodology="consensus",
        )

    def _get_enhanced_credit_spread(self, bond: Bond) -> float:
        """Enhanced credit spread calculation (more granular than standard)"""
        base_spread = self.valuator._get_credit_spread(bond.credit_rating)

        # Additional adjustments based on bond characteristics
        maturity_adjustment = 1.0 + (bond.time_to_maturity - 5.0) * 0.0001  # Longer = wider
        type_adjustment = 1.2 if bond.bond_type.value == "High Yield" else 1.0

        return base_spread * maturity_adjustment * type_adjustment

    def _calculate_liquidity_adjustment(self, bond: Bond) -> float:
        """Calculate liquidity adjustment factor"""
        # Factors affecting liquidity:
        # - Credit rating (higher = more liquid)
        # - Maturity (shorter = more liquid)
        # - Issue size (larger = more liquid, approximated by face value)

        rating_factor = {
            "AAA": 1.002,
            "AA": 1.001,
            "A": 1.000,
            "BBB": 0.999,
            "BB": 0.998,
            "B": 0.997,
            "CCC": 0.995,
        }.get(bond.credit_rating[:3] if len(bond.credit_rating) >= 3 else "BBB", 0.999)

        maturity_factor = 1.0 + min(0, (5.0 - bond.time_to_maturity) * 0.0002)
        size_factor = 1.0 + min(0.002, (bond.face_value - 1000) / 1000000)

        return rating_factor * maturity_factor * size_factor

    def _calculate_risk_adjustment(self, bond: Bond) -> float:
        """Calculate risk adjustment factor"""
        ytm = self.valuator.calculate_yield_to_maturity(bond)
        duration = self.valuator.calculate_duration(bond, ytm)

        # Higher duration = higher risk = lower value
        risk_factor = 1.0 - (duration - 5.0) * 0.0001
        return max(0.995, min(1.005, risk_factor))

    def _calculate_liquidity_premium(self, bond: Bond) -> float:
        """Calculate liquidity premium (discount for illiquid bonds)"""
        # Based on credit rating and maturity
        base_premium = self._calculate_liquidity_adjustment(bond)
        return base_premium

    def _calculate_credit_research_premium(self, bond: Bond) -> float:
        """Calculate credit research premium (Goldman specialty)"""
        # Higher credit research quality for investment grade
        if bond.credit_rating.startswith(("AAA", "AA", "A")):
            return 1.001
        elif bond.credit_rating.startswith("BBB"):
            return 1.0005
        else:
            return 0.9995  # Less premium for high yield

    def _calculate_transaction_cost_adjustment(self, bond: Bond) -> float:
        """Calculate transaction cost adjustment"""
        # Base transaction cost: 0.05% to 0.5% depending on liquidity
        base_cost = 0.0005 if bond.credit_rating.startswith(("AAA", "AA", "A")) else 0.002

        # Size impact
        size_impact = max(0, (10000 - bond.face_value) / 100000)
        total_cost = base_cost + size_impact * 0.001

        return 1.0 - total_cost

    def _calculate_liquidity_cost(self, bond: Bond) -> float:
        """Calculate liquidity cost"""
        return self._calculate_transaction_cost_adjustment(bond)

    def _calculate_confidence(self, bond: Bond, fair_value: float) -> float:
        """Calculate confidence in benchmark prediction"""
        # Confidence based on:
        # - Data quality (higher for investment grade)
        # - Market conditions (simplified)
        # - Bond complexity

        base_confidence = 0.9 if bond.credit_rating.startswith(("AAA", "AA", "A")) else 0.8

        # Adjust for complexity
        complexity_penalty = 0.05 if (bond.callable or bond.convertible) else 0.0

        # Adjust for maturity (very long or very short = less confidence)
        maturity_penalty = abs(bond.time_to_maturity - 10.0) / 100.0 * 0.05

        confidence = base_confidence - complexity_penalty - maturity_penalty
        return max(0.7, min(0.95, confidence))


class DriftDetector:
    """
    Detects and measures drift between model predictions and benchmark outputs
    """

    def __init__(self, benchmark_generator: BenchmarkGenerator = None):
        """Initialize drift detector"""
        self.benchmark_generator = benchmark_generator if benchmark_generator else BenchmarkGenerator()

    def calculate_drift(
        self,
        bonds: List[Bond],
        model_predictions: List[float],
        benchmark_methodology: str = "consensus",
    ) -> DriftMetrics:
        """
        Calculate drift metrics between model predictions and benchmarks

        Args:
            bonds: List of bonds
            model_predictions: Model predictions (prices or values)
            benchmark_methodology: Which benchmark to use ('bloomberg', 'aladdin', 'goldman', 'jpmorgan', 'consensus')

        Returns:
            DriftMetrics object with all drift statistics
        """
        if len(bonds) != len(model_predictions):
            raise ValueError("Bonds and predictions must have same length")

        # Generate benchmark predictions
        benchmark_methods = {
            "bloomberg": self.benchmark_generator.generate_bloomberg_benchmark,
            "aladdin": self.benchmark_generator.generate_aladdin_benchmark,
            "goldman": self.benchmark_generator.generate_goldman_benchmark,
            "jpmorgan": self.benchmark_generator.generate_jpmorgan_benchmark,
            "consensus": self.benchmark_generator.generate_consensus_benchmark,
        }

        if benchmark_methodology not in benchmark_methods:
            raise ValueError(f"Unknown benchmark methodology: {benchmark_methodology}")

        benchmark_method = benchmark_methods[benchmark_methodology]
        benchmark_predictions = [benchmark_method(bond).predicted_value for bond in bonds]

        # Calculate errors
        errors = np.array(model_predictions) - np.array(benchmark_predictions)
        absolute_errors = np.abs(errors)

        # Calculate metrics
        mae = np.mean(absolute_errors)
        mse = np.mean(errors**2)
        rmse = np.sqrt(mse)

        # MAPE (handle division by zero)
        benchmark_array = np.array(benchmark_predictions)
        non_zero_mask = benchmark_array != 0
        if np.any(non_zero_mask):
            mape = np.mean(np.abs(errors[non_zero_mask] / benchmark_array[non_zero_mask])) * 100
        else:
            mape = float("inf")

        max_abs_error = np.max(absolute_errors)

        # Correlation
        if np.std(model_predictions) > 0 and np.std(benchmark_predictions) > 0:
            correlation = np.corrcoef(model_predictions, benchmark_predictions)[0, 1]
        else:
            correlation = 0.0

        # Bias (systematic error)
        bias = np.mean(errors)

        # Variance ratio (how much more/less variable are our errors than benchmark)
        error_variance = np.var(errors)
        benchmark_variance = np.var(benchmark_predictions)
        variance_ratio = error_variance / benchmark_variance if benchmark_variance > 0 else float("inf")

        # Drift score (0-1, normalized composite metric)
        # Normalize RMSE relative to benchmark range
        benchmark_range = np.max(benchmark_predictions) - np.min(benchmark_predictions)
        normalized_rmse = rmse / benchmark_range if benchmark_range > 0 else 1.0

        # Drift score combines: normalized RMSE, correlation (inverted), and bias
        drift_score = min(
            1.0,
            (
                0.4 * min(1.0, normalized_rmse)
                + 0.3 * (1.0 - max(0.0, correlation))
                + 0.3 * min(1.0, abs(bias) / (benchmark_range * 0.01) if benchmark_range > 0 else 1.0)
            ),
        )

        return DriftMetrics(
            mean_absolute_error=mae,
            mean_squared_error=mse,
            root_mean_squared_error=rmse,
            mean_absolute_percentage_error=mape,
            max_absolute_error=max_abs_error,
            drift_score=drift_score,
            correlation=correlation,
            bias=bias,
            variance_ratio=variance_ratio,
        )

    def calculate_drift_by_regime(
        self,
        bonds: List[Bond],
        model_predictions: List[float],
        regimes: List[str],
        benchmark_methodology: str = "consensus",
    ) -> Dict[str, DriftMetrics]:
        """
        Calculate drift metrics by market regime

        Args:
            bonds: List of bonds
            model_predictions: Model predictions
            regimes: List of regime labels (e.g., 'normal', 'bull', 'bear')
            benchmark_methodology: Which benchmark to use

        Returns:
            Dictionary mapping regime names to DriftMetrics
        """
        regime_drifts = {}

        unique_regimes = list(set(regimes))

        for regime in unique_regimes:
            regime_mask = [r == regime for r in regimes]
            regime_bonds = [b for b, m in zip(bonds, regime_mask) if m]
            regime_predictions = [p for p, m in zip(model_predictions, regime_mask) if m]

            if len(regime_bonds) > 0:
                drift = self.calculate_drift(regime_bonds, regime_predictions, benchmark_methodology)
                regime_drifts[regime] = drift

        return regime_drifts


class ModelTuner:
    """
    Tunes models to minimize drift against benchmarks
    """

    def __init__(self, drift_detector: DriftDetector = None):
        """Initialize model tuner"""
        self.drift_detector = drift_detector if drift_detector else DriftDetector()

    def tune_model_for_minimal_drift(
        self,
        model,
        bonds: List[Bond],
        validation_bonds: List[Bond],
        benchmark_methodology: str = "consensus",
        tuning_params: Optional[Dict] = None,
        n_iter: int = 25,
    ) -> Dict:
        """
        Tune model parameters to minimize drift using RandomizedSearchCV

        More efficient than exhaustive grid search - samples parameter space intelligently

        Args:
            model: Model to tune (must have train/predict methods)
            bonds: Training bonds
            validation_bonds: Validation bonds for drift measurement
            benchmark_methodology: Which benchmark to use
            tuning_params: Parameters to tune (if None, uses defaults)
            n_iter: Number of parameter settings sampled (default: 25)

        Returns:
            Dictionary with tuning results
        """
        if tuning_params is None:
            # Default tuning parameters for common models
            tuning_params = {
                "learning_rate": [0.01, 0.05, 0.1],
                "regularization": [0.001, 0.01, 0.1],
                "n_estimators": [50, 100, 200],
            }

        # Limit exhaustive search to small parameter spaces
        from itertools import product

        param_names = list(tuning_params.keys())
        param_values = list(tuning_params.values())
        total_combinations = 1
        for values in param_values:
            total_combinations *= len(values)

        # Use RandomizedSearchCV for large parameter spaces, grid search for small ones
        if total_combinations <= 50:
            # Small parameter space - use grid search
            param_combinations = list(product(*param_values))
            print(f"Tuning model with {len(param_combinations)} parameter combinations (grid search)...")
            use_randomized = False
        else:
            # Large parameter space - use randomized search
            print(f"Tuning model with {n_iter} random samples from {total_combinations} combinations (randomized search)...")
            use_randomized = True
            # Limit iterations to reasonable number
            n_iter = min(n_iter, 100)

        best_drift_score = float("inf")
        best_params = None
        best_model = None
        results = []

        if use_randomized:
            # Randomized search
            import random

            random.seed(42)
            np.random.seed(42)

            # Sample random combinations
            param_combinations = []
            for _ in range(n_iter):
                combo = tuple(np.random.choice(values) for values in param_values)
                param_combinations.append(combo)
        else:
            # Exhaustive grid search
            param_combinations = list(product(*param_values))

        for param_combo in param_combinations:
            params = dict(zip(param_names, param_combo))

            # Update model with parameters
            model_copy = self._clone_model(model)
            try:
                self._set_model_params(model_copy, params)

                # Train model
                if hasattr(model_copy, "train"):
                    model_copy.train(bonds)
                elif hasattr(model_copy, "fit"):
                    # For sklearn-style models
                    from bondtrader.ml.ml_adjuster_enhanced import EnhancedMLBondAdjuster

                    if isinstance(model_copy, EnhancedMLBondAdjuster):
                        model_copy.train_with_tuning(bonds, tune_hyperparameters=False)
                        # Apply parameters manually if possible
                        if hasattr(model_copy, "model") and model_copy.model is not None:
                            # Try to update model parameters
                            for param, value in params.items():
                                if hasattr(model_copy.model, param):
                                    setattr(model_copy.model, param, value)

                # Predict on validation set
                predictions = []
                for bond in validation_bonds:
                    if hasattr(model_copy, "predict_adjusted_value"):
                        pred = model_copy.predict_adjusted_value(bond)
                        value = pred.get(
                            "ml_adjusted_value",
                            pred.get("ml_adjusted_fair_value", bond.current_price),
                        )
                    elif hasattr(model_copy, "predict"):
                        # For sklearn models
                        # Need to extract features first
                        from bondtrader.core.container import get_container

                        valuator = get_container().get_valuator()
                        fair_value = valuator.calculate_fair_value(bond)
                        # Simplified feature extraction
                        features = np.array([[bond.coupon_rate, bond.time_to_maturity]])
                        pred = model_copy.predict(features)[0]
                        value = fair_value * pred
                    else:
                        value = bond.current_price

                    predictions.append(value)

                # Calculate drift
                drift_metrics = self.drift_detector.calculate_drift(validation_bonds, predictions, benchmark_methodology)

                results.append({"params": params, "drift_metrics": drift_metrics})

                # Update best if better
                if drift_metrics.drift_score < best_drift_score:
                    best_drift_score = drift_metrics.drift_score
                    best_params = params
                    best_model = model_copy

            except Exception as e:
                logger.warning(f"Error with params {params}: {e}")
                continue

        return {
            "best_params": best_params,
            "best_drift_score": best_drift_score,
            "best_model": best_model,
            "all_results": results,
            "tuning_summary": self._summarize_tuning_results(results),
            "method": "randomized_search" if use_randomized else "grid_search",
            "total_combinations": total_combinations,
            "combinations_tested": len(param_combinations),
        }

    def _clone_model(self, model):
        """Clone model for tuning (avoid mutating original)"""
        import copy

        try:
            return copy.deepcopy(model)
        except (TypeError, AttributeError, ValueError) as e:
            # Fallback: create new instance if deepcopy fails
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"Deepcopy failed for model, using fallback: {e}")
            return type(model)(model.valuator if hasattr(model, "valuator") else None)

    def _set_model_params(self, model, params: Dict):
        """Set parameters on model"""
        for param, value in params.items():
            if hasattr(model, param):
                setattr(model, param, value)

    def _summarize_tuning_results(self, results: List[Dict]) -> Dict:
        """Summarize tuning results"""
        if not results:
            return {}

        drift_scores = [r["drift_metrics"].drift_score for r in results]

        return {
            "num_combinations_tested": len(results),
            "min_drift_score": min(drift_scores),
            "max_drift_score": max(drift_scores),
            "mean_drift_score": np.mean(drift_scores),
            "std_drift_score": np.std(drift_scores),
            "improvement_potential": (max(drift_scores) - min(drift_scores)),
        }


def compare_models_against_benchmarks(
    models: Dict[str, any], bonds: List[Bond], benchmark_methodology: str = "consensus"
) -> Dict[str, DriftMetrics]:
    """
    Compare multiple models against benchmarks

    Args:
        models: Dictionary mapping model names to model objects
        bonds: List of bonds to evaluate on
        benchmark_methodology: Which benchmark to use

    Returns:
        Dictionary mapping model names to DriftMetrics
    """
    detector = DriftDetector()
    results = {}

    for model_name, model in models.items():
        print(f"Evaluating {model_name}...")

        predictions = []
        for bond in bonds:
            try:
                if hasattr(model, "predict_adjusted_value"):
                    pred = model.predict_adjusted_value(bond)
                    value = pred.get("ml_adjusted_value", pred.get("ml_adjusted_fair_value", bond.current_price))
                elif hasattr(model, "predict"):
                    from bondtrader.core.container import get_container

                    valuator = get_container().get_valuator()
                    fair_value = valuator.calculate_fair_value(bond)
                    features = np.array([[bond.coupon_rate, bond.time_to_maturity]])
                    pred_value = model.predict(features)[0]
                    value = fair_value * pred_value
                else:
                    # Fallback to current price
                    value = bond.current_price

                predictions.append(value)
            except Exception as e:
                print(f"  Error predicting for bond {bond.bond_id}: {e}")
                predictions.append(bond.current_price)

        drift = detector.calculate_drift(bonds, predictions, benchmark_methodology)
        results[model_name] = drift

    return results
