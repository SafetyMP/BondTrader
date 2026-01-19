"""
Enhanced Model Explainability with SHAP Integration
Comprehensive explainability for ML models

Industry Best Practices:
- SHAP values for local and global explanations
- Feature importance analysis
- Prediction explanations
- Model interpretability reports
"""

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from bondtrader.config import get_config
from bondtrader.core.bond_models import Bond
from bondtrader.utils.utils import logger

# SHAP integration
try:
    import shap

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    logger.warning("SHAP not available. Install with: pip install shap")


@dataclass
class Explanation:
    """Model explanation for a single prediction"""

    bond_id: str
    predicted_value: float
    feature_contributions: Dict[str, float]
    top_features: List[Tuple[str, float]]  # (feature_name, contribution)
    shap_values: Optional[np.ndarray] = None
    base_value: Optional[float] = None


@dataclass
class GlobalExplanation:
    """Global model explanation"""

    feature_importance: Dict[str, float]
    feature_importance_std: Dict[str, float]
    summary_statistics: Dict[str, Any]
    shap_summary: Optional[Dict[str, Any]] = None


class ModelExplainer:
    """
    Enhanced model explainability with SHAP integration

    Industry Best Practices:
    - SHAP values for local explanations
    - Feature importance for global explanations
    - Prediction explanations
    - Model interpretability reports
    """

    def __init__(
        self,
        model,
        feature_names: List[str],
        explainer_type: str = "tree",  # tree, linear, kernel, deep
        background_data: Optional[np.ndarray] = None,
    ):
        """
        Initialize model explainer

        Args:
            model: Trained model to explain
            feature_names: List of feature names
            explainer_type: Type of SHAP explainer
            background_data: Background data for SHAP (optional)
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer_type = explainer_type
        self.background_data = background_data

        # SHAP explainer
        self.shap_explainer = None
        self._initialize_shap_explainer()

    def _initialize_shap_explainer(self):
        """Initialize SHAP explainer based on model type"""
        if not HAS_SHAP:
            logger.warning("SHAP not available, using fallback explanations")
            return

        try:
            model_type = type(self.model).__name__.lower()

            if "randomforest" in model_type or "gradientboosting" in model_type or "xgboost" in model_type:
                # Tree-based models
                self.shap_explainer = shap.TreeExplainer(self.model)
                logger.info("Initialized TreeExplainer for SHAP")
            elif "linear" in model_type or "logistic" in model_type:
                # Linear models
                if self.background_data is not None:
                    self.shap_explainer = shap.LinearExplainer(self.model, self.background_data)
                else:
                    self.shap_explainer = shap.LinearExplainer(self.model, np.zeros((1, len(self.feature_names))))
                logger.info("Initialized LinearExplainer for SHAP")
            else:
                # Generic explainer (kernel or permutation)
                if self.background_data is not None:
                    self.shap_explainer = shap.KernelExplainer(
                        self.model.predict, self.background_data[:100]  # Sample for speed
                    )
                else:
                    logger.warning("No background data provided, using fallback explanations")
            return True
        except Exception as e:
            logger.warning(f"Failed to initialize SHAP explainer: {e}")
            return False

    def explain_prediction(
        self,
        bond: Bond,
        features: np.ndarray,
        use_shap: bool = True,
    ) -> Explanation:
        """
        Explain a single prediction

        Args:
            bond: Bond object
            features: Feature vector for the bond
            use_shap: Whether to use SHAP (if available)

        Returns:
            Explanation object
        """
        # Get prediction
        if hasattr(self.model, "predict_adjusted_value"):
            result = self.model.predict_adjusted_value(bond)
            predicted_value = result.get("ml_adjusted_value", bond.current_price)
        else:
            predicted_value = self.model.predict(features.reshape(1, -1))[0]

        # Get feature contributions
        feature_contributions = {}
        shap_values = None
        base_value = None

        if use_shap and self.shap_explainer is not None:
            try:
                # Compute SHAP values
                shap_values_array = self.shap_explainer.shap_values(features.reshape(1, -1))

                # Handle different SHAP output formats
                if isinstance(shap_values_array, list):
                    shap_values_array = shap_values_array[0]  # For multi-output

                if len(shap_values_array.shape) > 1:
                    shap_values_array = shap_values_array[0]  # Single prediction

                shap_values = shap_values_array

                # Get base value
                if hasattr(self.shap_explainer, "expected_value"):
                    base_value = float(self.shap_explainer.expected_value)
                    if isinstance(base_value, (list, np.ndarray)):
                        base_value = float(base_value[0])

                # Create feature contributions dictionary
                for i, feature_name in enumerate(self.feature_names):
                    if i < len(shap_values):
                        feature_contributions[feature_name] = float(shap_values[i])

            except Exception as e:
                logger.warning(f"SHAP explanation failed: {e}, using fallback")
                feature_contributions = self._fallback_explanation(features)
        else:
            # Fallback explanation using feature importance
            feature_contributions = self._fallback_explanation(features)

        # Sort features by absolute contribution
        top_features = sorted(feature_contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:10]  # Top 10 features

        return Explanation(
            bond_id=bond.bond_id,
            predicted_value=predicted_value,
            feature_contributions=feature_contributions,
            top_features=top_features,
            shap_values=shap_values,
            base_value=base_value,
        )

    def _fallback_explanation(self, features: np.ndarray) -> Dict[str, float]:
        """Fallback explanation using feature importance"""
        feature_contributions = {}

        # Use feature importance if available
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            feature_values = features.flatten()

            for i, feature_name in enumerate(self.feature_names):
                if i < len(importances) and i < len(feature_values):
                    # Contribution = importance * (value - mean_value)
                    # Simplified: use importance * normalized value
                    contribution = float(importances[i] * feature_values[i])
                    feature_contributions[feature_name] = contribution
        else:
            # Equal contributions if no importance available
            for i, feature_name in enumerate(self.feature_names):
                if i < len(features.flatten()):
                    feature_contributions[feature_name] = float(features.flatten()[i] / len(self.feature_names))

        return feature_contributions

    def explain_global(
        self,
        sample_data: np.ndarray,
        sample_size: int = 100,
    ) -> GlobalExplanation:
        """
        Explain model globally (feature importance)

        Args:
            sample_data: Sample of data to compute global explanations
            sample_size: Number of samples to use for SHAP

        Returns:
            GlobalExplanation object
        """
        # Use feature importance from model
        feature_importance = {}
        feature_importance_std = {}

        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            for i, feature_name in enumerate(self.feature_names):
                if i < len(importances):
                    feature_importance[feature_name] = float(importances[i])
                    feature_importance_std[feature_name] = 0.0  # Standard deviation not available
        else:
            # Equal importance if not available
            equal_importance = 1.0 / len(self.feature_names)
            for feature_name in self.feature_names:
                feature_importance[feature_name] = equal_importance
                feature_importance_std[feature_name] = 0.0

        # Compute SHAP summary if available
        shap_summary = None
        if HAS_SHAP and self.shap_explainer is not None and len(sample_data) > 0:
            try:
                # Sample data for efficiency
                sample_indices = np.random.choice(len(sample_data), min(sample_size, len(sample_data)), replace=False)
                sample = sample_data[sample_indices]

                # Compute SHAP values
                shap_values = self.shap_explainer.shap_values(sample)

                if isinstance(shap_values, list):
                    shap_values = shap_values[0]

                # Compute summary statistics
                shap_summary = {
                    "mean_abs_shap": np.abs(shap_values).mean(axis=0).tolist(),
                    "std_shap": shap_values.std(axis=0).tolist(),
                }

            except Exception as e:
                logger.warning(f"SHAP global explanation failed: {e}")

        # Summary statistics
        summary_statistics = {
            "n_features": len(self.feature_names),
            "sample_size": len(sample_data),
            "explainer_type": self.explainer_type,
        }

        return GlobalExplanation(
            feature_importance=feature_importance,
            feature_importance_std=feature_importance_std,
            summary_statistics=summary_statistics,
            shap_summary=shap_summary,
        )

    def generate_explanation_report(
        self,
        bond: Bond,
        features: np.ndarray,
        output_path: str = None,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive explanation report

        Args:
            bond: Bond object
            features: Feature vector
            output_path: Path to save report (optional)

        Returns:
            Explanation report dictionary
        """
        explanation = self.explain_prediction(bond, features)

        report = {
            "bond_id": bond.bond_id,
            "timestamp": datetime.now().isoformat(),
            "prediction": {
                "predicted_value": explanation.predicted_value,
                "base_value": explanation.base_value,
            },
            "feature_contributions": explanation.feature_contributions,
            "top_features": [{"feature": name, "contribution": value} for name, value in explanation.top_features],
            "shap_available": explanation.shap_values is not None,
        }

        if output_path:
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Explanation report saved to {output_path}")

        return report

    def visualize_explanation(
        self,
        bond: Bond,
        features: np.ndarray,
        output_path: str = None,
    ):
        """
        Visualize explanation (SHAP plots)

        Args:
            bond: Bond object
            features: Feature vector
            output_path: Path to save visualization (optional)
        """
        if not HAS_SHAP or self.shap_explainer is None:
            logger.warning("SHAP not available for visualization")
            return

        try:
            # Compute SHAP values
            shap_values = self.shap_explainer.shap_values(features.reshape(1, -1))

            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            if len(shap_values.shape) > 1:
                shap_values = shap_values[0]

            # Create SHAP explanation object
            explanation = shap.Explanation(
                values=shap_values.reshape(1, -1),
                base_values=(self.shap_explainer.expected_value if hasattr(self.shap_explainer, "expected_value") else 0.0),
                data=features.reshape(1, -1),
                feature_names=self.feature_names,
            )

            # Waterfall plot
            if output_path:
                shap.plots.waterfall(explanation[0], show=False)
                import matplotlib.pyplot as plt

                plt.savefig(output_path, bbox_inches="tight")
                plt.close()
                logger.info(f"SHAP visualization saved to {output_path}")
            else:
                shap.plots.waterfall(explanation[0])

        except Exception as e:
            logger.error(f"Failed to create SHAP visualization: {e}", exc_info=True)


def create_model_explainer(
    model,
    feature_names: List[str],
    background_data: Optional[np.ndarray] = None,
) -> ModelExplainer:
    """
    Convenience function to create model explainer

    Args:
        model: Trained model
        feature_names: Feature names
        background_data: Background data for SHAP

    Returns:
        ModelExplainer instance
    """
    return ModelExplainer(
        model=model,
        feature_names=feature_names,
        background_data=background_data,
    )
