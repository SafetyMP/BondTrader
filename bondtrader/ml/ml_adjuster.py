"""
Machine Learning Model for Bond Price Adjustments
Uses regression to predict fair value adjustments based on market factors

NOTE: This module now uses the unified MLBondAdjuster implementation.
The old implementation has been consolidated to eliminate duplication.
All functionality is preserved through the unified implementation.
"""

import warnings
from typing import Optional

from bondtrader.core.bond_valuation import BondValuator

# Import unified implementation
from bondtrader.ml.ml_adjuster_unified import MLBondAdjuster as UnifiedMLBondAdjuster


# Backward compatibility: MLBondAdjuster now uses unified implementation
# with feature_level="basic" by default
class MLBondAdjuster(UnifiedMLBondAdjuster):
    """
    ML model to adjust bond valuations (backward compatible wrapper)

    This class now uses the unified MLBondAdjuster with feature_level="basic"
    to maintain backward compatibility with existing code.

    For enhanced features, use feature_level="enhanced" or "advanced" directly
    on the unified implementation, or use EnhancedMLBondAdjuster/AdvancedMLBondAdjuster
    aliases from the ml module.
    """

    def __init__(self, model_type: str = "random_forest", valuator: Optional[BondValuator] = None):
        """
        Initialize ML adjuster (backward compatible wrapper)

        This is now a wrapper around the unified MLBondAdjuster with feature_level="basic"

        Args:
            model_type: 'random_forest', 'gradient_boosting', 'xgboost', 'lightgbm', or 'catboost'
            valuator: Optional BondValuator instance. If None, gets from container.
        """
        super().__init__(model_type=model_type, feature_level="basic", valuator=valuator, use_ensemble=False)
