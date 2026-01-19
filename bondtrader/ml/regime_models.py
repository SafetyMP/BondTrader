"""
Regime-Dependent Models Module
Market regime detection and adaptive models
More sophisticated than static models
"""

from typing import Dict, List, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from bondtrader.core.bond_models import Bond
from bondtrader.core.bond_valuation import BondValuator
from bondtrader.utils.utils import logger


class RegimeDetector:
    """
    Market regime detection and regime-dependent modeling
    Detects bull/bear, high/low volatility regimes
    """

    def __init__(self, valuator: BondValuator = None):
        """
        Initialize regime detector

        Args:
            valuator: Optional BondValuator instance. If None, gets from container.
        """
        if valuator is None:
            from bondtrader.core.container import get_container

            self.valuator = get_container().get_valuator()
        else:
            self.valuator = valuator
        self.regimes = None
        self.regime_model = None

    def detect_regimes(self, bonds: List[Bond], num_regimes: int = 3, method: str = "kmeans") -> Dict:
        """
        Detect market regimes from bond data

        Regimes could be:
        - High volatility / Low volatility
        - Bull / Bear / Neutral
        - Tight spreads / Wide spreads

        Args:
            bonds: List of bonds
            num_regimes: Number of regimes to detect
            method: 'kmeans' or 'gmm'

        Returns:
            Regime detection results
        """
        # Create regime features
        features = []
        for bond in bonds:
            ytm = self.valuator.calculate_yield_to_maturity(bond)
            spread = self.valuator._get_credit_spread(bond.credit_rating)
            duration = self.valuator.calculate_duration(bond, ytm)

            # Features for regime detection
            feature_vector = [
                ytm * 100,  # Yield level
                spread * 10000,  # Credit spread (bps)
                duration,  # Duration
                bond.current_price / bond.face_value,  # Price to par
            ]
            features.append(feature_vector)

        X = np.array(features)

        # Detect regimes
        if method == "kmeans":
            self.regime_model = KMeans(n_clusters=num_regimes, random_state=42, n_init=10)
            regime_labels = self.regime_model.fit_predict(X)
        else:  # GMM
            self.regime_model = GaussianMixture(n_components=num_regimes, random_state=42)
            regime_labels = self.regime_model.fit_predict(X)

        # Analyze each regime
        regime_analysis = {}
        for regime_id in range(num_regimes):
            regime_bonds = [b for b, label in zip(bonds, regime_labels) if label == regime_id]

            if len(regime_bonds) > 0:
                avg_ytm = np.mean([self.valuator.calculate_yield_to_maturity(b) for b in regime_bonds])
                avg_spread = np.mean([self.valuator._get_credit_spread(b.credit_rating) for b in regime_bonds])
                avg_duration = np.mean([self.valuator.calculate_duration(b) for b in regime_bonds])

                regime_analysis[f"Regime {regime_id}"] = {
                    "num_bonds": len(regime_bonds),
                    "avg_ytm": avg_ytm * 100,
                    "avg_spread_bps": avg_spread * 10000,
                    "avg_duration": avg_duration,
                    "regime_type": self._classify_regime(avg_ytm, avg_spread, avg_duration),
                }

        self.regimes = regime_labels

        return {
            "regime_labels": regime_labels.tolist(),
            "num_regimes": num_regimes,
            "regime_analysis": regime_analysis,
            "method": method,
        }

    def _classify_regime(self, avg_ytm: float, avg_spread: float, avg_duration: float) -> str:
        """Classify regime type based on characteristics"""
        if avg_spread > 0.05:  # High spreads
            return "High Stress / Wide Spreads"
        elif avg_ytm > self.valuator.risk_free_rate + 0.02:
            return "High Yield Environment"
        elif avg_duration > 7:
            return "Long Duration"
        else:
            return "Normal / Tight Spreads"

    def regime_dependent_pricing(self, bond: Bond, current_regime: Optional[int] = None) -> Dict:
        """
        Price bond using regime-dependent models

        Different pricing models for different market regimes

        Args:
            bond: Bond object
            current_regime: Current regime (if None, detects)

        Returns:
            Regime-dependent pricing
        """
        if current_regime is None:
            # Detect regime for this bond
            if self.regime_model is None:
                raise ValueError("Regimes not detected. Run detect_regimes() first.")

            ytm = self.valuator.calculate_yield_to_maturity(bond)
            spread = self.valuator._get_credit_spread(bond.credit_rating)
            duration = self.valuator.calculate_duration(bond, ytm)

            features = np.array([[ytm * 100, spread * 10000, duration, bond.current_price / bond.face_value]])
            current_regime = self.regime_model.predict(features)[0]

        # Base fair value
        base_fv = self.valuator.calculate_fair_value(bond)

        # Regime adjustments
        # High stress regimes: wider spreads, lower prices
        # Normal regimes: tighter spreads, higher prices
        regime_adjustments = {
            0: 0.02,  # Normal: +2% adjustment
            1: -0.01,  # Moderate: -1% adjustment
            2: -0.03,  # High stress: -3% adjustment
        }

        adjustment = regime_adjustments.get(current_regime % 3, 0.0)
        regime_adjusted_value = base_fv * (1 + adjustment)

        return {
            "current_regime": int(current_regime),
            "base_fair_value": base_fv,
            "regime_adjustment": adjustment,
            "regime_adjusted_value": regime_adjusted_value,
            "adjustment_pct": adjustment * 100,
        }

    def adaptive_risk_metrics(self, bonds: List[Bond], weights: Optional[List[float]] = None) -> Dict:
        """
        Calculate risk metrics that adapt to current regime

        More sophisticated than static risk models

        Args:
            bonds: List of bonds
            weights: Portfolio weights

        Returns:
            Adaptive risk metrics by regime
        """
        if self.regimes is None:
            self.detect_regimes(bonds)

        if weights is None:
            weights = [1.0 / len(bonds)] * len(bonds)

        regime_risks = {}

        for regime_id in np.unique(self.regimes):
            regime_bonds = [b for b, label in zip(bonds, self.regimes) if label == regime_id]
            regime_weights = [w for w, label in zip(weights, self.regimes) if label == regime_id]

            # Normalize weights
            if sum(regime_weights) > 0:
                regime_weights = [w / sum(regime_weights) for w in regime_weights]
            else:
                regime_weights = [1.0 / len(regime_bonds)] * len(regime_bonds)

            # Calculate VaR for this regime
            var_result = self.valuator  # Use risk manager
            from bondtrader.risk.risk_management import RiskManager

            risk_mgr = RiskManager(self.valuator)

            try:
                var_result = risk_mgr.calculate_var(regime_bonds, regime_weights, confidence_level=0.95, method="monte_carlo")

                regime_risks[f"Regime {int(regime_id)}"] = {
                    "var_value": var_result["var_value"],
                    "var_pct": var_result["var_percentage"],
                    "num_bonds": len(regime_bonds),
                }
            except (KeyError, ValueError, TypeError) as e:
                # Skip this regime if calculation fails
                import logging

                logger = logging.getLogger(__name__)
                logger.debug(f"Skipping regime risk calculation: {e}")
                continue

        return {
            "regime_risks": regime_risks,
            "num_regimes": len(np.unique(self.regimes)) if self.regimes is not None else 0,
        }
