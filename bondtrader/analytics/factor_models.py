"""
Factor Models Module
PCA-based factors, statistical factors, and risk attribution
Industry-standard factor analysis for bond portfolios
"""

from typing import Dict, List, Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from bondtrader.core.bond_models import Bond
from bondtrader.core.bond_valuation import BondValuator


class FactorModel:
    """
    Factor model for bond returns
    Decomposes returns into systematic factors and idiosyncratic risk
    """

    def __init__(self, valuator: Optional[BondValuator] = None) -> None:
        """
        Initialize factor model

        Args:
            valuator: Bond valuator instance (optional, creates default if None)
        """
        if valuator is None:
            from bondtrader.core.container import get_container

            self.valuator = get_container().get_valuator()
        else:
            self.valuator = valuator
        self.factors = None
        self.factor_loadings = None
        self.scaler = StandardScaler()

    def extract_bond_factors(self, bonds: List[Bond], num_factors: Optional[int] = None) -> Dict:
        """
        Extract factors from bond characteristics using PCA

        Common factors:
        - Level (parallel shift)
        - Slope (steepening/flattening)
        - Curvature (hump)

        Args:
            bonds: List of bonds
            num_factors: Number of factors to extract (if None, auto-select)

        Returns:
            Factor analysis results
        """
        n = len(bonds)

        # OPTIMIZED: Batch calculate YTM, duration, and convexity to leverage caching
        # Build feature matrix with batched calculations
        ytms = [self.valuator.calculate_yield_to_maturity(bond) for bond in bonds]
        durations = [self.valuator.calculate_duration(bond, ytm) for bond, ytm in zip(bonds, ytms)]
        convexities = [
            self.valuator.calculate_convexity(bond, ytm) for bond, ytm in zip(bonds, ytms)
        ]

        # Build feature matrix using pre-calculated values
        features = []
        for bond, ytm, duration, convexity in zip(bonds, ytms, durations, convexities):
            feature_vector = [
                bond.coupon_rate,
                bond.time_to_maturity,
                ytm * 100,
                duration,
                convexity,
                self.valuator._get_credit_spread(bond.credit_rating) * 10000,  # In bps
                bond.current_price / bond.face_value,
            ]
            features.append(feature_vector)

        X = np.array(features)

        # Standardize features
        X_scaled = self.scaler.fit_transform(X)

        # Determine number of factors
        if num_factors is None:
            # Use elbow method or keep factors explaining 80% variance
            pca_temp = PCA()
            pca_temp.fit(X_scaled)
            cumsum_variance = np.cumsum(pca_temp.explained_variance_ratio_)
            num_factors = np.argmax(cumsum_variance >= 0.80) + 1
            num_factors = min(num_factors, n - 1, 5)  # Cap at 5 factors

        # Perform PCA
        pca = PCA(n_components=num_factors)
        factors = pca.fit_transform(X_scaled)
        factor_loadings = pca.components_.T

        self.factors = factors
        self.factor_loadings = factor_loadings

        # Interpret factors
        factor_names = self._interpret_factors(factor_loadings, pca.explained_variance_ratio_)

        return {
            "factors": factors.tolist(),
            "factor_loadings": factor_loadings.tolist(),
            "explained_variance": pca.explained_variance_ratio_.tolist(),
            "cumulative_variance": np.cumsum(pca.explained_variance_ratio_).tolist(),
            "factor_names": factor_names,
            "num_factors": num_factors,
            "feature_names": [
                "coupon_rate",
                "maturity",
                "ytm",
                "duration",
                "convexity",
                "credit_spread",
                "price_to_par",
            ],
        }

    def _interpret_factors(self, loadings: np.ndarray, variance: np.ndarray) -> List[str]:
        """Interpret factors based on loadings"""
        factor_names = []

        for i in range(loadings.shape[1]):
            load = loadings[:, i]

            # Check which features have highest loadings
            max_idx = np.argmax(np.abs(load))

            if max_idx == 2:  # YTM
                factor_names.append("Level Factor")
            elif max_idx == 3:  # Duration
                factor_names.append("Slope Factor")
            elif max_idx == 4:  # Convexity
                factor_names.append("Curvature Factor")
            elif max_idx == 1:  # Maturity
                factor_names.append("Maturity Factor")
            elif max_idx == 5:  # Credit spread
                factor_names.append("Credit Factor")
            else:
                factor_names.append(f"Factor {i+1}")

        return factor_names

    def calculate_factor_exposures(
        self, bonds: List[Bond], portfolio_weights: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Calculate portfolio factor exposures

        Args:
            bonds: List of bonds
            portfolio_weights: Portfolio weights (if None, equal weights)

        Returns:
            Factor exposure analysis
        """
        # Always re-extract factors if they don't exist or if the number of bonds has changed
        if (
            self.factors is None
            or self.factor_loadings is None
            or (self.factors is not None and self.factors.shape[0] != len(bonds))
        ):
            # Extract factors first
            self.extract_bond_factors(bonds)

        if portfolio_weights is None:
            portfolio_weights = np.ones(len(bonds)) / len(bonds)
        else:
            portfolio_weights = np.array(portfolio_weights)

        # Portfolio factor exposure = weighted sum of individual exposures
        # self.factors has shape (n_bonds, n_factors), representing each bond's factor exposure
        portfolio_exposure = np.dot(portfolio_weights, self.factors)

        # Factor contribution to portfolio risk
        factor_contributions = portfolio_exposure**2

        return {
            "portfolio_exposures": portfolio_exposure.tolist(),
            "factor_contributions": factor_contributions.tolist(),
            "total_exposure": np.sum(np.abs(portfolio_exposure)),
            "dominant_factor": np.argmax(np.abs(portfolio_exposure)),
        }

    def risk_attribution(
        self, bonds: List[Bond], portfolio_weights: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Risk attribution by factor

        Decomposes portfolio risk into factor contributions

        Args:
            bonds: List of bonds
            portfolio_weights: Portfolio weights

        Returns:
            Risk attribution analysis
        """
        if portfolio_weights is None:
            portfolio_weights = np.ones(len(bonds)) / len(bonds)
        else:
            portfolio_weights = np.array(portfolio_weights)

        # Get factor exposures
        exposure_result = self.calculate_factor_exposures(bonds, portfolio_weights)

        # Calculate covariance matrix
        from bondtrader.analytics.portfolio_optimization import PortfolioOptimizer

        optimizer = PortfolioOptimizer(self.valuator)
        _, covariance = optimizer.calculate_returns_and_covariance(bonds)

        # Portfolio variance
        portfolio_variance = np.dot(portfolio_weights, np.dot(covariance, portfolio_weights))

        # Factor risk contributions - optimized with list comprehension
        portfolio_exposures = exposure_result["portfolio_exposures"]
        if self.factors is not None:
            factor_variances = [self.factors[:, i].var() for i in range(len(portfolio_exposures))]
        else:
            factor_variances = [1.0] * len(portfolio_exposures)

        # Vectorized calculation: factor risk = exposure^2 * factor_variance
        factor_risks = [
            (exposure**2) * var for exposure, var in zip(portfolio_exposures, factor_variances)
        ]

        total_factor_risk = sum(factor_risks)
        factor_risk_pct = [
            r / portfolio_variance * 100 if portfolio_variance > 0 else 0 for r in factor_risks
        ]

        return {
            "portfolio_variance": portfolio_variance,
            "portfolio_volatility": np.sqrt(portfolio_variance),
            "factor_risks": factor_risks,
            "factor_risk_percentages": factor_risk_pct,
            "total_factor_risk": total_factor_risk,
            "idiosyncratic_risk": portfolio_variance - total_factor_risk,
            "idiosyncratic_risk_pct": (
                ((portfolio_variance - total_factor_risk) / portfolio_variance * 100)
                if portfolio_variance > 0
                else 0
            ),
        }

    def statistical_factors(
        self, bonds: List[Bond], return_data: Optional[np.ndarray] = None, num_factors: int = 3
    ) -> Dict:
        """
        Extract statistical factors from return data

        Args:
            bonds: List of bonds
            return_data: Historical return matrix (if None, simulated)
            num_factors: Number of factors

        Returns:
            Statistical factor analysis
        """
        n = len(bonds)

        if return_data is None:
            # Simulate returns
            periods = 252
            return_data = np.random.randn(n, periods) * 0.01

        # Standardize returns
        return_data_scaled = StandardScaler().fit_transform(return_data.T).T

        # PCA on returns
        pca = PCA(n_components=num_factors)
        factors = pca.fit_transform(return_data_scaled.T).T
        loadings = pca.components_

        return {
            "factors": factors.tolist(),
            "loadings": loadings.tolist(),
            "explained_variance": pca.explained_variance_ratio_.tolist(),
            "num_factors": num_factors,
        }
