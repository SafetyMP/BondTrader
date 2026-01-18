"""
Correlation Analysis Module
Covariance matrices, correlation analysis, and portfolio diversification
Industry-standard correlation metrics
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from bondtrader.core.bond_models import Bond
from bondtrader.core.bond_valuation import BondValuator
from bondtrader.utils.utils import logger


class CorrelationAnalyzer:
    """
    Correlation and covariance analysis for bond portfolios
    """

    def __init__(self, valuator: BondValuator = None):
        """Initialize correlation analyzer"""
        self.valuator = valuator if valuator else BondValuator()

    def calculate_correlation_matrix(self, bonds: List[Bond], method: str = "characteristics") -> Dict:
        """
        Calculate correlation matrix for bonds

        Args:
            bonds: List of bonds
            method: 'characteristics' or 'returns' (if historical data available)

        Returns:
            Correlation matrix and analysis
        """
        n = len(bonds)
        correlation_matrix = np.zeros((n, n))

        if method == "characteristics":
            # Correlation based on bond characteristics
            for i in range(n):
                for j in range(n):
                    if i == j:
                        correlation_matrix[i, j] = 1.0
                    else:
                        # Calculate similarity based on characteristics
                        correlation = self._calculate_bond_similarity(bonds[i], bonds[j])
                        correlation_matrix[i, j] = correlation

        else:  # returns method (would use historical data)
            # Simulate returns for demonstration
            returns = np.random.randn(n, 252)  # 252 days
            correlation_matrix = np.corrcoef(returns)

        # Convert to DataFrame for better visualization
        bond_ids = [b.bond_id for b in bonds]
        corr_df = pd.DataFrame(correlation_matrix, index=bond_ids, columns=bond_ids)

        # Calculate average correlation
        # Exclude diagonal (self-correlation = 1)
        mask = ~np.eye(n, dtype=bool)
        avg_correlation = correlation_matrix[mask].mean()

        # Diversification ratio
        portfolio_vol = np.sqrt(np.mean(np.diag(correlation_matrix)))  # Simplified
        avg_corr_vol = np.sqrt(avg_correlation) if avg_correlation > 0 else 0
        diversification_ratio = portfolio_vol / avg_corr_vol if avg_corr_vol > 0 else 1.0

        return {
            "correlation_matrix": correlation_matrix.tolist(),
            "correlation_dataframe": corr_df,
            "average_correlation": avg_correlation,
            "diversification_ratio": diversification_ratio,
            "bond_ids": bond_ids,
            "method": method,
        }

    def _calculate_bond_similarity(self, bond1: Bond, bond2: Bond) -> float:
        """Calculate similarity/correlation between two bonds"""
        # Factors affecting correlation:
        # 1. Credit rating similarity
        rating_sim = 1.0 if bond1.credit_rating == bond2.credit_rating else 0.5

        # 2. Maturity similarity
        ttm1 = bond1.time_to_maturity
        ttm2 = bond2.time_to_maturity
        maturity_diff = abs(ttm1 - ttm2) / max(ttm1, ttm2, 1)
        maturity_sim = 1.0 - min(maturity_diff, 1.0)

        # 3. Bond type similarity
        type_sim = 1.0 if bond1.bond_type == bond2.bond_type else 0.3

        # 4. Issuer similarity
        issuer_sim = 1.0 if bond1.issuer == bond2.issuer else 0.2

        # Weighted average
        correlation = 0.3 * rating_sim + 0.3 * maturity_sim + 0.2 * type_sim + 0.2 * issuer_sim

        return correlation

    def calculate_covariance_matrix(self, bonds: List[Bond], correlation_matrix: Optional[np.ndarray] = None) -> Dict:
        """
        Calculate covariance matrix

        Args:
            bonds: List of bonds
            correlation_matrix: Pre-calculated correlation matrix (if None, calculates)

        Returns:
            Covariance matrix
        """
        n = len(bonds)

        # Calculate individual volatilities
        volatilities = []
        for bond in bonds:
            ytm = self.valuator.calculate_yield_to_maturity(bond)
            duration = self.valuator.calculate_duration(bond, ytm)
            # Volatility ≈ duration * yield_volatility
            volatility = abs(duration * 0.01)  # Simplified
            volatilities.append(volatility)

        volatilities = np.array(volatilities)

        # Get correlation matrix
        if correlation_matrix is None:
            corr_result = self.calculate_correlation_matrix(bonds)
            correlation_matrix = np.array(corr_result["correlation_matrix"])

        # Covariance = correlation * std_i * std_j
        covariance_matrix = np.outer(volatilities, volatilities) * correlation_matrix

        return {
            "covariance_matrix": covariance_matrix.tolist(),
            "volatilities": volatilities.tolist(),
            "correlation_matrix": correlation_matrix.tolist(),
        }

    def portfolio_diversification_metrics(self, bonds: List[Bond], weights: Optional[np.ndarray] = None) -> Dict:
        """
        Calculate portfolio diversification metrics

        Args:
            bonds: List of bonds
            weights: Portfolio weights (if None, equal weights)

        Returns:
            Diversification metrics
        """
        if weights is None:
            weights = np.ones(len(bonds)) / len(bonds)
        else:
            weights = np.array(weights)

        # Get correlation matrix
        corr_result = self.calculate_correlation_matrix(bonds)
        correlation_matrix = np.array(corr_result["correlation_matrix"])

        # Effective number of positions
        # Measures how many uncorrelated positions portfolio has
        portfolio_variance = np.dot(weights, np.dot(correlation_matrix, weights))
        avg_weight = np.mean(weights)
        effective_positions = 1 / np.sum(weights**2) if np.sum(weights**2) > 0 else len(bonds)

        # Concentration metrics
        herfindahl_index = np.sum(weights**2)  # Concentration measure
        gini_coefficient = self._calculate_gini(weights)

        # Diversification benefit
        # How much risk reduction from diversification
        avg_correlation = corr_result["average_correlation"]
        undiversified_risk = np.sqrt(np.sum(weights**2))
        diversified_risk = np.sqrt(portfolio_variance)
        diversification_benefit = (undiversified_risk - diversified_risk) / undiversified_risk if undiversified_risk > 0 else 0

        return {
            "effective_positions": effective_positions,
            "herfindahl_index": herfindahl_index,
            "gini_coefficient": gini_coefficient,
            "average_correlation": avg_correlation,
            "portfolio_variance": portfolio_variance,
            "diversification_benefit": diversification_benefit,
            "diversification_benefit_pct": diversification_benefit * 100,
        }

    def _calculate_gini(self, weights: np.ndarray) -> float:
        """Calculate Gini coefficient for concentration"""
        sorted_weights = np.sort(weights)
        n = len(weights)
        cumsum = np.cumsum(sorted_weights)

        # Gini = 2 * sum(i * w_i) / (n * sum(w_i)) - (n+1)/n
        numerator = 2 * np.sum((np.arange(1, n + 1) * sorted_weights))
        denominator = n * np.sum(sorted_weights)

        if denominator == 0:
            return 0

        gini = (numerator / denominator) - (n + 1) / n
        return gini

    def sector_correlation_analysis(self, bonds: List[Bond]) -> Dict:
        """
        Analyze correlations within and across sectors

        Args:
            bonds: List of bonds

        Returns:
            Sector correlation analysis
        """
        # Group by issuer/sector (simplified: use issuer as sector)
        sectors = {}
        for bond in bonds:
            sector = bond.issuer if bond.issuer else "Unknown"
            if sector not in sectors:
                sectors[sector] = []
            sectors[sector].append(bond)

        # Within-sector correlations
        within_sector_corrs = {}
        for sector, sector_bonds in sectors.items():
            if len(sector_bonds) > 1:
                corr_result = self.calculate_correlation_matrix(sector_bonds)
                within_sector_corrs[sector] = corr_result["average_correlation"]

        # Cross-sector correlations
        sector_list = list(sectors.keys())
        cross_sector_corr = np.zeros((len(sector_list), len(sector_list)))

        for i, sector1 in enumerate(sector_list):
            for j, sector2 in enumerate(sector_list):
                if i == j:
                    cross_sector_corr[i, j] = 1.0
                else:
                    # Average correlation between sectors
                    bonds1 = sectors[sector1]
                    bonds2 = sectors[sector2]
                    cross_corrs = []
                    for b1 in bonds1[:3]:  # Sample
                        for b2 in bonds2[:3]:
                            cross_corrs.append(self._calculate_bond_similarity(b1, b2))
                    cross_sector_corr[i, j] = np.mean(cross_corrs) if cross_corrs else 0.5

        return {
            "sectors": sector_list,
            "within_sector_correlations": within_sector_corrs,
            "cross_sector_correlation_matrix": cross_sector_corr.tolist(),
            "num_bonds_per_sector": {s: len(bonds) for s, bonds in sectors.items()},
        }
