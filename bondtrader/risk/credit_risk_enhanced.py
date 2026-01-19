"""
Enhanced Credit Risk Module (DEPRECATED)
Implements Merton structural model, credit migration matrices, and CVaR

NOTE: This module is deprecated. All enhanced credit risk methods have been
merged into RiskManager. Use RiskManager methods directly:
- RiskManager.merton_structural_model()
- RiskManager.credit_migration_analysis()
- RiskManager.calculate_credit_var()
- RiskManager.calculate_expected_credit_loss()

This class is kept for backward compatibility and simply delegates to RiskManager.
"""

from typing import Dict, List, Optional

from bondtrader.core.bond_models import Bond
from bondtrader.core.bond_valuation import BondValuator
from bondtrader.risk.risk_management import RiskManager


class CreditRiskEnhanced:
    """
    Enhanced credit risk analysis (DEPRECATED - use RiskManager instead)

    This class is a thin wrapper around RiskManager for backward compatibility.
    All methods delegate to RiskManager methods.
    """

    def __init__(self, valuator: BondValuator = None):
        """
        Initialize enhanced credit risk analyzer

        Args:
            valuator: Bond valuator instance (passed to RiskManager)
        """
        self.risk_manager = RiskManager(valuator=valuator)
        self.valuator = self.risk_manager.valuator
        self.migration_matrix = self.risk_manager.migration_matrix

    def merton_structural_model(
        self,
        bond: Bond,
        asset_value: Optional[float] = None,
        asset_volatility: float = 0.25,
        debt_value: Optional[float] = None,
    ) -> Dict:
        """Calculate default probability using Merton structural model (delegates to RiskManager)"""
        return self.risk_manager.merton_structural_model(bond, asset_value, asset_volatility, debt_value)

    def credit_migration_analysis(self, bond: Bond, time_horizon: float = 1.0, num_scenarios: int = 10000) -> Dict:
        """Analyze credit migration risk using migration matrix (delegates to RiskManager)"""
        return self.risk_manager.credit_migration_analysis(bond, time_horizon, num_scenarios)

    def calculate_credit_var(
        self,
        bonds: List[Bond],
        weights: Optional[List[float]] = None,
        confidence_level: float = 0.95,
        time_horizon: float = 1.0,
    ) -> Dict:
        """Calculate Credit Value at Risk (CVaR) (delegates to RiskManager)"""
        return self.risk_manager.calculate_credit_var(bonds, weights, confidence_level, time_horizon)

    def calculate_expected_credit_loss(self, bonds: List[Bond], weights: Optional[List[float]] = None) -> Dict:
        """Calculate expected credit loss across portfolio (delegates to RiskManager)"""
        return self.risk_manager.calculate_expected_credit_loss(bonds, weights)
