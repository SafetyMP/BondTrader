"""
Factory Patterns for Common Object Creation
Provides consistent, centralized object creation with proper dependency injection
"""

from typing import List, Optional

from bondtrader.core.bond_models import Bond, BondType
from bondtrader.core.container import get_container
from bondtrader.core.exceptions import InvalidBondError


class BondFactory:
    """
    Factory for creating Bond objects
    Ensures consistent creation with validation
    """

    @staticmethod
    def create(
        bond_id: str,
        bond_type: BondType,
        face_value: float,
        coupon_rate: float,
        maturity_date,
        issue_date,
        current_price: float,
        credit_rating: str = "BBB",
        issuer: str = "Unknown",
        frequency: int = 2,
        callable: bool = False,
        convertible: bool = False,
    ) -> Bond:
        """
        Create a Bond object with validation

        Args:
            bond_id: Unique bond identifier
            bond_type: Type of bond
            face_value: Face value of bond
            coupon_rate: Annual coupon rate (as decimal)
            maturity_date: Maturity date
            issue_date: Issue date
            current_price: Current market price
            credit_rating: Credit rating
            issuer: Issuer name
            frequency: Coupon payment frequency per year
            callable: Whether bond is callable
            convertible: Whether bond is convertible

        Returns:
            Bond object

        Raises:
            InvalidBondError: If validation fails
        """
        # Validate inputs
        if face_value <= 0:
            raise InvalidBondError("Face value must be positive")
        if current_price <= 0:
            raise InvalidBondError("Current price must be positive")
        if issue_date >= maturity_date:
            raise InvalidBondError("Issue date must be before maturity date")
        if not 0 <= coupon_rate <= 1:
            raise InvalidBondError("Coupon rate must be between 0 and 1")
        if frequency < 1:
            raise InvalidBondError("Frequency must be at least 1")

        return Bond(
            bond_id=bond_id,
            bond_type=bond_type,
            face_value=face_value,
            coupon_rate=coupon_rate,
            maturity_date=maturity_date,
            issue_date=issue_date,
            current_price=current_price,
            credit_rating=credit_rating,
            issuer=issuer,
            frequency=frequency,
            callable=callable,
            convertible=convertible,
        )

    @staticmethod
    def create_from_dict(data: dict) -> Bond:
        """
        Create Bond from dictionary

        Args:
            data: Dictionary with bond data

        Returns:
            Bond object
        """
        # Convert string bond_type to enum if needed
        bond_type = data.get("bond_type")
        if isinstance(bond_type, str):
            bond_type = BondType[bond_type.upper()]

        # Convert string dates to datetime if needed
        from datetime import datetime

        maturity_date = data.get("maturity_date")
        if isinstance(maturity_date, str):
            maturity_date = datetime.fromisoformat(maturity_date.replace("Z", "+00:00"))

        issue_date = data.get("issue_date")
        if isinstance(issue_date, str):
            issue_date = datetime.fromisoformat(issue_date.replace("Z", "+00:00"))

        return BondFactory.create(
            bond_id=data["bond_id"],
            bond_type=bond_type,
            face_value=data["face_value"],
            coupon_rate=data["coupon_rate"],
            maturity_date=maturity_date,
            issue_date=issue_date,
            current_price=data["current_price"],
            credit_rating=data.get("credit_rating", "BBB"),
            issuer=data.get("issuer", "Unknown"),
            frequency=data.get("frequency", 2),
            callable=data.get("callable", False),
            convertible=data.get("convertible", False),
        )


class MLModelFactory:
    """
    Factory for creating ML model instances
    Ensures consistent model creation with proper dependencies
    """

    @staticmethod
    def create_basic(model_type: str = "random_forest"):
        """Create basic ML adjuster"""
        from bondtrader.ml.ml_adjuster import MLBondAdjuster

        container = get_container()
        valuator = container.get_valuator()
        return MLBondAdjuster(model_type=model_type, valuator=valuator)

    @staticmethod
    def create_enhanced(model_type: str = "random_forest"):
        """Create enhanced ML adjuster"""
        from bondtrader.ml.ml_adjuster_enhanced import EnhancedMLBondAdjuster

        container = get_container()
        valuator = container.get_valuator()
        return EnhancedMLBondAdjuster(model_type=model_type, valuator=valuator)

    @staticmethod
    def create_advanced():
        """Create advanced ML adjuster"""
        from bondtrader.ml.ml_advanced import AdvancedMLBondAdjuster

        container = get_container()
        valuator = container.get_valuator()
        return AdvancedMLBondAdjuster(valuator=valuator)

    @staticmethod
    def create_automl():
        """Create AutoML adjuster"""
        from bondtrader.ml.automl import AutoMLBondAdjuster

        container = get_container()
        valuator = container.get_valuator()
        return AutoMLBondAdjuster(valuator=valuator)


class AnalyticsFactory:
    """
    Factory for creating analytics instances
    Ensures consistent analytics object creation
    """

    @staticmethod
    def create_portfolio_optimizer():
        """Create portfolio optimizer"""
        from bondtrader.analytics.portfolio_optimization import PortfolioOptimizer

        container = get_container()
        valuator = container.get_valuator()
        return PortfolioOptimizer(valuator=valuator)

    @staticmethod
    def create_factor_model():
        """Create factor model"""
        from bondtrader.analytics.factor_models import FactorModel

        container = get_container()
        valuator = container.get_valuator()
        return FactorModel(valuator=valuator)

    @staticmethod
    def create_backtest_engine():
        """Create backtest engine"""
        from bondtrader.analytics.backtesting import BacktestEngine

        container = get_container()
        valuator = container.get_valuator()
        return BacktestEngine(valuator=valuator)

    @staticmethod
    def create_correlation_analyzer():
        """Create correlation analyzer"""
        from bondtrader.analytics.correlation_analysis import CorrelationAnalyzer

        container = get_container()
        valuator = container.get_valuator()
        return CorrelationAnalyzer(valuator=valuator)


class RiskFactory:
    """
    Factory for creating risk management instances
    """

    @staticmethod
    def create_risk_manager():
        """Create risk manager (already in container, but provides factory interface)"""
        container = get_container()
        return container.get_risk_manager()

    @staticmethod
    def create_liquidity_risk_analyzer():
        """Create liquidity risk analyzer"""
        from bondtrader.risk.liquidity_risk_enhanced import LiquidityRiskEnhanced

        container = get_container()
        valuator = container.get_valuator()
        return LiquidityRiskEnhanced(valuator=valuator)

    @staticmethod
    def create_tail_risk_analyzer():
        """Create tail risk analyzer"""
        from bondtrader.risk.tail_risk import TailRiskAnalyzer

        container = get_container()
        valuator = container.get_valuator()
        return TailRiskAnalyzer(valuator=valuator)
