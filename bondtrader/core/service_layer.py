"""
Service Layer Pattern
Separates business logic from presentation and data access
Following Domain-Driven Design principles
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

from bondtrader.core.audit import AuditEventType, get_audit_logger
from bondtrader.core.bond_models import Bond
from bondtrader.core.bond_valuation import BondValuator
from bondtrader.core.exceptions import (
    BusinessRuleViolation,
    DataNotFoundError,
    InvalidBondError,
    MLError,
    RiskCalculationError,
    ValuationError,
)
from bondtrader.core.observability import get_metrics, trace
from bondtrader.core.repository import BondRepository, IBondRepository
from bondtrader.core.result import Result
from bondtrader.utils.utils import logger

if TYPE_CHECKING:
    from bondtrader.analytics.portfolio_optimization import PortfolioOptimizer
    from bondtrader.ml.ml_adjuster import MLBondAdjuster
    from bondtrader.ml.ml_adjuster_enhanced import EnhancedMLBondAdjuster
    from bondtrader.risk.risk_management import RiskManager


class BondService:
    """
    Service layer for bond operations
    Encapsulates business logic and orchestrates domain operations
    """

    def __init__(self, repository: Optional[IBondRepository] = None, valuator: Optional[BondValuator] = None):
        """Initialize service with dependencies"""
        self.repository = repository or BondRepository()
        self.valuator = valuator or BondValuator()
        self.audit_logger = get_audit_logger()

    @trace
    def create_bond(self, bond: Bond) -> Result[Bond, Exception]:
        """
        Create a new bond

        Returns Result type for explicit error handling
        """
        try:
            # Validate bond
            if bond.current_price <= 0:
                return Result.err(InvalidBondError("Current price must be positive"))

            if bond.face_value <= 0:
                return Result.err(InvalidBondError("Face value must be positive"))

            # Business rule: Check if bond already exists
            if self.repository.exists(bond.bond_id):
                return Result.err(BusinessRuleViolation(f"Bond {bond.bond_id} already exists"))

            # Save bond
            self.repository.save(bond)

            # Audit log
            self.audit_logger.log(
                AuditEventType.BOND_CREATED,
                bond.bond_id,
                "bond_created",
                details={"bond_type": bond.bond_type.name, "face_value": bond.face_value},
            )

            # Metrics
            get_metrics().increment("bond.created", tags={"bond_type": bond.bond_type.name})

            return Result.ok(bond)

        except Exception as e:
            get_metrics().increment("bond.create_error")
            return Result.err(e)

    @trace
    def get_bond(self, bond_id: str) -> Result[Bond, Exception]:
        """Get bond by ID"""
        try:
            bond = self.repository.find_by_id(bond_id)
            if not bond:
                return Result.err(DataNotFoundError(f"Bond {bond_id} not found"))

            # Audit log
            self.audit_logger.log(AuditEventType.DATA_ACCESSED, bond_id, "bond_accessed")

            return Result.ok(bond)

        except Exception as e:
            get_metrics().increment("bond.get_error")
            return Result.err(e)

    @trace
    def calculate_valuation(self, bond_id: str) -> Result[Dict[str, Any], Exception]:
        """Calculate valuation for a bond"""
        try:
            # Get bond
            bond_result = self.get_bond(bond_id)
            if bond_result.is_err():
                return bond_result.map_err(lambda e: ValuationError(f"Failed to get bond: {e}"))

            bond = bond_result.value

            # Calculate valuation
            fair_value = self.valuator.calculate_fair_value(bond)
            ytm = self.valuator.calculate_yield_to_maturity(bond)
            duration = self.valuator.calculate_duration(bond, ytm)
            convexity = self.valuator.calculate_convexity(bond, ytm)

            valuation = {
                "bond_id": bond_id,
                "fair_value": fair_value,
                "ytm": ytm,
                "duration": duration,
                "convexity": convexity,
                "market_price": bond.current_price,
                "mismatch_percentage": ((bond.current_price - fair_value) / fair_value) * 100,
            }

            # Audit log
            self.audit_logger.log_valuation(bond_id, fair_value, ytm, duration=duration, convexity=convexity)

            # Metrics
            get_metrics().histogram("valuation.fair_value", fair_value)
            get_metrics().histogram("valuation.mismatch_percentage", abs(valuation["mismatch_percentage"]))

            return Result.ok(valuation)

        except Exception as e:
            get_metrics().increment("valuation.error")
            return Result.err(ValuationError(f"Valuation calculation failed: {e}"))

    @trace
    def find_bonds(self, filters: Optional[Dict[str, Any]] = None) -> Result[List[Bond], Exception]:
        """Find bonds with optional filters"""
        try:
            bonds = self.repository.find_all(filters)

            # Audit log
            filter_str = str(filters) if filters else "none"
            get_metrics().increment("bond.search", tags={"has_filters": str(bool(filters))})

            return Result.ok(bonds)

        except Exception as e:
            get_metrics().increment("bond.search_error")
            return Result.err(e)

    @trace
    def get_bond_count(self, filters: Optional[Dict[str, Any]] = None) -> Result[int, Exception]:
        """Get count of bonds"""
        try:
            count = self.repository.count(filters)
            return Result.ok(count)
        except Exception as e:
            return Result.err(e)

    @trace
    def calculate_valuation_for_bond(self, bond: Bond) -> Result[Dict[str, Any], Exception]:
        """
        Calculate valuation for a Bond object (without requiring repository lookup)

        Args:
            bond: Bond object to value

        Returns:
            Result containing valuation dictionary
        """
        try:
            # Calculate valuation
            fair_value = self.valuator.calculate_fair_value(bond)
            ytm = self.valuator.calculate_yield_to_maturity(bond)
            duration = self.valuator.calculate_duration(bond, ytm)
            convexity = self.valuator.calculate_convexity(bond, ytm)

            valuation = {
                "bond_id": bond.bond_id,
                "fair_value": fair_value,
                "ytm": ytm,
                "duration": duration,
                "convexity": convexity,
                "market_price": bond.current_price,
                "mismatch_percentage": ((bond.current_price - fair_value) / fair_value) * 100,
            }

            # Audit log
            self.audit_logger.log_valuation(bond.bond_id, fair_value, ytm, duration=duration, convexity=convexity)

            # Metrics
            get_metrics().histogram("valuation.fair_value", fair_value)
            get_metrics().histogram("valuation.mismatch_percentage", abs(valuation["mismatch_percentage"]))

            return Result.ok(valuation)

        except Exception as e:
            get_metrics().increment("valuation.error")
            return Result.err(ValuationError(f"Valuation calculation failed: {e}"))

    @trace
    def calculate_valuations_batch(self, bonds: List[Bond]) -> Result[List[Dict[str, Any]], Exception]:
        """
        Calculate valuations for multiple bonds

        Args:
            bonds: List of Bond objects

        Returns:
            Result containing list of valuation dictionaries
        """
        try:
            valuations = []
            for bond in bonds:
                result = self.calculate_valuation_for_bond(bond)
                if result.is_err():
                    # Log error but continue with other bonds
                    get_metrics().increment("valuation.batch_error")
                    continue
                valuations.append(result.value)

            get_metrics().increment("valuation.batch", tags={"count": len(valuations)})
            return Result.ok(valuations)

        except Exception as e:
            get_metrics().increment("valuation.batch_error")
            return Result.err(ValuationError(f"Batch valuation calculation failed: {e}"))

    @trace
    def predict_with_ml(
        self,
        bond_id: str,
        ml_model: Optional["MLBondAdjuster"] = None,
        model_type: str = "enhanced",
    ) -> Result[Dict[str, Any], Exception]:
        """
        Predict ML-adjusted fair value for a bond with graceful degradation.

        CRITICAL: Falls back to simple DCF if ML model fails.
        """
        """
        Predict ML-adjusted fair value for a bond

        Args:
            bond_id: Bond identifier
            ml_model: Optional pre-instantiated ML model. If None, creates one.
            model_type: Type of ML model to use if ml_model not provided ('basic', 'enhanced', 'advanced')

        Returns:
            Result containing ML prediction with adjusted fair value
        """
        try:
            # Get bond
            bond_result = self.get_bond(bond_id)
            if bond_result.is_err():
                return bond_result.map_err(lambda e: MLError(f"Failed to get bond: {e}"))

            bond = bond_result.value

            # Get or create ML model
            if ml_model is None:
                from bondtrader.core.container import get_container

                valuator = get_container().get_valuator()

                if model_type == "enhanced":
                    from bondtrader.ml.ml_adjuster_enhanced import EnhancedMLBondAdjuster

                    ml_model = EnhancedMLBondAdjuster(valuator=valuator)
                elif model_type == "advanced":
                    from bondtrader.ml.ml_advanced import AdvancedMLBondAdjuster

                    ml_model = AdvancedMLBondAdjuster(valuator=valuator)
                else:
                    from bondtrader.ml.ml_adjuster import MLBondAdjuster

                    ml_model = MLBondAdjuster(valuator=valuator)

            # Predict
            if not ml_model.is_trained:
                # If model not trained, return theoretical value
                valuation_result = self.calculate_valuation_for_bond(bond)
                if valuation_result.is_err():
                    return valuation_result.map_err(lambda e: MLError(f"Valuation failed: {e}"))

                valuation = valuation_result.value
                prediction = {
                    "bond_id": bond_id,
                    "theoretical_fair_value": valuation["fair_value"],
                    "ml_adjusted_fair_value": valuation["fair_value"],
                    "adjustment_factor": 1.0,
                    "ml_confidence": 0.0,
                    "model_trained": False,
                }
            else:
                try:
                    ml_result = ml_model.predict_adjusted_value(bond)
                    prediction = {
                        "bond_id": bond_id,
                        "theoretical_fair_value": ml_result.get("theoretical_fair_value", 0),
                        "ml_adjusted_fair_value": ml_result.get("ml_adjusted_fair_value", 0),
                        "adjustment_factor": ml_result.get("adjustment_factor", 1.0),
                        "ml_confidence": ml_result.get("ml_confidence", 0.5),
                        "model_trained": True,
                        "fallback_used": False,
                    }
                except Exception as ml_error:
                    # CRITICAL: Graceful degradation - fallback to simple DCF
                    logger.warning(f"ML prediction failed for bond {bond_id}, using DCF fallback: {ml_error}")
                    valuation_result = self.calculate_valuation_for_bond(bond)
                    if valuation_result.is_err():
                        return valuation_result.map_err(lambda e: MLError(f"Both ML and DCF failed: {e}"))

                    valuation = valuation_result.value
                    prediction = {
                        "bond_id": bond_id,
                        "theoretical_fair_value": valuation["fair_value"],
                        "ml_adjusted_fair_value": valuation["fair_value"],
                        "adjustment_factor": 1.0,
                        "ml_confidence": 0.0,
                        "model_trained": False,
                        "fallback_used": True,
                    }

            # Audit log
            self.audit_logger.log(
                AuditEventType.MODEL_PREDICTION,
                bond_id,
                "ml_prediction",
                details={
                    "model_type": model_type,
                    "adjustment_factor": prediction["adjustment_factor"],
                    "model_trained": prediction["model_trained"],
                },
            )

            # Metrics
            get_metrics().increment(
                "ml.prediction",
                tags={"model_type": model_type, "trained": str(prediction["model_trained"])},
            )
            get_metrics().histogram("ml.adjustment_factor", prediction["adjustment_factor"])

            return Result.ok(prediction)

        except Exception as e:
            get_metrics().increment("ml.prediction_error")
            return Result.err(MLError(f"ML prediction failed: {e}"))

    @trace
    def find_arbitrage_opportunities(
        self,
        filters: Optional[Dict[str, Any]] = None,
        min_profit_percentage: float = 0.01,
        use_ml: bool = True,
        limit: Optional[int] = None,
    ) -> Result[List[Dict[str, Any]], Exception]:
        """
        Find arbitrage opportunities in bonds

        Args:
            filters: Optional filters for bond search
            min_profit_percentage: Minimum profit percentage threshold
            use_ml: Whether to use ML-adjusted valuations
            limit: Maximum number of opportunities to return

        Returns:
            Result containing list of arbitrage opportunities
        """
        try:
            # Get bonds
            bonds_result = self.find_bonds(filters=filters)
            if bonds_result.is_err():
                return bonds_result.map_err(lambda e: ValuationError(f"Failed to get bonds: {e}"))

            bonds = bonds_result.value

            # Get arbitrage detector from container
            from bondtrader.core.container import get_container

            arbitrage_detector = get_container().get_arbitrage_detector()

            # Set min_profit_threshold on detector before finding opportunities
            arbitrage_detector.min_profit_threshold = min_profit_percentage
            # Find opportunities
            opportunities = arbitrage_detector.find_arbitrage_opportunities(bonds, use_ml=use_ml)

            # Sort by profit and limit
            opportunities.sort(key=lambda x: x.get("profit_percentage", 0), reverse=True)
            if limit:
                opportunities = opportunities[:limit]

            # Audit log
            self.audit_logger.log(
                AuditEventType.ARBITRAGE_DETECTED,
                "system",
                "arbitrage_search",
                details={
                    "num_opportunities": len(opportunities),
                    "min_profit": min_profit_percentage,
                    "used_ml": use_ml,
                },
            )

            # Metrics
            get_metrics().increment("arbitrage.search", tags={"found": str(len(opportunities)), "used_ml": str(use_ml)})

            return Result.ok(opportunities)

        except Exception as e:
            get_metrics().increment("arbitrage.search_error")
            return Result.err(ValuationError(f"Arbitrage detection failed: {e}"))

    @trace
    def calculate_portfolio_risk(
        self,
        bond_ids: List[str],
        weights: Optional[List[float]] = None,
        confidence_level: float = 0.95,
    ) -> Result[Dict[str, Any], Exception]:
        """
        Calculate portfolio risk metrics

        Args:
            bond_ids: List of bond identifiers
            weights: Optional portfolio weights (defaults to equal weights)
            confidence_level: Confidence level for VaR calculations

        Returns:
            Result containing risk metrics
        """
        try:
            # Get bonds
            bonds = []
            for bond_id in bond_ids:
                bond_result = self.get_bond(bond_id)
                if bond_result.is_err():
                    return bond_result.map_err(lambda e: RiskCalculationError(f"Failed to get bond {bond_id}: {e}"))
                bonds.append(bond_result.value)

            # Default to equal weights
            if weights is None:
                weights = [1.0 / len(bonds)] * len(bonds)

            if len(weights) != len(bonds):
                return Result.err(RiskCalculationError("Weights length must match bonds length"))

            # Get risk manager from container
            from bondtrader.core.container import get_container

            risk_manager = get_container().get_risk_manager()

            # Calculate VaR using different methods
            var_historical = risk_manager.calculate_var(
                bonds,
                weights,
                confidence_level=confidence_level,
                time_horizon=1,
                method="historical",
            )
            var_parametric = risk_manager.calculate_var(
                bonds,
                weights,
                confidence_level=confidence_level,
                time_horizon=1,
                method="parametric",
            )
            var_monte_carlo = risk_manager.calculate_var(
                bonds,
                weights,
                confidence_level=confidence_level,
                time_horizon=1,
                method="monte_carlo",
            )

            # Calculate portfolio credit risk
            portfolio_credit_risk = risk_manager.calculate_portfolio_credit_risk(bonds, weights)

            risk_metrics = {
                "var_historical": var_historical.get("var_value"),
                "var_parametric": var_parametric.get("var_value"),
                "var_monte_carlo": var_monte_carlo.get("var_value"),
                "credit_risk": portfolio_credit_risk,
                "num_bonds": len(bonds),
                "confidence_level": confidence_level,
            }

            # Audit log
            self.audit_logger.log(
                AuditEventType.RISK_CALCULATED,
                "portfolio",
                "portfolio_risk",
                details={"num_bonds": len(bonds), "confidence_level": confidence_level},
            )

            # Metrics
            get_metrics().histogram("risk.var_historical", risk_metrics["var_historical"] or 0)
            get_metrics().histogram("risk.var_parametric", risk_metrics["var_parametric"] or 0)

            # Business metrics: Track risk metrics
            if risk_metrics["var_historical"]:
                get_metrics().track_risk_metric("var_95", risk_metrics["var_historical"])
            if portfolio_credit_risk:
                get_metrics().track_risk_metric("credit_risk", portfolio_credit_risk.get("expected_loss", 0))

            return Result.ok(risk_metrics)

        except Exception as e:
            get_metrics().increment("risk.calculation_error")
            return Result.err(RiskCalculationError(f"Risk calculation failed: {e}"))

    @trace
    def calculate_portfolio_metrics(
        self, bond_ids: List[str], weights: Optional[List[float]] = None
    ) -> Result[Dict[str, Any], Exception]:
        """
        Calculate portfolio analytics metrics

        Args:
            bond_ids: List of bond identifiers
            weights: Optional portfolio weights (defaults to equal weights)

        Returns:
            Result containing portfolio metrics
        """
        try:
            # Get bonds
            bonds = []
            for bond_id in bond_ids:
                bond_result = self.get_bond(bond_id)
                if bond_result.is_err():
                    return bond_result.map_err(lambda e: ValuationError(f"Failed to get bond {bond_id}: {e}"))
                bonds.append(bond_result.value)

            # Default to equal weights
            if weights is None:
                weights = [1.0 / len(bonds)] * len(bonds)

            if len(weights) != len(bonds):
                return Result.err(ValuationError("Weights length must match bonds length"))

            # Get portfolio optimizer from container
            from bondtrader.core.container import get_container

            container = get_container()
            valuator = container.get_valuator()

            from bondtrader.analytics.portfolio_optimization import PortfolioOptimizer

            optimizer = PortfolioOptimizer(valuator=valuator)

            # Calculate returns and covariance
            returns, covariance = optimizer.calculate_returns_and_covariance(bonds)

            # Calculate portfolio metrics
            portfolio_return = np.dot(returns, weights)
            portfolio_variance = np.dot(weights, np.dot(covariance, weights))
            portfolio_std = np.sqrt(portfolio_variance)

            # Calculate individual bond valuations
            valuations_result = self.calculate_valuations_batch(bonds)
            if valuations_result.is_err():
                return valuations_result

            valuations = valuations_result.value
            total_fair_value = sum(v["fair_value"] for v in valuations)
            total_market_value = sum(b.current_price for b in bonds)

            metrics = {
                "portfolio_return": portfolio_return,
                "portfolio_variance": portfolio_variance,
                "portfolio_std": portfolio_std,
                "total_fair_value": total_fair_value,
                "total_market_value": total_market_value,
                "mismatch_percentage": (
                    ((total_market_value - total_fair_value) / total_fair_value * 100) if total_fair_value > 0 else 0
                ),
                "num_bonds": len(bonds),
                "weights": weights,
            }

            # Audit log
            self.audit_logger.log(
                AuditEventType.PORTFOLIO_UPDATED,
                "portfolio",
                "portfolio_metrics",
                details={"num_bonds": len(bonds)},
            )

            # Metrics
            get_metrics().histogram("portfolio.return", portfolio_return)
            get_metrics().histogram("portfolio.std", portfolio_std)

            return Result.ok(metrics)

        except Exception as e:
            get_metrics().increment("portfolio.analysis_error")
            return Result.err(ValuationError(f"Portfolio analysis failed: {e}"))

    @trace
    def create_bonds_batch(self, bonds: List[Bond]) -> Result[List[Bond], Exception]:
        """
        Create multiple bonds in a batch operation with transaction support.

        CRITICAL: Uses database transaction for atomicity - all bonds created or none.

        Args:
            bonds: List of Bond objects to create

        Returns:
            Result containing list of created bonds
        """
        try:
            # CRITICAL: Use transaction for atomicity
            # All bonds must be created successfully, or entire operation rolls back
            created_bonds = []
            errors = []

            if hasattr(self.repository, "db") and hasattr(self.repository.db, "transaction"):
                with self.repository.db.transaction() as session:
                    # Validate all bonds first
                    for bond in bonds:
                        # Validate bond
                        if bond.current_price <= 0:
                            errors.append(f"Bond {bond.bond_id}: Current price must be positive")
                            continue
                        if bond.face_value <= 0:
                            errors.append(f"Bond {bond.bond_id}: Face value must be positive")
                            continue

                        # Check if bond already exists (within transaction)
                        if self.repository.exists(bond.bond_id):
                            errors.append(f"Bond {bond.bond_id}: Already exists")
                            continue

                        created_bonds.append(bond)

                    if errors and not created_bonds:
                        # All failed validation - transaction will rollback on exception
                        return Result.err(BusinessRuleViolation(f"All bonds failed validation: {'; '.join(errors)}"))

                    # Save all bonds within transaction (pass session for atomicity)
                    for bond in created_bonds:
                        try:
                            self.repository.save(bond, session=session)
                        except Exception as e:
                            # Error will cause transaction rollback
                            logger.error(f"Error saving bond {bond.bond_id} in transaction: {e}")
                            raise

                    # Transaction commits automatically on context exit (success)
                    # If any error occurs, transaction rolls back automatically
            else:
                # Fallback: Process without transaction (not recommended for production)
                for bond in bonds:
                    result = self.create_bond(bond)
                    if result.is_err():
                        errors.append(f"Bond {bond.bond_id}: {result.error}")
                    else:
                        created_bonds.append(result.value)

                if errors and not created_bonds:
                    return Result.err(BusinessRuleViolation(f"All bonds failed to create: {'; '.join(errors)}"))

            # Audit log
            self.audit_logger.log(
                AuditEventType.BOND_CREATED,
                "batch",
                "bonds_created_batch",
                details={
                    "total": len(bonds),
                    "created": len(created_bonds),
                    "failed": len(errors) if "errors" in locals() else 0,
                },
            )

            # Metrics
            get_metrics().increment(
                "bond.batch_created",
                tags={"total": str(len(bonds)), "success": str(len(created_bonds))},
            )

            return Result.ok(created_bonds)

        except Exception as e:
            get_metrics().increment("bond.batch_create_error")
            return Result.err(BusinessRuleViolation(f"Batch bond creation failed: {e}"))

    @trace
    def calculate_valuations_with_ml_batch(
        self,
        bond_ids: List[str],
        ml_model: Optional["MLBondAdjuster"] = None,
        model_type: str = "enhanced",
    ) -> Result[List[Dict[str, Any]], Exception]:
        """
        Calculate ML-adjusted valuations for multiple bonds

        Args:
            bond_ids: List of bond identifiers
            ml_model: Optional pre-instantiated ML model
            model_type: Type of ML model to use if ml_model not provided

        Returns:
            Result containing list of ML predictions
        """
        try:
            predictions = []
            errors = []

            for bond_id in bond_ids:
                result = self.predict_with_ml(bond_id, ml_model=ml_model, model_type=model_type)
                if result.is_err():
                    errors.append(f"Bond {bond_id}: {result.error}")
                else:
                    predictions.append(result.value)

            if errors and not predictions:
                return Result.err(MLError(f"All predictions failed: {'; '.join(errors)}"))

            # Audit log
            self.audit_logger.log(
                AuditEventType.MODEL_PREDICTION,
                "batch",
                "ml_predictions_batch",
                details={
                    "total": len(bond_ids),
                    "success": len(predictions),
                    "failed": len(errors),
                },
            )

            # Metrics
            get_metrics().increment(
                "ml.batch_prediction",
                tags={"total": str(len(bond_ids)), "success": str(len(predictions))},
            )

            return Result.ok(predictions)

        except Exception as e:
            get_metrics().increment("ml.batch_prediction_error")
            return Result.err(MLError(f"Batch ML prediction failed: {e}"))
