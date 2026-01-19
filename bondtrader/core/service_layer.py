"""
Service Layer Pattern
Separates business logic from presentation and data access
Following Domain-Driven Design principles
"""

from typing import Any, Dict, List, Optional

from bondtrader.core.audit import AuditEventType, get_audit_logger
from bondtrader.core.bond_models import Bond
from bondtrader.core.bond_valuation import BondValuator
from bondtrader.core.exceptions import BusinessRuleViolation, DataNotFoundError, InvalidBondError, ValuationError
from bondtrader.core.observability import get_metrics, trace
from bondtrader.core.repository import BondRepository, IBondRepository
from bondtrader.core.result import Result


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
