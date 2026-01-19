"""
Repository Pattern Implementation
Abstracts data access layer following industry best practices
Provides clean separation between business logic and data persistence

CRITICAL: Supports transaction management through optional session parameter
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from bondtrader.core.bond_models import Bond

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


class IBondRepository(ABC):
    """
    Repository interface for bond data access
    Abstracts persistence implementation details
    """

    @abstractmethod
    def save(self, bond: Bond, session: Optional["Session"] = None) -> None:
        """
        Save a bond

        Args:
            bond: Bond to save
            session: Optional SQLAlchemy session for transaction support
        """
        pass

    @abstractmethod
    def find_by_id(self, bond_id: str) -> Optional[Bond]:
        """Find bond by ID"""
        pass

    @abstractmethod
    def find_all(self, filters: Optional[Dict[str, Any]] = None) -> List[Bond]:
        """Find all bonds with optional filters"""
        pass

    @abstractmethod
    def delete(self, bond_id: str) -> bool:
        """Delete bond by ID"""
        pass

    @abstractmethod
    def exists(self, bond_id: str) -> bool:
        """Check if bond exists"""
        pass

    @abstractmethod
    def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count bonds matching filters"""
        pass


class BondRepository(IBondRepository):
    """
    Concrete repository implementation using EnhancedBondDatabase
    Adapter pattern to convert between repository interface and database implementation
    """

    def __init__(self, database=None):
        """Initialize with database instance"""
        if database is None:
            from bondtrader.data.data_persistence import EnhancedBondDatabase

            database = EnhancedBondDatabase()
        self.db = database

    def save(self, bond: Bond, session: Optional["Session"] = None) -> None:
        """
        Save a bond with optional transaction support.

        CRITICAL: If session is provided, uses that session for transaction atomicity.
        If not provided, creates its own session (standalone operation).

        Args:
            bond: Bond to save
            session: Optional SQLAlchemy session (for transaction support)
        """
        self.db.save_bond(bond, session=session)

    def find_by_id(self, bond_id: str) -> Optional[Bond]:
        """Find bond by ID"""
        try:
            return self.db.load_bond(bond_id)
        except (FileNotFoundError, KeyError, ValueError) as e:
            # Bond not found or invalid ID - return None
            return None
        except Exception as e:
            # Unexpected error - log and return None
            from bondtrader.utils.utils import logger

            logger.warning(f"Unexpected error loading bond {bond_id}: {e}")
            return None

    def find_all(self, filters: Optional[Dict[str, Any]] = None) -> List[Bond]:
        """Find all bonds with optional filters"""
        bonds = self.db.load_all_bonds()

        if not filters:
            return bonds

        # Apply filters
        filtered = bonds
        if "bond_type" in filters:
            filtered = [b for b in filtered if b.bond_type == filters["bond_type"]]
        if "issuer" in filters:
            filtered = [b for b in filtered if b.issuer and filters["issuer"].lower() in b.issuer.lower()]
        if "credit_rating" in filters:
            filtered = [b for b in filtered if b.credit_rating == filters["credit_rating"]]

        return filtered

    def delete(self, bond_id: str) -> bool:
        """Delete bond by ID"""
        # EnhancedBondDatabase doesn't have delete method, so we'll skip for now
        # In production, implement proper delete
        return False

    def exists(self, bond_id: str) -> bool:
        """Check if bond exists"""
        return self.find_by_id(bond_id) is not None

    def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count bonds matching filters"""
        return len(self.find_all(filters))


class InMemoryBondRepository(IBondRepository):
    """
    In-memory repository for testing
    """

    def __init__(self):
        self._bonds: Dict[str, Bond] = {}

    def save(self, bond: Bond, session: Optional["Session"] = None) -> None:
        """
        Save a bond (in-memory, session parameter ignored for compatibility)

        Args:
            bond: Bond to save
            session: Ignored (in-memory repository doesn't use sessions)
        """
        self._bonds[bond.bond_id] = bond

    def find_by_id(self, bond_id: str) -> Optional[Bond]:
        return self._bonds.get(bond_id)

    def find_all(self, filters: Optional[Dict[str, Any]] = None) -> List[Bond]:
        bonds = list(self._bonds.values())

        if not filters:
            return bonds

        # Apply filters (same as BondRepository)
        filtered = bonds
        if "bond_type" in filters:
            filtered = [b for b in filtered if b.bond_type == filters["bond_type"]]
        if "issuer" in filters:
            filtered = [b for b in filtered if b.issuer and filters["issuer"].lower() in b.issuer.lower()]
        if "credit_rating" in filters:
            filtered = [b for b in filtered if b.credit_rating == filters["credit_rating"]]

        return filtered

    def delete(self, bond_id: str) -> bool:
        if bond_id in self._bonds:
            del self._bonds[bond_id]
            return True
        return False

    def exists(self, bond_id: str) -> bool:
        return bond_id in self._bonds

    def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        return len(self.find_all(filters))
