"""
Dependency Injection Container
Manages service instances and ensures consistent architecture
"""

import os
from typing import Optional

from bondtrader.config import Config, get_config
from bondtrader.core.arbitrage_detector import ArbitrageDetector
from bondtrader.core.bond_valuation import BondValuator
from bondtrader.core.repository import BondRepository, IBondRepository
from bondtrader.core.service_layer import BondService
from bondtrader.data.data_persistence import EnhancedBondDatabase
from bondtrader.risk.risk_management import RiskManager


class ServiceContainer:
    """
    Dependency Injection Container
    Manages singleton instances of core services
    Ensures consistent configuration and shared state
    """

    _instance: Optional["ServiceContainer"] = None
    _initialized: bool = False

    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize container (only once)"""
        if self._initialized:
            return

        self._config: Optional[Config] = None
        self._valuator: Optional[BondValuator] = None
        self._repository: Optional[IBondRepository] = None
        self._bond_service: Optional[BondService] = None
        self._arbitrage_detector: Optional[ArbitrageDetector] = None
        self._risk_manager: Optional[RiskManager] = None
        self._database: Optional[EnhancedBondDatabase] = None

        ServiceContainer._initialized = True

    @property
    def config(self) -> Config:
        """Get configuration instance"""
        if self._config is None:
            self._config = get_config()
        return self._config

    def get_valuator(self, risk_free_rate: Optional[float] = None) -> BondValuator:
        """
        Get BondValuator instance (singleton)

        Args:
            risk_free_rate: Optional override for risk-free rate

        Returns:
            Shared BondValuator instance
        """
        if self._valuator is None:
            rfr = (
                risk_free_rate if risk_free_rate is not None else self.config.default_risk_free_rate
            )
            self._valuator = BondValuator(risk_free_rate=rfr)
        elif risk_free_rate is not None and self._valuator.risk_free_rate != risk_free_rate:
            # Update risk-free rate if different
            self._valuator.risk_free_rate = risk_free_rate
        return self._valuator

    def get_database(self, db_path: Optional[str] = None) -> EnhancedBondDatabase:
        """
        Get database instance (singleton)

        Args:
            db_path: Optional database path override

        Returns:
            Shared EnhancedBondDatabase instance
        """
        if self._database is None:
            path = db_path or os.path.join(self.config.data_dir, "bonds.db")
            self._database = EnhancedBondDatabase(db_path=path)
        return self._database

    def get_repository(self, database: Optional[EnhancedBondDatabase] = None) -> IBondRepository:
        """
        Get repository instance (singleton)

        Args:
            database: Optional database instance override

        Returns:
            Shared BondRepository instance
        """
        if self._repository is None:
            db = database or self.get_database()
            self._repository = BondRepository(database=db)
        return self._repository

    def get_bond_service(
        self,
        repository: Optional[IBondRepository] = None,
        valuator: Optional[BondValuator] = None,
    ) -> BondService:
        """
        Get BondService instance (singleton)

        Args:
            repository: Optional repository override
            valuator: Optional valuator override

        Returns:
            Shared BondService instance
        """
        if self._bond_service is None:
            repo = repository or self.get_repository()
            val = valuator or self.get_valuator()
            self._bond_service = BondService(repository=repo, valuator=val)
        return self._bond_service

    def get_arbitrage_detector(self, valuator: Optional[BondValuator] = None) -> ArbitrageDetector:
        """
        Get ArbitrageDetector instance (singleton)

        Args:
            valuator: Optional valuator override

        Returns:
            Shared ArbitrageDetector instance
        """
        if self._arbitrage_detector is None:
            val = valuator or self.get_valuator()
            self._arbitrage_detector = ArbitrageDetector(
                valuator=val,
                min_profit_threshold=self.config.min_profit_threshold,
                include_transaction_costs=self.config.include_transaction_costs,
            )
        return self._arbitrage_detector

    def get_risk_manager(self, valuator: Optional[BondValuator] = None) -> RiskManager:
        """
        Get RiskManager instance (singleton)

        Args:
            valuator: Optional valuator override

        Returns:
            Shared RiskManager instance
        """
        if self._risk_manager is None:
            val = valuator or self.get_valuator()
            self._risk_manager = RiskManager(valuator=val)
        return self._risk_manager

    def reset(self):
        """Reset all instances (useful for testing)"""
        self._valuator = None
        self._repository = None
        self._bond_service = None
        self._arbitrage_detector = None
        self._risk_manager = None
        self._database = None
        self._config = None


# Global container instance
_container: Optional[ServiceContainer] = None


def get_container() -> ServiceContainer:
    """Get global service container instance"""
    global _container
    if _container is None:
        _container = ServiceContainer()
    return _container


def reset_container():
    """Reset global container (useful for testing)"""
    global _container
    if _container is not None:
        _container.reset()
    _container = None
