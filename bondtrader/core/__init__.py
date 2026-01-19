"""Core bond trading modules with industry best practices"""

from bondtrader.core.arbitrage_detector import ArbitrageDetector
from bondtrader.core.bond_models import Bond, BondType
from bondtrader.core.bond_valuation import BondValuator

# Import new architectural components
try:
    from bondtrader.core.audit import AuditEventType, AuditLogger, audit_log, get_audit_logger
    from bondtrader.core.circuit_breaker import (
        CircuitBreaker,
        CircuitBreakerConfig,
        circuit_breaker,
        get_circuit_breaker,
    )
    from bondtrader.core.container import ServiceContainer, get_container, reset_container
    from bondtrader.core.exceptions import (
        BondTraderException,
        DataError,
        MLError,
        RiskCalculationError,
        TradingError,
        ValidationError,
        ValuationError,
    )
    from bondtrader.core.factories import (
        AnalyticsFactory,
        BondFactory,
        MLModelFactory,
        RiskFactory,
    )
    from bondtrader.core.helpers import (
        calculate_portfolio_value,
        format_valuation_result,
        get_bond_or_error,
        get_bonds_or_error,
        validate_bond_data,
    )
    from bondtrader.core.observability import Metrics, get_metrics, trace, trace_context
    from bondtrader.core.repository import (
        BondRepository,
        IBondRepository,
        InMemoryBondRepository,
    )
    from bondtrader.core.result import Result, safe
    from bondtrader.core.service_layer import BondService
    from bondtrader.utils.utils import safe_divide
except ImportError:
    # Graceful degradation if new modules not available
    pass

__all__ = [
    "Bond",
    "BondType",
    "BondValuator",
    "ArbitrageDetector",
    "BondService",
    "ServiceContainer",
    "get_container",
    "reset_container",
    "BondFactory",
    "MLModelFactory",
    "AnalyticsFactory",
    "RiskFactory",
    "get_bond_or_error",
    "get_bonds_or_error",
    "validate_bond_data",
    "calculate_portfolio_value",
    "format_valuation_result",
    "safe_divide",
]
