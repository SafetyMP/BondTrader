"""
Domain-Specific Exception Hierarchy for BondTrader
Following industry best practices for financial systems with proper error categorization
"""


class BondTraderException(Exception):
    """Base exception for all BondTrader errors"""

    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}

    def __str__(self):
        if self.error_code and self.error_code != self.__class__.__name__:
            return f"[{self.error_code}] {self.message}"
        return self.message

    def to_dict(self):
        """Convert exception to dictionary for serialization"""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
        }


# Valuation Exceptions
class ValuationError(BondTraderException):
    """Base exception for valuation-related errors"""

    pass


class InvalidBondError(ValuationError):
    """Raised when bond data is invalid"""

    pass


class CalculationError(ValuationError):
    """Raised when calculation fails (e.g., YTM convergence)"""

    pass


class InsufficientDataError(ValuationError):
    """Raised when insufficient data for calculation"""

    pass


# Risk Management Exceptions
class RiskCalculationError(BondTraderException):
    """Base exception for risk calculation errors"""

    pass


class VaRCalculationError(RiskCalculationError):
    """Raised when VaR calculation fails"""

    pass


class CreditRiskError(RiskCalculationError):
    """Raised when credit risk calculation fails"""

    pass


class LiquidityRiskError(RiskCalculationError):
    """Raised when liquidity risk calculation fails"""

    pass


# Data Exceptions
class DataError(BondTraderException):
    """Base exception for data-related errors"""

    pass


class DataNotFoundError(DataError):
    """Raised when requested data is not found"""

    pass


class DataIntegrityError(DataError):
    """Raised when data integrity check fails"""

    pass


class DataProviderError(DataError):
    """Raised when external data provider fails"""

    pass


# ML Exceptions
class MLError(BondTraderException):
    """Base exception for ML-related errors"""

    pass


class ModelTrainingError(MLError):
    """Raised when model training fails"""

    pass


class ModelPredictionError(MLError):
    """Raised when model prediction fails"""

    pass


class ModelNotFoundError(MLError):
    """Raised when requested model is not found"""

    pass


# Trading Exceptions
class TradingError(BondTraderException):
    """Base exception for trading-related errors"""

    pass


class ArbitrageDetectionError(TradingError):
    """Raised when arbitrage detection fails"""

    pass


class ExecutionError(TradingError):
    """Raised when trade execution fails"""

    pass


# Configuration Exceptions
class ConfigurationError(BondTraderException):
    """Raised when configuration is invalid"""

    pass


# Validation Exceptions (keep existing for backward compatibility)
class ValidationError(BondTraderException):
    """Raised when validation fails"""

    pass


# External Service Exceptions
class ExternalServiceError(BondTraderException):
    """Base exception for external service errors"""

    pass


class FREDAPIError(ExternalServiceError):
    """Raised when FRED API call fails"""

    pass


class FINRAAPIError(ExternalServiceError):
    """Raised when FINRA API call fails"""

    pass


# Business Logic Exceptions
class BusinessRuleViolation(BondTraderException):
    """Raised when a business rule is violated"""

    pass


class InsufficientFundsError(BusinessRuleViolation):
    """Raised when insufficient funds for operation"""

    pass


class PortfolioConstraintError(BusinessRuleViolation):
    """Raised when portfolio constraint is violated"""

    pass
