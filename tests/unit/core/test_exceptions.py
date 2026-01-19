"""
Tests for exception hierarchy
"""

import pytest

from bondtrader.core.exceptions import (
    ArbitrageDetectionError,
    BondTraderException,
    BusinessRuleViolation,
    CalculationError,
    ConfigurationError,
    CreditRiskError,
    DataError,
    DataIntegrityError,
    DataNotFoundError,
    DataProviderError,
    ExecutionError,
    ExternalServiceError,
    FINRAAPIError,
    FREDAPIError,
    InsufficientDataError,
    InsufficientFundsError,
    InvalidBondError,
    LiquidityRiskError,
    MLError,
    ModelNotFoundError,
    ModelPredictionError,
    ModelTrainingError,
    PortfolioConstraintError,
    RiskCalculationError,
    TradingError,
    ValidationError,
    ValuationError,
    VaRCalculationError,
)


@pytest.mark.unit
class TestExceptionHierarchy:
    """Test exception hierarchy"""

    def test_bond_trader_exception_base(self):
        """Test base exception"""
        exc = BondTraderException("Test message")
        assert str(exc) == "Test message"
        assert exc.message == "Test message"
        assert exc.error_code == "BondTraderException"
        assert exc.details == {}

    def test_bond_trader_exception_with_code(self):
        """Test exception with error code"""
        exc = BondTraderException("Test", error_code="CUSTOM_CODE")
        assert exc.error_code == "CUSTOM_CODE"
        assert "[CUSTOM_CODE]" in str(exc)

    def test_bond_trader_exception_with_details(self):
        """Test exception with details"""
        exc = BondTraderException("Test", details={"key": "value"})
        assert exc.details == {"key": "value"}
        assert exc.to_dict()["details"] == {"key": "value"}

    def test_valuation_errors(self):
        """Test valuation error hierarchy"""
        assert issubclass(ValuationError, BondTraderException)
        assert issubclass(InvalidBondError, ValuationError)
        assert issubclass(CalculationError, ValuationError)
        assert issubclass(InsufficientDataError, ValuationError)

    def test_risk_errors(self):
        """Test risk error hierarchy"""
        assert issubclass(RiskCalculationError, BondTraderException)
        assert issubclass(VaRCalculationError, RiskCalculationError)
        assert issubclass(CreditRiskError, RiskCalculationError)
        assert issubclass(LiquidityRiskError, RiskCalculationError)

    def test_data_errors(self):
        """Test data error hierarchy"""
        assert issubclass(DataError, BondTraderException)
        assert issubclass(DataNotFoundError, DataError)
        assert issubclass(DataIntegrityError, DataError)
        assert issubclass(DataProviderError, DataError)

    def test_ml_errors(self):
        """Test ML error hierarchy"""
        assert issubclass(MLError, BondTraderException)
        assert issubclass(ModelTrainingError, MLError)
        assert issubclass(ModelPredictionError, MLError)
        assert issubclass(ModelNotFoundError, MLError)

    def test_trading_errors(self):
        """Test trading error hierarchy"""
        assert issubclass(TradingError, BondTraderException)
        assert issubclass(ArbitrageDetectionError, TradingError)
        assert issubclass(ExecutionError, TradingError)

    def test_external_service_errors(self):
        """Test external service error hierarchy"""
        assert issubclass(ExternalServiceError, BondTraderException)
        assert issubclass(FREDAPIError, ExternalServiceError)
        assert issubclass(FINRAAPIError, ExternalServiceError)

    def test_business_rule_errors(self):
        """Test business rule error hierarchy"""
        assert issubclass(BusinessRuleViolation, BondTraderException)
        assert issubclass(InsufficientFundsError, BusinessRuleViolation)
        assert issubclass(PortfolioConstraintError, BusinessRuleViolation)

    def test_exception_to_dict(self):
        """Test exception serialization"""
        exc = ValuationError("Test message", error_code="VAL001", details={"key": "value"})
        exc_dict = exc.to_dict()
        assert exc_dict["error_type"] == "ValuationError"
        assert exc_dict["error_code"] == "VAL001"
        assert exc_dict["message"] == "Test message"
        assert exc_dict["details"] == {"key": "value"}
