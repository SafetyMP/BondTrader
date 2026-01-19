"""
Tests for logging utilities
"""

import pytest

from bondtrader.utils.logging import (
    StructuredLogger,
    correlation_id,
    get_logger,
    log_performance,
    setup_structured_logging,
)


@pytest.mark.unit
class TestStructuredLogger:
    """Test StructuredLogger functionality"""

    def test_structured_logger_init(self):
        """Test structured logger initialization"""
        logger = StructuredLogger("test")
        assert logger.name == "test"

    def test_set_correlation_id(self):
        """Test setting correlation ID"""
        logger = StructuredLogger("test")
        cid = logger.set_correlation_id()
        assert cid is not None
        assert isinstance(cid, str)

    def test_set_custom_correlation_id(self):
        """Test setting custom correlation ID"""
        logger = StructuredLogger("test")
        custom_id = "custom-123"
        cid = logger.set_correlation_id(custom_id)
        assert cid == custom_id
        assert logger.get_correlation_id() == custom_id

    def test_get_correlation_id(self):
        """Test getting correlation ID"""
        logger = StructuredLogger("test")
        cid = logger.set_correlation_id()
        assert logger.get_correlation_id() == cid

    def test_add_context(self):
        """Test adding context"""
        logger = StructuredLogger("test")
        logger.add_context(user_id="123", request_id="456")
        assert logger._context["user_id"] == "123"
        assert logger._context["request_id"] == "456"

    def test_clear_context(self):
        """Test clearing context"""
        logger = StructuredLogger("test")
        logger.add_context(user_id="123")
        logger.clear_context()
        assert len(logger._context) == 0

    def test_log_with_context(self):
        """Test logging with context"""
        logger = StructuredLogger("test")
        logger.add_context(user_id="123")
        logger.set_correlation_id("test-cid")
        logger.info("Test message")

    def test_log_levels(self):
        """Test different log levels"""
        logger = StructuredLogger("test")
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")


@pytest.mark.unit
class TestEnhancedLogging:
    """Test enhanced logging functionality"""

    def test_get_logger(self):
        """Test getting logger"""
        logger = get_logger("test_module")
        assert logger is not None

    def test_get_logger_consistency(self):
        """Test getting logger with same name"""
        logger1 = get_logger("test_module")
        logger2 = get_logger("test_module")
        assert logger1 is not None
        assert logger2 is not None

    def test_setup_structured_logging(self):
        """Test setting up structured logging"""
        logger = setup_structured_logging(use_structlog=False, use_loguru=False, log_level="INFO")
        assert logger is not None

    def test_log_performance_decorator(self):
        """Test log_performance decorator"""

        @log_performance()
        def test_func():
            return 42

        result = test_func()
        assert result == 42

    def test_log_performance_with_name(self):
        """Test log_performance with custom name"""

        @log_performance(func_name="custom_name")
        def test_func():
            return 42

        result = test_func()
        assert result == 42

    def test_log_performance_with_error(self):
        """Test log_performance with error"""

        @log_performance()
        def failing_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            failing_func()
