"""
Tests for enhanced logging utilities
"""

import pytest

from bondtrader.utils.enhanced_logging import get_logger, log_performance, setup_structured_logging


@pytest.mark.unit
class TestEnhancedLogging:
    """Test enhanced logging functionality"""

    def test_get_logger(self):
        """Test getting logger"""
        logger = get_logger("test_module")
        assert logger is not None

    def test_get_logger(self):
        """Test getting logger with same name returns logger"""
        logger1 = get_logger("test_module")
        logger2 = get_logger("test_module")
        # Loggers may not be same object but should have same name
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
