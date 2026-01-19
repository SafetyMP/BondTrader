"""
Tests for structured logging utilities
"""

import pytest

from bondtrader.utils.structured_logging import StructuredLogger, correlation_id


@pytest.mark.unit
class TestStructuredLogger:
    """Test StructuredLogger functionality"""

    def test_structured_logger_init(self):
        """Test structured logger initialization"""
        logger = StructuredLogger("test")
        assert logger.logger.name == "test"

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
        # Just verify it doesn't raise
        logger.info("Test message")

    def test_log_levels(self):
        """Test different log levels"""
        logger = StructuredLogger("test")
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        # Just verify they don't raise
