"""
Structured Logging Utilities
Industry-standard structured logging with context and correlation IDs
"""

import logging
import uuid
from contextvars import ContextVar
from functools import wraps
from typing import Any, Dict, Optional

from bondtrader.utils.utils import logger as base_logger

# Context variable for correlation IDs (thread-safe)
correlation_id: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)


class StructuredLogger:
    """
    Structured logger with context support
    Adds correlation IDs and structured fields to logs
    """

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self._context: Dict[str, Any] = {}

    def set_correlation_id(self, cid: Optional[str] = None) -> str:
        """Set correlation ID for request tracing"""
        if cid is None:
            cid = str(uuid.uuid4())
        correlation_id.set(cid)
        return cid

    def get_correlation_id(self) -> Optional[str]:
        """Get current correlation ID"""
        return correlation_id.get()

    def add_context(self, **kwargs):
        """Add context fields to all subsequent logs"""
        self._context.update(kwargs)

    def clear_context(self):
        """Clear context fields"""
        self._context.clear()

    def _get_extra(self, **kwargs) -> Dict[str, Any]:
        """Get extra fields for structured logging"""
        extra = {"correlation_id": self.get_correlation_id(), **self._context, **kwargs}
        # Remove None values
        return {k: v for k, v in extra.items() if v is not None}

    def info(self, message: str, **kwargs):
        """Log info message with structure"""
        self.logger.info(message, extra=self._get_extra(**kwargs))

    def warning(self, message: str, **kwargs):
        """Log warning message with structure"""
        self.logger.warning(message, extra=self._get_extra(**kwargs))

    def error(self, message: str, **kwargs):
        """Log error message with structure"""
        self.logger.error(message, extra=self._get_extra(**kwargs))

    def debug(self, message: str, **kwargs):
        """Log debug message with structure"""
        self.logger.debug(message, extra=self._get_extra(**kwargs))

    def exception(self, message: str, **kwargs):
        """Log exception with structure"""
        self.logger.exception(message, extra=self._get_extra(**kwargs))


def with_correlation_id(func):
    """
    Decorator to add correlation ID to function execution

    Usage:
        @with_correlation_id
        def my_function():
            ...
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Try to get logger from module or create new one
        module_logger = StructuredLogger(func.__module__)
        cid = module_logger.set_correlation_id()

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # Clear correlation ID after execution
            correlation_id.set(None)

    return wrapper


def get_structured_logger(name: str = None) -> StructuredLogger:
    """Get or create structured logger"""
    if name is None:
        import inspect

        frame = inspect.currentframe().f_back
        name = frame.f_globals.get("__name__", "bondtrader")
    return StructuredLogger(name)
