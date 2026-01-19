"""
Unified Logging Utilities
Provides structured logging with correlation IDs, context, and optional external library support
"""

import logging
import sys
import uuid
from contextvars import ContextVar
from functools import wraps
from typing import Any, Dict, Optional

# Optional structured logging libraries
try:
    import structlog

    HAS_STRUCTLOG = True
except ImportError:
    HAS_STRUCTLOG = False

try:
    from loguru import logger as loguru_logger

    HAS_LOGURU = True
except ImportError:
    HAS_LOGURU = False

# Context variable for correlation IDs (thread-safe)
correlation_id: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)


class StructuredLogger:
    """
    Structured logger with context support and correlation IDs
    Supports both standard logging and optional external libraries (structlog/loguru)
    """

    def __init__(self, name: str, use_external: bool = True):
        """
        Initialize structured logger

        Args:
            name: Logger name
            use_external: Use external logging libraries (structlog/loguru) if available
        """
        self.name = name
        self._context: Dict[str, Any] = {}
        self.use_external = use_external

        # Choose logger implementation
        if use_external and HAS_LOGURU:
            self._logger = loguru_logger.bind(name=name)
            self._logger_type = "loguru"
        elif use_external and HAS_STRUCTLOG:
            self._logger = structlog.get_logger(name)
            self._logger_type = "structlog"
        else:
            self._logger = logging.getLogger(name)
            self._logger_type = "standard"

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

    def _format_message(self, message: str, **kwargs) -> tuple:
        """Format message based on logger type"""
        if self._logger_type in ("loguru", "structlog"):
            # External loggers handle structured data natively
            return message, kwargs
        else:
            # Standard logger needs extra dict
            return message, self._get_extra(**kwargs)

    def info(self, message: str, **kwargs):
        """Log info message with structure"""
        msg, extra = self._format_message(message, **kwargs)
        if self._logger_type == "standard":
            self._logger.info(msg, extra=extra)
        elif self._logger_type == "loguru":
            self._logger.info(msg, **extra)
        else:  # structlog
            self._logger.info(msg, **extra)

    def warning(self, message: str, **kwargs):
        """Log warning message with structure"""
        msg, extra = self._format_message(message, **kwargs)
        if self._logger_type == "standard":
            self._logger.warning(msg, extra=extra)
        elif self._logger_type == "loguru":
            self._logger.warning(msg, **extra)
        else:  # structlog
            self._logger.warning(msg, **extra)

    def error(self, message: str, **kwargs):
        """Log error message with structure"""
        msg, extra = self._format_message(message, **kwargs)
        if self._logger_type == "standard":
            self._logger.error(msg, extra=extra)
        elif self._logger_type == "loguru":
            self._logger.error(msg, **extra)
        else:  # structlog
            self._logger.error(msg, **extra)

    def debug(self, message: str, **kwargs):
        """Log debug message with structure"""
        msg, extra = self._format_message(message, **kwargs)
        if self._logger_type == "standard":
            self._logger.debug(msg, extra=extra)
        elif self._logger_type == "loguru":
            self._logger.debug(msg, **extra)
        else:  # structlog
            self._logger.debug(msg, **extra)

    def exception(self, message: str, **kwargs):
        """Log exception with structure"""
        msg, extra = self._format_message(message, **kwargs)
        if self._logger_type == "standard":
            self._logger.exception(msg, extra=extra)
        elif self._logger_type == "loguru":
            self._logger.exception(msg, **extra)
        else:  # structlog
            self._logger.exception(msg, **extra)


def setup_structured_logging(use_structlog: bool = True, use_loguru: bool = False, log_level: str = "INFO") -> Any:
    """
    Setup structured logging with optional structlog or loguru

    Args:
        use_structlog: Use structlog for structured logging
        use_loguru: Use loguru for enhanced logging
        log_level: Logging level

    Returns:
        Logger instance
    """
    if use_loguru and HAS_LOGURU:
        # Configure loguru
        loguru_logger.remove()  # Remove default handler
        loguru_logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=log_level,
            colorize=True,
        )
        loguru_logger.add(
            "logs/bond_trading.log",
            rotation="10 MB",
            retention="30 days",
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        )
        return loguru_logger

    elif use_structlog and HAS_STRUCTLOG:
        # Configure structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                (structlog.processors.JSONRenderer() if log_level == "DEBUG" else structlog.dev.ConsoleRenderer()),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        return structlog.get_logger()

    else:
        # Fallback to standard logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper(), logging.INFO),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()],
        )
        return logging.getLogger(__name__)


def get_logger(name: str = None, use_enhanced: bool = True) -> Any:
    """
    Get logger instance with optional enhanced logging

    Args:
        name: Logger name
        use_enhanced: Use enhanced logging (structlog/loguru) if available

    Returns:
        Logger instance (StructuredLogger or external logger)
    """
    if name is None:
        import inspect

        frame = inspect.currentframe().f_back
        name = frame.f_globals.get("__name__", "bondtrader")

    if use_enhanced and (HAS_LOGURU or HAS_STRUCTLOG):
        # Return StructuredLogger for consistency
        return StructuredLogger(name, use_external=use_enhanced)
    else:
        return StructuredLogger(name, use_external=False)


def get_structured_logger(name: str = None) -> StructuredLogger:
    """Get or create structured logger with correlation ID support"""
    if name is None:
        import inspect

        frame = inspect.currentframe().f_back
        name = frame.f_globals.get("__name__", "bondtrader")
    return StructuredLogger(name)


class LoggerMixin:
    """Mixin class to add logging capabilities to any class"""

    @property
    def logger(self) -> StructuredLogger:
        """Get logger instance for this class"""
        if not hasattr(self, "_logger"):
            class_name = self.__class__.__name__
            self._logger = StructuredLogger(f"bondtrader.{class_name}")
        return self._logger


def log_performance(func_name: str = None):
    """
    Decorator to log function performance

    Args:
        func_name: Optional function name (defaults to function.__name__)

    Usage:
        @log_performance()
        def my_function():
            ...
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_structured_logger(func.__module__)
            name = func_name or func.__name__
            from time import time

            start_time = time()

            try:
                result = func(*args, **kwargs)
                duration = time() - start_time
                logger.info(f"{name} completed", duration_seconds=duration, status="success")
                return result
            except Exception as e:
                duration = time() - start_time
                logger.error(f"{name} failed", duration_seconds=duration, error=str(e), status="error")
                raise

        return wrapper

    return decorator


def log_with_context(**context):
    """
    Decorator to add context to logging

    Usage:
        @log_with_context(bond_id="BOND-001", model="xgboost")
        def my_function():
            ...
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_structured_logger(func.__module__)
            logger.add_context(**context)
            try:
                return func(*args, **kwargs)
            finally:
                logger.clear_context()

        return wrapper

    return decorator


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
        logger = get_structured_logger(func.__module__)
        cid = logger.set_correlation_id()

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # Clear correlation ID after execution
            correlation_id.set(None)

    return wrapper


# Export correlation_id for backward compatibility
__all__ = [
    "StructuredLogger",
    "LoggerMixin",
    "setup_structured_logging",
    "get_logger",
    "get_structured_logger",
    "log_performance",
    "log_with_context",
    "with_correlation_id",
    "correlation_id",
]
