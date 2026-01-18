"""
Enhanced Logging Utilities
Optional structured logging with structlog and loguru support
"""

import logging
import sys
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
                structlog.processors.JSONRenderer() if log_level == "DEBUG" else structlog.dev.ConsoleRenderer(),
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


def get_logger(name: str = __name__, use_enhanced: bool = True) -> Any:
    """
    Get logger instance with optional enhanced logging

    Args:
        name: Logger name
        use_enhanced: Use enhanced logging (structlog/loguru) if available

    Returns:
        Logger instance
    """
    if use_enhanced and HAS_LOGURU:
        return loguru_logger.bind(name=name)
    elif use_enhanced and HAS_STRUCTLOG:
        return structlog.get_logger(name)
    else:
        return logging.getLogger(name)


class LoggerMixin:
    """Mixin class to add logging capabilities to any class"""

    @property
    def logger(self):
        """Get logger instance for this class"""
        if not hasattr(self, "_logger"):
            class_name = self.__class__.__name__
            self._logger = get_logger(f"bondtrader.{class_name}")
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
        from functools import wraps
        from time import time

        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            name = func_name or func.__name__
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
        from functools import wraps

        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            logger = logger.bind(**context)
            return func(*args, **kwargs)

        return wrapper

    return decorator
