"""
Utility functions for bond trading system
Includes caching, logging, validation, and parallel processing
"""

import hashlib
import json
import logging
import os
from datetime import datetime
from typing import Any, Callable, Dict


# Import config lazily to avoid circular imports
def _get_log_config():
    """Get log configuration from config module"""
    try:
        from bondtrader.config import get_config

        config = get_config()
        return config.logs_dir, config.log_file, config.log_level
    except Exception:
        # Fallback if config not available
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
        return log_dir, "bond_trading.log", "INFO"


# Configure logging once
_log_dir, _log_file, _log_level = _get_log_config()
os.makedirs(_log_dir, exist_ok=True)
_log_path = os.path.join(_log_dir, _log_file)

# Enhanced structured logging for production
# Try to use structlog if available, fall back to standard logging
try:
    import structlog
    from structlog.stdlib import LoggerFactory

    # Configure structlog for structured JSON logging
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
            structlog.processors.JSONRenderer(),  # JSON format for production
        ],
        context_class=dict,
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    logger = structlog.get_logger(__name__)
    HAS_STRUCTLOG = True
except ImportError:
    # Fallback to standard logging with enhanced format
    if not logging.getLogger().handlers:
        # Enhanced format with more context
        log_format = "%(asctime)s.%(msecs)03d UTC - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
        date_format = "%Y-%m-%d %H:%M:%S"
        logging.basicConfig(
            level=getattr(logging, _log_level.upper(), logging.INFO),
            format=log_format,
            datefmt=date_format,
            handlers=[logging.FileHandler(_log_path), logging.StreamHandler()],
        )
    logger = logging.getLogger(__name__)
    HAS_STRUCTLOG = False


class ValidationError(Exception):
    """Custom exception for validation errors"""

    pass


def cache_key(*args, **kwargs) -> str:
    """Generate cache key from arguments - optimized for performance"""
    # Optimize: avoid string conversion if args are already hashable
    try:
        # Try to hash directly (faster for hashable types)
        return str(hash((args, tuple(sorted(kwargs.items())))))
    except (TypeError, ValueError):
        # Fallback to JSON serialization for complex types
        key_data = json.dumps({"args": str(args), "kwargs": kwargs}, sort_keys=True)
        return hashlib.md5(key_data.encode()).hexdigest()


def memoize(func: Callable) -> Callable:
    """Decorator to memoize function results"""
    cache = {}

    def wrapper(*args, **kwargs):
        key = cache_key(*args, **kwargs)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    wrapper.cache_clear = lambda: cache.clear()
    return wrapper


def handle_exceptions(func: Callable) -> Callable:
    """
    Decorator to handle exceptions gracefully with specific exception types

    Catches and logs specific exceptions, re-raises critical errors.
    Use for functions that need error logging but should propagate errors.
    """

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (ValueError, TypeError, AttributeError) as e:
            # Input validation errors - log and re-raise
            logger.warning(f"Input error in {func.__name__}: {e}", exc_info=False)
            raise
        except (FileNotFoundError, PermissionError, OSError) as e:
            # File I/O errors - log and re-raise
            logger.error(f"File error in {func.__name__}: {e}", exc_info=True)
            raise
        except Exception as e:
            # Unexpected errors - log with full traceback and re-raise
            logger.error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
            raise

    return wrapper


def format_currency(value: float, decimals: int = 2) -> str:
    """Format number as currency"""
    return f"${value:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format number as percentage"""
    return f"{value:.{decimals}f}%"


def format_date(date: datetime) -> str:
    """Format datetime as string"""
    return date.strftime("%Y-%m-%d")


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safe division with default value on zero denominator

    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if denominator is zero

    Returns:
        Division result or default
    """
    if denominator == 0:
        return default
    return numerator / denominator
