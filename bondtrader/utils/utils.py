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

# Only configure if not already configured
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=getattr(logging, _log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(_log_path), logging.StreamHandler()],
    )

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors"""

    pass


def validate_bond_data(bond_data: Dict[str, Any]) -> bool:
    """Validate bond data before creating Bond object"""
    required_fields = ["bond_id", "bond_type", "face_value", "coupon_rate", "maturity_date", "issue_date", "current_price"]

    for field in required_fields:
        if field not in bond_data:
            raise ValidationError(f"Missing required field: {field}")

    if bond_data["current_price"] <= 0:
        raise ValidationError("Current price must be positive")

    if bond_data["face_value"] <= 0:
        raise ValidationError("Face value must be positive")

    if bond_data["maturity_date"] <= bond_data["issue_date"]:
        raise ValidationError("Maturity date must be after issue date")

    return True


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
    """Decorator to handle exceptions gracefully"""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
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
