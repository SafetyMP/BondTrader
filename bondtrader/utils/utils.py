"""
Utility functions for bond trading system
Includes caching, logging, validation, and parallel processing
"""

import logging
from functools import lru_cache
from typing import Callable, Any, List, Dict
from datetime import datetime
import hashlib
import json


# Configure logging
import os
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'bond_trading.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass


def validate_bond_data(bond_data: Dict[str, Any]) -> bool:
    """Validate bond data before creating Bond object"""
    required_fields = ['bond_id', 'bond_type', 'face_value', 'coupon_rate', 
                      'maturity_date', 'issue_date', 'current_price']
    
    for field in required_fields:
        if field not in bond_data:
            raise ValidationError(f"Missing required field: {field}")
    
    if bond_data['current_price'] <= 0:
        raise ValidationError("Current price must be positive")
    
    if bond_data['face_value'] <= 0:
        raise ValidationError("Face value must be positive")
    
    if bond_data['maturity_date'] <= bond_data['issue_date']:
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
        key_data = json.dumps({'args': str(args), 'kwargs': kwargs}, sort_keys=True)
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
    return date.strftime('%Y-%m-%d')
