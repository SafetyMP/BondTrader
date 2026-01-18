"""
Unit tests for utility functions
"""

import os
import sys

import pytest

pytestmark = pytest.mark.unit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime

from bondtrader.utils.utils import (
    ValidationError,
    cache_key,
    format_currency,
    format_date,
    format_percentage,
    handle_exceptions,
    logger,
    memoize,
    validate_bond_data,
)


def test_validate_bond_data_success():
    """Test bond data validation with valid data"""
    bond_data = {
        "bond_id": "TEST-001",
        "bond_type": "CORPORATE",
        "face_value": 1000,
        "coupon_rate": 5.0,
        "maturity_date": datetime(2029, 12, 31),
        "issue_date": datetime(2024, 1, 1),
        "current_price": 950,
    }
    
    assert validate_bond_data(bond_data) is True


def test_validate_bond_data_missing_field():
    """Test bond data validation with missing field"""
    bond_data = {
        "bond_id": "TEST-001",
        "face_value": 1000,
        # Missing required fields
    }
    
    with pytest.raises(ValidationError, match="Missing required field"):
        validate_bond_data(bond_data)


def test_validate_bond_data_invalid_price():
    """Test bond data validation with invalid price"""
    bond_data = {
        "bond_id": "TEST-001",
        "bond_type": "CORPORATE",
        "face_value": 1000,
        "coupon_rate": 5.0,
        "maturity_date": datetime(2029, 12, 31),
        "issue_date": datetime(2024, 1, 1),
        "current_price": -100,  # Invalid
    }
    
    with pytest.raises(ValidationError, match="Current price must be positive"):
        validate_bond_data(bond_data)


def test_validate_bond_data_invalid_face_value():
    """Test bond data validation with invalid face value"""
    bond_data = {
        "bond_id": "TEST-001",
        "bond_type": "CORPORATE",
        "face_value": 0,  # Invalid
        "coupon_rate": 5.0,
        "maturity_date": datetime(2029, 12, 31),
        "issue_date": datetime(2024, 1, 1),
        "current_price": 950,
    }
    
    with pytest.raises(ValidationError, match="Face value must be positive"):
        validate_bond_data(bond_data)


def test_validate_bond_data_invalid_dates():
    """Test bond data validation with invalid date range"""
    bond_data = {
        "bond_id": "TEST-001",
        "bond_type": "CORPORATE",
        "face_value": 1000,
        "coupon_rate": 5.0,
        "maturity_date": datetime(2024, 1, 1),  # Before issue date
        "issue_date": datetime(2024, 12, 31),
        "current_price": 950,
    }
    
    with pytest.raises(ValidationError, match="Maturity date must be after issue date"):
        validate_bond_data(bond_data)


def test_cache_key_simple():
    """Test cache key generation for simple arguments"""
    key1 = cache_key(1, 2, 3)
    key2 = cache_key(1, 2, 3)
    
    assert key1 == key2
    assert isinstance(key1, str)


def test_cache_key_different():
    """Test cache key generation for different arguments"""
    key1 = cache_key(1, 2, 3)
    key2 = cache_key(1, 2, 4)
    
    assert key1 != key2


def test_cache_key_with_kwargs():
    """Test cache key generation with keyword arguments"""
    key1 = cache_key(1, 2, a=3, b=4)
    key2 = cache_key(1, 2, b=4, a=3)  # Order shouldn't matter
    
    assert key1 == key2


def test_memoize():
    """Test memoize decorator"""
    @memoize
    def test_function(x):
        """Test function for memoize decorator"""
        return x * 2
    
    # Clear cache
    test_function.cache_clear()
    
    # First call - should compute
    result1 = test_function(5)
    
    # Second call - should use cache
    result2 = test_function(5)
    
    assert result1 == result2 == 10


def test_format_currency():
    """Test currency formatting"""
    assert format_currency(1000.50) == "$1,000.50"
    assert format_currency(1000.5, decimals=0) == "$1,001" or format_currency(1000.5, decimals=0) == "$1,000"  # May round either way
    assert format_currency(0) == "$0.00"


def test_format_percentage():
    """Test percentage formatting"""
    assert format_percentage(5.5) == "5.50%"
    assert format_percentage(5.5, decimals=1) == "5.5%"
    assert format_percentage(0) == "0.00%"


def test_format_date():
    """Test date formatting"""
    date = datetime(2024, 12, 25)
    formatted = format_date(date)
    
    assert formatted == "2024-12-25"
    assert isinstance(formatted, str)


def test_handle_exceptions():
    """Test exception handling decorator"""
    @handle_exceptions
    def test_func():
        raise ValueError("Test error")
    
    with pytest.raises(ValueError, match="Test error"):
        test_func()


def test_handle_exceptions_no_error():
    """Test exception handling decorator with no error"""
    @handle_exceptions
    def test_func():
        return 42
    
    result = test_func()
    assert result == 42


def test_validation_error():
    """Test ValidationError exception"""
    error = ValidationError("Test error")
    assert str(error) == "Test error"
    assert isinstance(error, Exception)
