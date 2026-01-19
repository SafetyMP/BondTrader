"""
Tests for data validation module
"""

import numpy as np
import pandas as pd
import pytest

from bondtrader.ml.data_validation import DataValidator


@pytest.mark.unit
class TestDataValidator:
    """Test DataValidator functionality"""

    @pytest.fixture
    def validator(self):
        """Create data validator"""
        return DataValidator()

    def test_validator_init(self, validator):
        """Test validator initialization"""
        assert validator is not None

    def test_validate_schema(self, validator):
        """Test validating schema"""
        X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = validator.validate_schema(X)
        assert isinstance(result, dict)
        assert "errors" in result
        assert "warnings" in result
        assert "checks" in result

    def test_validate_statistics(self, validator):
        """Test validating statistics"""
        X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        y = np.array([1.0, 2.0])
        result = validator.validate_statistics(X, y)
        assert isinstance(result, dict)
        assert "errors" in result
        assert "warnings" in result
        assert "checks" in result

    def test_validate_complete(self, validator):
        """Test complete validation"""
        X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        y = np.array([1.0, 2.0])
        result = validator.validate_complete(X, y)
        assert result is not None
        assert hasattr(result, "passed")
        assert hasattr(result, "checks")
        assert hasattr(result, "errors")
