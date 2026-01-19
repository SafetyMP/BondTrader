"""
Tests for Result pattern implementation
"""

import pytest

from bondtrader.core.exceptions import BondTraderException, ValuationError
from bondtrader.core.result import Result


class TestResult:
    """Test Result pattern functionality"""

    def test_result_ok(self):
        """Test creating successful result"""
        result = Result.ok(42)
        assert result.is_ok()
        assert not result.is_err()
        assert result.value == 42

    def test_result_err(self):
        """Test creating error result"""
        error = ValueError("Test error")
        result = Result.err(error)
        assert result.is_err()
        assert not result.is_ok()
        assert result.error == error

    def test_result_unwrap(self):
        """Test unwrapping successful result"""
        result = Result.ok(42)
        assert result.unwrap() == 42

    def test_result_unwrap_error(self):
        """Test unwrapping error result raises"""
        error = ValueError("Test error")
        result = Result.err(error)
        with pytest.raises(ValueError):
            result.unwrap()

    def test_result_unwrap_or(self):
        """Test unwrap_or with ok result"""
        result = Result.ok(42)
        assert result.unwrap_or(0) == 42

    def test_result_unwrap_or_error(self):
        """Test unwrap_or with error result"""
        result = Result.err(ValueError("Test"))
        assert result.unwrap_or(0) == 0

    def test_result_unwrap_or_else(self):
        """Test unwrap_or_else"""
        result = Result.ok(42)
        assert result.unwrap_or_else(lambda e: 0) == 42

        result = Result.err(ValueError("Test"))
        assert result.unwrap_or_else(lambda e: 0) == 0

    def test_result_map(self):
        """Test mapping over ok result"""
        result = Result.ok(2)
        mapped = result.map(lambda x: x * 2)
        assert mapped.is_ok()
        assert mapped.value == 4

    def test_result_map_error(self):
        """Test mapping over error result"""
        result = Result.err(ValueError("Test"))
        mapped = result.map(lambda x: x * 2)
        assert mapped.is_err()

    def test_result_map_err(self):
        """Test mapping error"""
        result = Result.err(ValueError("Test"))
        mapped = result.map_err(lambda e: RuntimeError(str(e)))
        assert mapped.is_err()
        assert isinstance(mapped.error, RuntimeError)

    def test_result_and_then(self):
        """Test chaining results"""
        result = Result.ok(2)
        chained = result.and_then(lambda x: Result.ok(x * 2))
        assert chained.is_ok()
        assert chained.value == 4

    def test_result_and_then_error(self):
        """Test chaining with error"""
        result = Result.err(ValueError("Test"))
        chained = result.and_then(lambda x: Result.ok(x * 2))
        assert chained.is_err()

    def test_result_repr(self):
        """Test result representation"""
        ok_result = Result.ok(42)
        assert "Ok(42)" in repr(ok_result)

        err_result = Result.err(ValueError("Test"))
        assert "Err" in repr(err_result)

    def test_result_value_property_error(self):
        """Test accessing value on error result"""
        result = Result.err(ValueError("Test"))
        with pytest.raises(ValueError, match="Cannot get value"):
            _ = result.value

    def test_result_error_property_ok(self):
        """Test accessing error on ok result"""
        result = Result.ok(42)
        with pytest.raises(ValueError, match="Cannot get error"):
            _ = result.error

    def test_result_post_init_validation(self):
        """Test result validation"""
        # Should raise if neither value nor error
        with pytest.raises(ValueError, match="must have either"):
            Result(_value=None, _error=None, _is_ok=True)

        # Should raise if both value and error
        with pytest.raises(ValueError, match="cannot have both"):
            Result(_value=42, _error=ValueError("Test"), _is_ok=True)


@pytest.mark.unit
class TestSafeDecorator:
    """Test safe decorator"""

    def test_safe_decorator_success(self):
        """Test safe decorator with successful function"""
        from bondtrader.core.result import safe

        @safe
        def successful_func():
            return 42

        result = successful_func()
        assert result.is_ok()
        assert result.value == 42

    def test_safe_decorator_error(self):
        """Test safe decorator with error"""
        from bondtrader.core.result import safe

        @safe
        def failing_func():
            raise ValueError("Test error")

        result = failing_func()
        assert result.is_err()
        assert isinstance(result.error, ValueError)
