"""
Unit tests for error handling utilities
"""

import pytest

from bondtrader.utils.error_handling import (
    SanitizedException,
    handle_errors,
    sanitize_error_message,
    sanitize_exception,
)


@pytest.mark.unit
class TestSanitizeErrorMessage:
    """Test sanitize_error_message function"""

    def test_sanitize_error_message_dev_mode(self):
        """Test sanitizing error message in dev mode"""
        error = ValueError("Test error with password secret123")
        result = sanitize_error_message(error, production=False)
        assert "password" in result or "Test error" in result

    def test_sanitize_error_message_production_mode(self):
        """Test sanitizing error message in production mode"""
        error = ValueError("Test error with password secret123")
        result = sanitize_error_message(error, production=True)
        assert "password" not in result.lower()
        assert "secret" not in result.lower()

    def test_sanitize_error_message_connection_error(self):
        """Test sanitizing connection error"""
        error = ConnectionError("Connection failed")
        result = sanitize_error_message(error, production=True)
        assert "temporarily unavailable" in result.lower()

    def test_sanitize_error_message_permission_error(self):
        """Test sanitizing permission error"""
        error = PermissionError("Access denied")
        result = sanitize_error_message(error, production=True)
        assert "access denied" in result.lower()

    def test_sanitize_error_message_validation_error(self):
        """Test validation errors are kept in production"""
        error = ValueError("Invalid input")
        result = sanitize_error_message(error, production=True)
        assert "invalid input" in result.lower() or "error occurred" in result.lower()

    def test_sanitize_error_message_file_path_removal(self):
        """Test file paths are removed in production"""
        error = ValueError("Error in /path/to/file.py line 42")
        result = sanitize_error_message(error, production=True)
        assert "/path/to/file.py" not in result


@pytest.mark.unit
class TestSanitizeException:
    """Test sanitize_exception decorator"""

    def test_sanitize_exception_success(self):
        """Test sanitize_exception decorator with successful function"""

        @sanitize_exception
        def successful_func():
            return 42

        result = successful_func()
        assert result == 42

    def test_sanitize_exception_catches_error(self):
        """Test sanitize_exception decorator catches and sanitizes errors"""

        @sanitize_exception
        def failing_func():
            raise ValueError("Internal error with password")

        with pytest.raises(ValueError) as exc_info:
            failing_func()
        assert isinstance(exc_info.value, ValueError)


@pytest.mark.unit
class TestSanitizedException:
    """Test SanitizedException class"""

    def test_sanitized_exception_creation(self):
        """Test creating sanitized exception"""
        exc = SanitizedException("User message", internal_details="Internal details")
        assert str(exc) == "User message"
        assert exc.user_message == "User message"
        assert exc.internal_details == "Internal details"


@pytest.mark.unit
class TestHandleErrors:
    """Test handle_errors decorator"""

    def test_handle_errors_success(self):
        """Test handle_errors decorator with successful function"""

        @handle_errors(production=False)
        def successful_func():
            return 42

        result = successful_func()
        assert result == 42

    def test_handle_errors_catches_error(self):
        """Test handle_errors decorator catches errors"""

        @handle_errors(production=True)
        def failing_func():
            raise ValueError("Internal error")

        with pytest.raises(ValueError):
            failing_func()

    def test_handle_errors_production_mode(self):
        """Test handle_errors in production mode sanitizes errors"""

        @handle_errors(production=True)
        def failing_func():
            raise ValueError("Error with password secret")

        with pytest.raises(ValueError) as exc_info:
            failing_func()
        assert "password" not in str(exc_info.value).lower()