"""
Error Handling and Message Sanitization
Prevents information leakage in error messages

CRITICAL: Required for security in Fortune 10 financial institutions
"""

import traceback
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

from bondtrader.utils.utils import logger

T = TypeVar("T")

# Production mode flag (should be set via environment variable)
PRODUCTION_MODE = False


def sanitize_error_message(error: Exception, production: bool = None) -> str:
    """
    Sanitize error message to prevent information leakage.

    CRITICAL: In production, removes internal details from error messages.

    Args:
        error: Exception object
        production: Whether in production mode (default: uses global PRODUCTION_MODE)

    Returns:
        Sanitized error message
    """
    if production is None:
        production = PRODUCTION_MODE

    if not production:
        # Development mode: return full error message
        return str(error)

    # Production mode: sanitize error message
    error_type = type(error).__name__
    error_message = str(error)

    # Remove file paths
    if "/" in error_message or "\\" in error_message:
        error_message = "Internal error occurred"

    # Remove stack trace references
    if "Traceback" in error_message or "File" in error_message:
        error_message = "Internal error occurred"

    # Remove sensitive patterns
    sensitive_patterns = [
        "password",
        "secret",
        "key",
        "token",
        "api_key",
        "connection string",
        "database",
    ]

    error_lower = error_message.lower()
    for pattern in sensitive_patterns:
        if pattern in error_lower:
            error_message = "Internal error occurred"
            break

    # Generic error messages for common exceptions
    if "ConnectionError" in error_type or "Timeout" in error_type:
        return "Service temporarily unavailable. Please try again later."
    elif "PermissionError" in error_type or "AccessDenied" in error_type:
        return "Access denied."
    elif "NotFound" in error_type or "DoesNotExist" in error_type:
        return "Resource not found."
    elif "ValidationError" in error_type:
        # Keep validation errors (they're user-facing)
        return error_message
    else:
        # Generic error for unknown exceptions
        return "An error occurred. Please contact support if the problem persists."


def sanitize_exception(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to sanitize exceptions in production.

    Usage:
        @sanitize_exception
        def my_function():
            ...
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> T:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Log full error internally
            logger.error(f"Error in {func.__name__}: {e}", exc_info=True)

            # Re-raise with sanitized message
            sanitized_msg = sanitize_error_message(e)
            raise type(e)(sanitized_msg) from None

    return wrapper


class SanitizedException(Exception):
    """
    Exception that automatically sanitizes its message in production.
    """

    def __init__(self, message: str, internal_details: Optional[str] = None):
        """
        Initialize sanitized exception.

        Args:
            message: User-facing error message
            internal_details: Internal error details (logged but not exposed)
        """
        super().__init__(message)
        self.user_message = message
        self.internal_details = internal_details

        # Log internal details if provided
        if internal_details and not PRODUCTION_MODE:
            logger.debug(f"Internal error details: {internal_details}")


def handle_errors(production: bool = None):
    """
    Decorator to handle errors with sanitization.

    Args:
        production: Whether in production mode

    Usage:
        @handle_errors(production=True)
        def my_function():
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Log full error
                logger.error(f"Error in {func.__name__}: {e}", exc_info=True)

                # Sanitize and re-raise
                sanitized = sanitize_error_message(e, production=production)
                raise type(e)(sanitized) from None

        return wrapper

    return decorator
