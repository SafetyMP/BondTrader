"""
API Helper Utilities
Shared error handling and response formatting for API endpoints
"""

import logging
from typing import Any, Callable, TypeVar

from fastapi import HTTPException

from bondtrader.core.exceptions import (
    BusinessRuleViolation,
    DataNotFoundError,
    InvalidBondError,
    MLError,
    RiskCalculationError,
    ValuationError,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


def handle_service_result(result: Any, default_error_message: str = "Operation failed") -> Any:
    """
    Handle Result type from service layer and convert to HTTPException if needed

    Args:
        result: Result object from service layer
        default_error_message: Default error message if error type is unknown

    Returns:
        Value from result if successful

    Raises:
        HTTPException: Appropriate HTTP exception based on error type
    """
    if result.is_err():
        error = result.error

        # Map domain exceptions to HTTP status codes
        if isinstance(error, DataNotFoundError):
            raise HTTPException(status_code=404, detail=str(error))
        elif isinstance(error, InvalidBondError):
            raise HTTPException(status_code=400, detail=str(error))
        elif isinstance(error, BusinessRuleViolation):
            raise HTTPException(status_code=409, detail=str(error))
        elif isinstance(error, ValuationError):
            raise HTTPException(status_code=500, detail=str(error))
        elif isinstance(error, MLError):
            raise HTTPException(status_code=503, detail=str(error))
        elif isinstance(error, RiskCalculationError):
            raise HTTPException(status_code=500, detail=str(error))
        else:
            # Log unexpected errors but don't expose internal details
            logger.error(f"Unexpected error: {error}", exc_info=True)
            raise HTTPException(status_code=500, detail=default_error_message)

    return result.value


def handle_api_errors(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to handle errors in API endpoints consistently

    Usage:
        @handle_api_errors
        async def my_endpoint():
            ...
    """

    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error")

    return wrapper
