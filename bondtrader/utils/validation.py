"""
Input Validation Utilities
Provides decorators and functions for validating function inputs and data structures
"""

import functools
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from bondtrader.utils.utils import ValidationError, logger


def validate_bond_input(func: Callable) -> Callable:
    """Decorator to validate Bond inputs for functions"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Check if first argument after self is a Bond
        bond = None
        if args and hasattr(args[0], "__class__"):
            # If method, second arg is bond; if function, first arg is bond
            bond = (
                args[1]
                if len(args) > 1 and args[0].__class__.__name__ in ["BondValuator", "ArbitrageDetector"]
                else (args[0] if len(args) > 0 else None)
            )

        # Also check kwargs
        if bond is None:
            bond = kwargs.get("bond", None)

        if bond is not None:
            from bondtrader.core.bond_models import Bond

            if not isinstance(bond, Bond):
                raise TypeError(f"Expected Bond instance, got {type(bond)}")
            if bond.current_price <= 0:
                raise ValueError(f"Bond current_price must be positive, got {bond.current_price}")
            if bond.face_value <= 0:
                raise ValueError(f"Bond face_value must be positive, got {bond.face_value}")

        return func(*args, **kwargs)

    return wrapper


def validate_numeric_range(
    value: float,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    name: str = "value",
) -> None:
    """
    Validate numeric value is within specified range

    Args:
        value: Numeric value to validate
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        name: Name of the parameter for error messages

    Raises:
        ValueError: If value is outside specified range
        TypeError: If value is not numeric
    """
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric, got {type(value)}")

    if min_val is not None and value < min_val:
        raise ValueError(f"{name} must be >= {min_val}, got {value}")

    if max_val is not None and value > max_val:
        raise ValueError(f"{name} must be <= {max_val}, got {value}")


def validate_positive(value: float, name: str = "value") -> None:
    """
    Validate value is positive

    Args:
        value: Value to validate
        name: Name of the parameter for error messages

    Raises:
        ValueError: If value is not positive
    """
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric, got {type(value)}")

    if value <= 0:
        raise ValueError(f"{name} must be positive (greater than 0), got {value}")


def validate_percentage(value: float, name: str = "percentage") -> None:
    """
    Validate value is a valid percentage (0-100 or 0-1)

    Args:
        value: Percentage value to validate
        name: Name of the parameter for error messages

    Raises:
        ValueError: If value is not a valid percentage
    """
    validate_numeric_range(value, min_val=0.0, max_val=100.0, name=name)

    # Also check if it's in decimal form (0-1)
    if value > 1.0 and value <= 100.0:
        # Likely a percentage, but check if should be decimal
        logger.warning(f"{name} appears to be in percentage form ({value}), consider using decimal form (0-1)")


def validate_probability(value: float, name: str = "probability") -> None:
    """
    Validate value is a valid probability (0-1)

    Args:
        value: Probability value to validate
        name: Name of the parameter for error messages

    Raises:
        ValueError: If value is not between 0 and 1
    """
    validate_numeric_range(value, min_val=0.0, max_val=1.0, name=name)


def validate_list_not_empty(items: List[Any], name: str = "list") -> None:
    """
    Validate list is not empty

    Args:
        items: List to validate
        name: Name of the parameter for error messages

    Raises:
        ValueError: If list is empty
        TypeError: If not a list
    """
    if not isinstance(items, list):
        raise TypeError(f"{name} must be a list, got {type(items)}")
    if len(items) == 0:
        raise ValueError(f"{name} cannot be empty")


def validate_file_path(
    filepath: str,
    must_exist: bool = False,
    name: str = "filepath",
    allow_absolute: bool = False,
    allowed_extensions: List[str] = None,
) -> None:
    """
    Validate and sanitize file path for security

    Args:
        filepath: File path to validate
        must_exist: Whether file must exist
        name: Name of the parameter for error messages
        allow_absolute: Whether to allow absolute paths (default: False for security)
        allowed_extensions: List of allowed file extensions (e.g., ['.joblib', '.pkl'])

    Raises:
        ValueError: If path is invalid or contains unsafe components
        FileNotFoundError: If must_exist=True and file doesn't exist
        TypeError: If path is not a string
    """
    import os
    from pathlib import Path

    if not isinstance(filepath, str):
        raise TypeError(f"{name} must be a string, got {type(filepath)}")

    if not filepath or not filepath.strip():
        raise ValueError(f"{name} cannot be empty")

    # Remove leading/trailing whitespace
    filepath = filepath.strip()

    # Check for null bytes (path traversal attempt)
    if "\x00" in filepath:
        raise ValueError(f"{name} contains null bytes - potential security risk")

    # Check for directory traversal attempts
    if ".." in filepath or "//" in filepath:
        raise ValueError(f"{name} contains directory traversal components (.. or //) - not allowed")

    # Check for absolute paths if not allowed
    if not allow_absolute:
        if os.path.isabs(filepath):
            raise ValueError(f"{name} cannot be an absolute path for security reasons. Use relative paths.")
    else:
        # Warn about absolute paths even if allowed
        if os.path.isabs(filepath):
            logger.warning(f"{name} is an absolute path: {filepath}. Ensure this is intentional.")

    # Check file extension if specified
    if allowed_extensions:
        path_obj = Path(filepath)
        ext = path_obj.suffix.lower()
        if ext not in [e.lower() for e in allowed_extensions]:
            raise ValueError(f"{name} must have one of these extensions: {allowed_extensions}, got: {ext}")

    # Validate path components don't contain dangerous characters
    # Windows: < > : " | ? *
    # Unix: / (already checked above)
    dangerous_chars = ["<", ">", ":", '"', "|", "?", "*"]
    for char in dangerous_chars:
        if char in filepath:
            raise ValueError(f"{name} contains dangerous character '{char}' - not allowed")

    # Check for file existence if required
    if must_exist and not os.path.exists(filepath):
        raise FileNotFoundError(f"{name} does not exist: {filepath}")


def sanitize_file_path(filepath: str, base_dir: str = None) -> str:
    """
    Sanitize file path by resolving relative to base directory

    Args:
        filepath: File path to sanitize
        base_dir: Base directory to resolve relative paths (default: current directory)

    Returns:
        Sanitized, normalized file path

    Raises:
        ValueError: If path cannot be safely sanitized
    """
    import os
    from pathlib import Path

    # Validate input
    validate_file_path(filepath, allow_absolute=False, name="filepath")

    # Normalize path
    path_obj = Path(filepath)

    # If base_dir provided, resolve relative to it
    if base_dir:
        base_path = Path(base_dir).resolve()
        resolved_path = (base_path / path_obj).resolve()

        # Ensure resolved path is within base directory (prevent directory traversal)
        try:
            resolved_path.relative_to(base_path)
        except ValueError:
            raise ValueError(f"File path resolves outside base directory: {filepath}")

        return str(resolved_path)

    # Otherwise, just normalize relative to current directory
    return str(path_obj.resolve())


def validate_weights_sum(
    weights: List[float], expected_sum: float = 1.0, tolerance: float = 1e-6, name: str = "weights"
) -> None:
    """
    Validate list of weights sums to expected value

    Args:
        weights: List of weights
        expected_sum: Expected sum (default 1.0)
        tolerance: Tolerance for floating point comparison
        name: Name of the parameter for error messages

    Raises:
        ValueError: If weights don't sum to expected value
    """
    validate_list_not_empty(weights, name)

    total = sum(weights)
    if abs(total - expected_sum) > tolerance:
        raise ValueError(f"{name} must sum to {expected_sum} (within {tolerance}), got {total}")


def validate_credit_rating(rating: str, name: str = "credit_rating") -> None:
    """
    Validate credit rating format

    Args:
        rating: Credit rating string
        name: Name of the parameter for error messages

    Raises:
        ValueError: If rating is invalid
    """
    if not isinstance(rating, str):
        raise TypeError(f"{name} must be a string, got {type(rating)}")

    valid_ratings = [
        "AAA",
        "AA+",
        "AA",
        "AA-",
        "A+",
        "A",
        "A-",
        "BBB+",
        "BBB",
        "BBB-",
        "BB+",
        "BB",
        "BB-",
        "B+",
        "B",
        "B-",
        "CCC+",
        "CCC",
        "CCC-",
        "D",
        "NR",
    ]

    if rating.upper() not in valid_ratings:
        logger.warning(f"{name} '{rating}' not in standard ratings list, allowing but may cause issues")
