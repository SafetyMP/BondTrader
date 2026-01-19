"""
Result Pattern Implementation
Industry-standard error handling pattern for functional programming style
Allows explicit error handling without exceptions for business logic
"""

from dataclasses import dataclass
from typing import Callable, Generic, Optional, TypeVar, Union

T = TypeVar("T")
E = TypeVar("E", bound=Exception)
U = TypeVar("U")  # For map operations
F = TypeVar("F", bound=Exception)  # For map_err operations


@dataclass(frozen=True)
class Result(Generic[T, E]):
    """
    Result type for explicit error handling

    Either contains a value (Ok) or an error (Err)
    Similar to Rust Result<T, E> or Haskell Either

    Usage:
        result = calculate_ytm(bond)
        if result.is_ok():
            ytm = result.value
        else:
            error = result.error
    """

    _value: Optional[T] = None
    _error: Optional[E] = None
    _is_ok: bool = True

    def __post_init__(self):
        if self._value is None and self._error is None:
            raise ValueError("Result must have either value or error")
        if self._value is not None and self._error is not None:
            raise ValueError("Result cannot have both value and error")

    @classmethod
    def ok(cls, value: T) -> "Result[T, E]":
        """Create a successful result"""
        return cls(_value=value, _is_ok=True)

    @classmethod
    def err(cls, error: E) -> "Result[T, E]":
        """Create an error result"""
        return cls(_error=error, _is_ok=False)

    def is_ok(self) -> bool:
        """Check if result is successful"""
        return self._is_ok

    def is_err(self) -> bool:
        """Check if result is an error"""
        return not self._is_ok

    @property
    def value(self) -> T:
        """Get the value (raises if error)"""
        if not self._is_ok:
            raise ValueError(f"Cannot get value from error result: {self._error}")
        return self._value

    @property
    def error(self) -> E:
        """Get the error (raises if ok)"""
        if self._is_ok:
            raise ValueError("Cannot get error from ok result")
        return self._error

    def unwrap(self) -> T:
        """Unwrap value, raising exception if error"""
        if not self._is_ok:
            raise self._error
        return self._value

    def unwrap_or(self, default: T) -> T:
        """Return value or default if error"""
        return self._value if self._is_ok else default

    def unwrap_or_else(self, func: Callable[[E], T]) -> T:
        """Return value or compute from error"""
        if self._is_ok:
            return self._value
        return func(self._error)

    def map(self, func: Callable[[T], U]) -> "Result[U, E]":
        """Map over the value if ok"""
        if self._is_ok:
            try:
                return Result.ok(func(self._value))
            except Exception as e:
                return Result.err(e)
        return Result.err(self._error)

    def map_err(self, func: Callable[[E], F]) -> "Result[T, F]":
        """Map over the error if error"""
        if not self._is_ok:
            try:
                return Result.err(func(self._error))
            except Exception as e:
                return Result.err(e)
        return Result.ok(self._value)

    def and_then(self, func: Callable[[T], "Result[U, E]"]) -> "Result[U, E]":  # noqa: F821
        """Chain operations that return Result"""
        if self._is_ok:
            return func(self._value)
        return Result.err(self._error)

    def __repr__(self):
        if self._is_ok:
            return f"Ok({self._value})"
        return f"Err({self._error})"


def safe(func: Callable[..., T]) -> Callable[..., Result[T, Exception]]:
    """
    Decorator to wrap function in Result type

    Usage:
        @safe
        def risky_operation():
            return 42

        result = risky_operation()
        if result.is_ok():
            print(result.value)
    """

    def wrapper(*args, **kwargs) -> Result[T, Exception]:
        try:
            return Result.ok(func(*args, **kwargs))
        except Exception as e:
            return Result.err(e)

    return wrapper
