"""
Retry Logic and Circuit Breaker Utilities
Provides resilience patterns for external service calls and database operations

CRITICAL: Prevents cascading failures and improves system reliability
"""

from functools import wraps
from typing import Any, Callable, Optional, Type, TypeVar

# Try to import tenacity, fall back to simple retry if not available
try:
    from tenacity import (
        RetryError,
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
    )

    HAS_TENACITY = True
except ImportError:
    HAS_TENACITY = False
    import time

# Try to import circuitbreaker, fall back to simple implementation if not available
try:
    from circuitbreaker import circuit

    HAS_CIRCUITBREAKER = True
except ImportError:
    HAS_CIRCUITBREAKER = False

from bondtrader.utils.utils import logger

T = TypeVar("T")


def retry_with_backoff(
    max_attempts: int = 3,
    initial_wait: float = 1.0,
    max_wait: float = 10.0,
    exponential_base: float = 2.0,
    retry_on: Optional[tuple] = None,
):
    """
    Decorator for retrying function calls with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        initial_wait: Initial wait time in seconds
        max_wait: Maximum wait time in seconds
        exponential_base: Base for exponential backoff
        retry_on: Tuple of exception types to retry on (None = all exceptions)

    Example:
        @retry_with_backoff(max_attempts=3, initial_wait=1.0)
        def fetch_data():
            # Will retry up to 3 times with exponential backoff
            ...
    """
    if HAS_TENACITY:
        # Use tenacity for robust retry logic
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            if retry_on:
                # Retry only on specific exceptions
                retry_exceptions = retry_on
                retry_condition = retry_if_exception_type(retry_exceptions)
                # Apply retry decorator with exception filter
                retry_decorator = retry(
                    stop=stop_after_attempt(max_attempts),
                    wait=wait_exponential(multiplier=initial_wait, min=initial_wait, max=max_wait),
                    retry=retry_condition,
                    reraise=True,
                )
            else:
                # Retry on all exceptions - don't pass retry parameter
                retry_decorator = retry(
                    stop=stop_after_attempt(max_attempts),
                    wait=wait_exponential(multiplier=initial_wait, min=initial_wait, max=max_wait),
                    reraise=True,
                )
            
            # Wrap function with retry and preserve metadata
            wrapped = retry_decorator(func)
            wrapped = wraps(func)(wrapped)
            
            return wrapped

        return decorator
    else:
        # Fallback: Simple retry implementation
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> T:
                wait_time = initial_wait
                last_exception = None

                for attempt in range(max_attempts):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        if attempt < max_attempts - 1:
                            if retry_on is None or isinstance(e, retry_on):
                                logger.warning(
                                    f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {e}. "
                                    f"Retrying in {wait_time:.2f}s..."
                                )
                                time.sleep(wait_time)
                                wait_time = min(wait_time * exponential_base, max_wait)
                            else:
                                # Don't retry on this exception type
                                raise
                        else:
                            logger.error(f"All {max_attempts} attempts failed for {func.__name__}")
                            raise

                # Should never reach here, but satisfy type checker
                if last_exception:
                    raise last_exception
                raise RuntimeError(f"Unexpected error in retry wrapper for {func.__name__}")

            return wrapper

        return decorator


def circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: int = 60,
    expected_exception: Type[Exception] = Exception,
):
    """
    Decorator for circuit breaker pattern to prevent cascading failures.

    Args:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Seconds to wait before attempting recovery
        expected_exception: Exception type that triggers circuit breaker

    Example:
        @circuit_breaker(failure_threshold=5, recovery_timeout=60)
        def call_external_api():
            # Will open circuit after 5 failures, wait 60s before retry
            ...
    """
    if HAS_CIRCUITBREAKER:
        # Use circuitbreaker library
        return circuit(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exception=expected_exception,
        )
    else:
        # Fallback: Simple circuit breaker implementation
        import threading
        from collections import deque
        from datetime import datetime, timedelta

        circuit_state = {"open": False, "failures": deque(maxlen=failure_threshold), "last_failure": None}
        lock = threading.Lock()

        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> T:
                with lock:
                    # Check if circuit is open
                    if circuit_state["open"]:
                        if circuit_state["last_failure"]:
                            time_since_failure = (datetime.now() - circuit_state["last_failure"]).total_seconds()
                            if time_since_failure < recovery_timeout:
                                raise RuntimeError(
                                    f"Circuit breaker is OPEN for {func.__name__}. "
                                    f"Retry after {recovery_timeout - time_since_failure:.0f} seconds."
                                )
                            else:
                                # Attempt recovery
                                circuit_state["open"] = False
                                circuit_state["failures"].clear()
                                logger.info(f"Circuit breaker attempting recovery for {func.__name__}")
                        else:
                            # No last failure time, attempt recovery anyway
                            circuit_state["open"] = False
                            circuit_state["failures"].clear()

                try:
                    result = func(*args, **kwargs)
                    # Success - reset failures if circuit was recovering
                    with lock:
                        if circuit_state["failures"]:
                            circuit_state["failures"].clear()
                            circuit_state["open"] = False
                    return result
                except expected_exception as e:
                    with lock:
                        circuit_state["failures"].append(datetime.now())
                        circuit_state["last_failure"] = datetime.now()

                        if len(circuit_state["failures"]) >= failure_threshold:
                            circuit_state["open"] = True
                            logger.error(f"Circuit breaker OPENED for {func.__name__} after {failure_threshold} failures")

                    raise

            return wrapper

        return decorator
