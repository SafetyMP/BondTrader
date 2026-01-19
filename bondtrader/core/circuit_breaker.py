"""
Circuit Breaker Pattern Implementation
Industry-standard pattern for handling external service failures
Prevents cascading failures and provides fallback mechanisms
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Any, Callable, Dict, Optional

from bondtrader.core.exceptions import ExternalServiceError
from bondtrader.utils.utils import logger


class CircuitState(Enum):
    """Circuit breaker states"""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""

    failure_threshold: int = 5  # Open circuit after N failures
    timeout: float = 60.0  # Time before attempting half-open (seconds)
    success_threshold: int = 2  # Close circuit after N successes in half-open
    expected_exception: type = Exception  # Exception type to catch


class CircuitBreaker:
    """
    Circuit breaker for external service calls

    Prevents cascading failures by:
    1. Opening circuit after threshold failures
    2. Rejecting requests when open
    3. Testing recovery in half-open state
    4. Closing circuit when service recovers
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        fallback: Optional[Callable] = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.fallback = fallback

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.lock = Lock()

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Call function through circuit breaker

        Args:
            func: Function to call
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result or fallback result

        Raises:
            ExternalServiceError: If circuit is open and no fallback
        """
        with self.lock:
            # Check circuit state
            if self.state == CircuitState.OPEN:
                # Check if timeout has passed
                if self.last_failure_time and (time.time() - self.last_failure_time) >= self.config.timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    logger.info(f"Circuit breaker {self.name} transitioning to HALF_OPEN")
                else:
                    # Circuit is open, reject request
                    if self.fallback:
                        logger.warning(f"Circuit breaker {self.name} is OPEN, using fallback")
                        return self.fallback(*args, **kwargs)
                    else:
                        raise ExternalServiceError(
                            f"Circuit breaker {self.name} is OPEN - service unavailable",
                            error_code="CIRCUIT_OPEN",
                        )

        # Try to call function
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.config.expected_exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        """Handle successful call"""
        with self.lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    logger.info(f"Circuit breaker {self.name} CLOSED - service recovered")
            else:
                # Reset failure count on success
                self.failure_count = 0

    def _on_failure(self):
        """Handle failed call"""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.state == CircuitState.HALF_OPEN:
                # Failed during half-open, open circuit again
                self.state = CircuitState.OPEN
                logger.warning(f"Circuit breaker {self.name} OPEN - service still failing")
            elif self.failure_count >= self.config.failure_threshold:
                # Threshold reached, open circuit
                self.state = CircuitState.OPEN
                logger.error(
                    f"Circuit breaker {self.name} OPEN - {self.failure_count} failures "
                    f"(threshold: {self.config.failure_threshold})"
                )

    def reset(self):
        """Manually reset circuit breaker"""
        with self.lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = None
            logger.info(f"Circuit breaker {self.name} manually reset")

    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state"""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
        }


# Global circuit breakers registry
_circuit_breakers: Dict[str, CircuitBreaker] = {}


def get_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
    """Get or create circuit breaker"""
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(name, config)
    return _circuit_breakers[name]


def circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None, fallback: Optional[Callable] = None):
    """
    Decorator for circuit breaker pattern

    Usage:
        @circuit_breaker("fred_api", config=CircuitBreakerConfig(failure_threshold=3))
        def fetch_fred_data():
            ...
    """
    breaker = get_circuit_breaker(name, config)

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)

        return wrapper

    return decorator
