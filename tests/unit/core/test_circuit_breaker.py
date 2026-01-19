"""
Tests for circuit breaker pattern
"""

import pytest
import time

from bondtrader.core.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    get_circuit_breaker,
    circuit_breaker,
)
from bondtrader.core.exceptions import ExternalServiceError


@pytest.mark.unit
class TestCircuitBreaker:
    """Test CircuitBreaker functionality"""

    def test_circuit_breaker_init(self):
        """Test circuit breaker initialization"""
        cb = CircuitBreaker("test", config=CircuitBreakerConfig(failure_threshold=5, timeout=60))
        assert cb.name == "test"
        assert cb.config.failure_threshold == 5
        assert cb.config.timeout == 60
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_circuit_breaker_call_success(self):
        """Test calling function through circuit breaker"""
        cb = CircuitBreaker("test", config=CircuitBreakerConfig(failure_threshold=5, timeout=60))

        def successful_func():
            return 42

        result = cb.call(successful_func)
        assert result == 42
        assert cb.state == CircuitState.CLOSED

    def test_circuit_breaker_call_failure(self):
        """Test calling function that fails through circuit breaker"""
        cb = CircuitBreaker("test", config=CircuitBreakerConfig(failure_threshold=2, timeout=60))

        def failing_func():
            raise ValueError("Test error")

        # First failure
        with pytest.raises(ValueError):
            cb.call(failing_func)
        assert cb.failure_count == 1

        # Second failure opens circuit
        with pytest.raises(ValueError):
            cb.call(failing_func)
        assert cb.state == CircuitState.OPEN

    def test_circuit_breaker_call_when_open(self):
        """Test calling function when circuit is open"""
        cb = CircuitBreaker("test", config=CircuitBreakerConfig(failure_threshold=2, timeout=60))
        # Open the circuit
        def failing_func():
            raise ValueError("Test")
        try:
            cb.call(failing_func)
        except ValueError:
            pass
        try:
            cb.call(failing_func)
        except ValueError:
            pass
        assert cb.state == CircuitState.OPEN

        def any_func():
            return 42

        # Should raise ExternalServiceError when circuit is open
        with pytest.raises(ExternalServiceError):
            cb.call(any_func)

    def test_circuit_breaker_fallback(self):
        """Test circuit breaker with fallback"""
        def fallback_func():
            return "fallback_value"

        cb = CircuitBreaker("test", config=CircuitBreakerConfig(failure_threshold=2, timeout=60), fallback=fallback_func)
        
        # Open the circuit
        def failing_func():
            raise ValueError("Test")
        try:
            cb.call(failing_func)
        except ValueError:
            pass
        try:
            cb.call(failing_func)
        except ValueError:
            pass
        assert cb.state == CircuitState.OPEN

        # Should use fallback instead of raising
        result = cb.call(lambda: None)
        assert result == "fallback_value"

    def test_circuit_breaker_timeout(self):
        """Test circuit breaker timeout transition to half-open"""
        cb = CircuitBreaker("test", config=CircuitBreakerConfig(failure_threshold=2, timeout=0.1))
        
        # Open the circuit
        def failing_func():
            raise ValueError("Test")
        try:
            cb.call(failing_func)
        except ValueError:
            pass
        try:
            cb.call(failing_func)
        except ValueError:
            pass
        assert cb.state == CircuitState.OPEN

        # Wait for timeout
        time.sleep(0.15)
        
        # Next call should transition to HALF_OPEN
        def dummy():
            pass
        cb.call(dummy)
        # State should be HALF_OPEN or CLOSED depending on success

    def test_circuit_breaker_half_open_success(self):
        """Test half-open state with success"""
        cb = CircuitBreaker("test", config=CircuitBreakerConfig(failure_threshold=2, timeout=0.1, success_threshold=1))
        
        # Open the circuit
        def failing_func():
            raise ValueError("Test")
        try:
            cb.call(failing_func)
        except ValueError:
            pass
        try:
            cb.call(failing_func)
        except ValueError:
            pass
        time.sleep(0.15)  # Wait for timeout

        def successful_func():
            return 42

        # Call should succeed and close circuit
        result = cb.call(successful_func)
        assert result == 42
        assert cb.state == CircuitState.CLOSED

    def test_circuit_breaker_half_open_failure(self):
        """Test half-open state with failure"""
        cb = CircuitBreaker("test", config=CircuitBreakerConfig(failure_threshold=2, timeout=0.1))
        
        # Open the circuit
        def failing_func():
            raise ValueError("Test")
        try:
            cb.call(failing_func)
        except ValueError:
            pass
        try:
            cb.call(failing_func)
        except ValueError:
            pass
        time.sleep(0.15)  # Wait for timeout

        # Failure in half-open should open circuit again
        with pytest.raises(ValueError):
            cb.call(failing_func)
        assert cb.state == CircuitState.OPEN

    def test_circuit_breaker_reset(self):
        """Test manually resetting circuit breaker"""
        cb = CircuitBreaker("test", config=CircuitBreakerConfig(failure_threshold=2, timeout=60))
        
        # Open the circuit
        def failing_func():
            raise ValueError("Test")
        try:
            cb.call(failing_func)
        except ValueError:
            pass
        try:
            cb.call(failing_func)
        except ValueError:
            pass
        assert cb.state == CircuitState.OPEN

        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_circuit_breaker_get_state(self):
        """Test getting circuit breaker state"""
        cb = CircuitBreaker("test", config=CircuitBreakerConfig(failure_threshold=5, timeout=60))
        state = cb.get_state()
        assert state["name"] == "test"
        assert state["state"] == CircuitState.CLOSED.value
        assert state["failure_count"] == 0

    def test_get_circuit_breaker_singleton(self):
        """Test get_circuit_breaker returns singleton"""
        cb1 = get_circuit_breaker("test_breaker")
        cb2 = get_circuit_breaker("test_breaker")
        assert cb1 is cb2

    def test_circuit_breaker_decorator(self):
        """Test circuit breaker decorator"""
        @circuit_breaker("decorator_test", config=CircuitBreakerConfig(failure_threshold=5, timeout=60))
        def test_func():
            return 42

        result = test_func()
        assert result == 42
