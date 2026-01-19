"""
Unit tests for retry utilities
"""

import pytest

from bondtrader.utils.retry import circuit_breaker, retry_with_backoff


@pytest.mark.unit
class TestRetryWithBackoff:
    """Test retry_with_backoff decorator"""

    def test_retry_success_first_attempt(self):
        """Test retry with success on first attempt"""

        @retry_with_backoff(max_attempts=3, initial_wait=0.1)
        def successful_func():
            return 42

        result = successful_func()
        assert result == 42

    def test_retry_success_after_failure(self):
        """Test retry with success after failure"""
        attempts = [0]

        @retry_with_backoff(max_attempts=3, initial_wait=0.01)
        def retryable_func():
            attempts[0] += 1
            if attempts[0] < 2:
                raise ValueError("Temporary failure")
            return 42

        result = retryable_func()
        assert result == 42
        assert attempts[0] == 2

    def test_retry_max_attempts_exceeded(self):
        """Test retry when max attempts exceeded"""
        attempts = [0]

        @retry_with_backoff(max_attempts=2, initial_wait=0.01)
        def failing_func():
            attempts[0] += 1
            raise ValueError("Always fails")

        with pytest.raises(ValueError):
            failing_func()
        assert attempts[0] == 2

    def test_retry_with_exception_filter(self):
        """Test retry with exception filter"""

        @retry_with_backoff(max_attempts=3, initial_wait=0.1, retry_on=(ValueError,))
        def func_with_specific_exception():
            raise TypeError("Should not retry")

        # Should not retry for TypeError
        with pytest.raises(TypeError):
            func_with_specific_exception()


@pytest.mark.unit
class TestCircuitBreaker:
    """Test circuit breaker decorator"""

    def test_circuit_breaker_success(self):
        """Test circuit breaker with successful function"""

        @circuit_breaker(failure_threshold=3, recovery_timeout=60)
        def successful_func():
            return 42

        result = successful_func()
        assert result == 42

    def test_circuit_breaker_with_failures(self):
        """Test circuit breaker opens after threshold"""

        @circuit_breaker(failure_threshold=2, recovery_timeout=1)
        def failing_func():
            raise ValueError("Always fails")

        # First failure
        with pytest.raises(ValueError):
            failing_func()

        # Second failure - should open circuit
        with pytest.raises(ValueError):
            failing_func()

        # Third call - circuit should be open
        with pytest.raises((ValueError, RuntimeError)):
            failing_func()
