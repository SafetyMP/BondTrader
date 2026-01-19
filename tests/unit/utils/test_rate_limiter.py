"""
Unit tests for rate limiting utilities
"""

import time
from datetime import timedelta
from unittest.mock import patch

import pytest

from bondtrader.utils.rate_limiter import (
    RateLimiter,
    get_api_rate_limiter,
    get_dashboard_rate_limiter,
)


@pytest.mark.unit
class TestRateLimiter:
    """Test RateLimiter class"""

    def test_rate_limiter_creation(self):
        """Test creating rate limiter"""
        limiter = RateLimiter(max_requests=10, time_window_seconds=60)
        assert limiter.max_requests == 10
        assert limiter.time_window == timedelta(seconds=60)

    def test_rate_limiter_allowed(self):
        """Test checking if request is allowed"""
        limiter = RateLimiter(max_requests=5, time_window_seconds=60)

        # First few requests should be allowed
        for i in range(5):
            allowed, error = limiter.is_allowed("user1")
            assert allowed is True
            assert error is None

    def test_rate_limiter_exceeded(self):
        """Test rate limit exceeded"""
        limiter = RateLimiter(max_requests=3, time_window_seconds=60)

        # Fill up the limit
        for i in range(3):
            limiter.is_allowed("user1")

        # Next request should be denied
        allowed, error = limiter.is_allowed("user1")
        assert allowed is False
        assert error is not None
        assert "Rate limit exceeded" in error

    def test_rate_limiter_per_user(self):
        """Test per-user rate limiting"""
        limiter = RateLimiter(max_requests=2, time_window_seconds=60, per_user=True)

        # User 1 fills limit
        limiter.is_allowed("user1")
        limiter.is_allowed("user1")

        # User 2 should still be allowed
        allowed, _ = limiter.is_allowed("user2")
        assert allowed is True

    def test_rate_limiter_global(self):
        """Test global rate limiting"""
        limiter = RateLimiter(max_requests=2, time_window_seconds=60, per_user=False)

        # Fill global limit
        limiter.is_allowed("user1")
        limiter.is_allowed("user2")

        # Any user should be denied
        allowed, _ = limiter.is_allowed("user3")
        assert allowed is False

    def test_reset(self):
        """Test resetting rate limit"""
        limiter = RateLimiter(max_requests=2, time_window_seconds=60)

        # Fill limit
        limiter.is_allowed("user1")
        limiter.is_allowed("user1")

        # Should be denied
        allowed, _ = limiter.is_allowed("user1")
        assert allowed is False

        # Reset
        limiter.reset("user1")

        # Should be allowed again
        allowed, _ = limiter.is_allowed("user1")
        assert allowed is True

    def test_get_remaining(self):
        """Test getting remaining requests"""
        limiter = RateLimiter(max_requests=10, time_window_seconds=60)

        # Initial remaining should be max
        remaining = limiter.get_remaining("user1")
        assert remaining == 10

        # After one request
        limiter.is_allowed("user1")
        remaining = limiter.get_remaining("user1")
        assert remaining == 9

    def test_time_window_expiry(self):
        """Test that old requests expire after time window"""
        limiter = RateLimiter(max_requests=2, time_window_seconds=1)

        # Fill limit
        limiter.is_allowed("user1")
        limiter.is_allowed("user1")

        # Should be denied
        allowed, _ = limiter.is_allowed("user1")
        assert allowed is False

        # Wait for time window to expire
        time.sleep(1.1)

        # Should be allowed again
        allowed, _ = limiter.is_allowed("user1")
        assert allowed is True


@pytest.mark.unit
class TestRateLimiterFunctions:
    """Test rate limiter helper functions"""

    @patch.dict("os.environ", {"API_RATE_LIMIT": "50", "API_RATE_LIMIT_WINDOW": "30"})
    def test_get_api_rate_limiter(self):
        """Test getting API rate limiter"""
        limiter = get_api_rate_limiter()
        assert limiter.max_requests == 50
        assert limiter.time_window == timedelta(seconds=30)

    @patch.dict("os.environ", {"DASHBOARD_RATE_LIMIT": "100", "DASHBOARD_RATE_LIMIT_WINDOW": "60"})
    def test_get_dashboard_rate_limiter(self):
        """Test getting dashboard rate limiter"""
        limiter = get_dashboard_rate_limiter()
        assert limiter.max_requests == 100
        assert limiter.time_window == timedelta(seconds=60)

    def test_rate_limit_decorator(self):
        """Test rate_limit decorator"""
        from bondtrader.utils.rate_limiter import rate_limit

        @rate_limit(max_requests=2, window_seconds=60)
        def test_endpoint(user_id=None, ip_address=None):
            return "success"

        # Should work with decorator
        result = test_endpoint(user_id="user1")
        assert result == "success"
