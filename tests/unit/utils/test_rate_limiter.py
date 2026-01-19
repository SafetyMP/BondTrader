"""
Tests for rate limiting utilities
"""

import time
from datetime import datetime, timedelta

import pytest

from bondtrader.utils.rate_limiter import (
    RateLimiter,
    get_api_rate_limiter,
    get_dashboard_rate_limiter,
)


class TestRateLimiter:
    """Test RateLimiter functionality"""

    def test_rate_limiter_init(self):
        """Test RateLimiter initialization"""
        limiter = RateLimiter(max_requests=10, time_window_seconds=60)
        assert limiter.max_requests == 10
        assert limiter.time_window == timedelta(seconds=60)

    def test_rate_limiter_allows_requests(self):
        """Test that rate limiter allows requests within limit"""
        limiter = RateLimiter(max_requests=5, time_window_seconds=60)

        for i in range(5):
            allowed, error = limiter.is_allowed("user1")
            assert allowed is True
            assert error is None

    def test_rate_limiter_blocks_excess_requests(self):
        """Test that rate limiter blocks requests over limit"""
        limiter = RateLimiter(max_requests=3, time_window_seconds=60)

        # Make 3 requests (should all be allowed)
        for i in range(3):
            allowed, error = limiter.is_allowed("user1")
            assert allowed is True

        # 4th request should be blocked
        allowed, error = limiter.is_allowed("user1")
        assert allowed is False
        assert error is not None
        assert "Rate limit exceeded" in error

    def test_rate_limiter_per_user(self):
        """Test per-user rate limiting"""
        limiter = RateLimiter(max_requests=2, time_window_seconds=60, per_user=True)

        # User1 makes 2 requests
        assert limiter.is_allowed("user1")[0] is True
        assert limiter.is_allowed("user1")[0] is True
        assert limiter.is_allowed("user1")[0] is False  # Blocked

        # User2 should still be able to make requests
        assert limiter.is_allowed("user2")[0] is True
        assert limiter.is_allowed("user2")[0] is True
        assert limiter.is_allowed("user2")[0] is False  # Blocked

    def test_rate_limiter_global(self):
        """Test global rate limiting"""
        limiter = RateLimiter(max_requests=2, time_window_seconds=60, per_user=False)

        # User1 makes 2 requests
        assert limiter.is_allowed("user1")[0] is True
        assert limiter.is_allowed("user1")[0] is True

        # User2 should also be blocked (global limit)
        assert limiter.is_allowed("user2")[0] is False

    def test_rate_limiter_reset(self):
        """Test resetting rate limit"""
        limiter = RateLimiter(max_requests=2, time_window_seconds=60)

        # Make 2 requests
        assert limiter.is_allowed("user1")[0] is True
        assert limiter.is_allowed("user1")[0] is True
        assert limiter.is_allowed("user1")[0] is False  # Blocked

        # Reset
        limiter.reset("user1")

        # Should be able to make requests again
        assert limiter.is_allowed("user1")[0] is True

    def test_get_remaining(self):
        """Test getting remaining requests"""
        limiter = RateLimiter(max_requests=5, time_window_seconds=60)

        # Initially 5 remaining
        assert limiter.get_remaining("user1") == 5

        # Make 2 requests
        limiter.is_allowed("user1")
        limiter.is_allowed("user1")

        # Should have 3 remaining
        assert limiter.get_remaining("user1") == 3

    def test_rate_limiter_window_expiry(self):
        """Test that old requests expire after time window"""
        limiter = RateLimiter(max_requests=2, time_window_seconds=1)  # 1 second window

        # Make 2 requests
        assert limiter.is_allowed("user1")[0] is True
        assert limiter.is_allowed("user1")[0] is True
        assert limiter.is_allowed("user1")[0] is False  # Blocked

        # Wait for window to expire
        time.sleep(1.1)

        # Should be able to make requests again
        assert limiter.is_allowed("user1")[0] is True


class TestGlobalRateLimiters:
    """Test global rate limiter instances"""

    def test_get_api_rate_limiter(self):
        """Test getting API rate limiter"""
        limiter = get_api_rate_limiter()
        assert limiter is not None
        assert isinstance(limiter, RateLimiter)

    def test_get_dashboard_rate_limiter(self):
        """Test getting dashboard rate limiter"""
        limiter = get_dashboard_rate_limiter()
        assert limiter is not None
        assert isinstance(limiter, RateLimiter)

    def test_rate_limiters_are_singletons(self):
        """Test that rate limiters are singletons"""
        limiter1 = get_api_rate_limiter()
        limiter2 = get_api_rate_limiter()
        assert limiter1 is limiter2
