"""
Rate Limiting Utilities
Provides rate limiting for API endpoints and dashboard
"""

import os
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from threading import Lock
from typing import Dict, Optional, Tuple


class RateLimiter:
    """
    Token bucket rate limiter

    Implements token bucket algorithm for rate limiting
    """

    def __init__(self, max_requests: int = 100, time_window_seconds: int = 60, per_user: bool = True):
        """
        Initialize rate limiter

        Args:
            max_requests: Maximum requests allowed in time window
            time_window_seconds: Time window in seconds
            per_user: If True, rate limit per user/IP, else global
        """
        self.max_requests = max_requests
        self.time_window = timedelta(seconds=time_window_seconds)
        self.per_user = per_user

        # Store request timestamps per identifier
        self.requests: Dict[str, deque] = defaultdict(lambda: deque())
        self.lock = Lock()

    def is_allowed(self, identifier: str = "default") -> Tuple[bool, Optional[str]]:
        """
        Check if request is allowed

        Args:
            identifier: User identifier (IP, username, etc.)

        Returns:
            Tuple of (is_allowed, error_message)
        """
        if not self.per_user:
            identifier = "global"

        with self.lock:
            now = datetime.now()
            request_times = self.requests[identifier]

            # Remove old requests outside time window
            while request_times and (now - request_times[0]) > self.time_window:
                request_times.popleft()

            # Check if limit exceeded
            if len(request_times) >= self.max_requests:
                oldest_request = request_times[0]
                retry_after = (oldest_request + self.time_window - now).total_seconds()
                return False, f"Rate limit exceeded. Try again in {int(retry_after)} seconds."

            # Add current request
            request_times.append(now)
            return True, None

    def reset(self, identifier: str = "default"):
        """Reset rate limit for identifier"""
        with self.lock:
            if identifier in self.requests:
                del self.requests[identifier]

    def get_remaining(self, identifier: str = "default") -> int:
        """Get remaining requests in current window"""
        if not self.per_user:
            identifier = "global"

        with self.lock:
            now = datetime.now()
            request_times = self.requests[identifier]

            # Remove old requests
            while request_times and (now - request_times[0]) > self.time_window:
                request_times.popleft()

            return max(0, self.max_requests - len(request_times))


# Global rate limiters
_api_rate_limiter: Optional[RateLimiter] = None
_dashboard_rate_limiter: Optional[RateLimiter] = None


def get_api_rate_limiter() -> RateLimiter:
    """Get API rate limiter instance"""
    global _api_rate_limiter
    if _api_rate_limiter is None:
        max_requests = int(os.getenv("API_RATE_LIMIT", "100"))
        window = int(os.getenv("API_RATE_LIMIT_WINDOW", "60"))
        _api_rate_limiter = RateLimiter(max_requests=max_requests, time_window_seconds=window, per_user=True)
    return _api_rate_limiter


def get_dashboard_rate_limiter() -> RateLimiter:
    """Get dashboard rate limiter instance"""
    global _dashboard_rate_limiter
    if _dashboard_rate_limiter is None:
        max_requests = int(os.getenv("DASHBOARD_RATE_LIMIT", "200"))
        window = int(os.getenv("DASHBOARD_RATE_LIMIT_WINDOW", "60"))
        _dashboard_rate_limiter = RateLimiter(max_requests=max_requests, time_window_seconds=window, per_user=True)
    return _dashboard_rate_limiter


def rate_limit(max_requests: int = 100, window_seconds: int = 60):
    """
    Decorator for rate limiting functions

    Usage:
        @rate_limit(max_requests=10, window_seconds=60)
        def my_api_endpoint():
            ...
    """
    limiter = RateLimiter(max_requests=max_requests, time_window_seconds=window_seconds)

    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get identifier from request (IP, user, etc.)
            identifier = kwargs.get("user_id") or kwargs.get("ip_address") or "default"

            allowed, error = limiter.is_allowed(identifier)
            if not allowed:
                from fastapi import HTTPException

                raise HTTPException(status_code=429, detail=error)

            return func(*args, **kwargs)

        return wrapper

    return decorator
