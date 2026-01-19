"""
API Middleware
Middleware functions for authentication, rate limiting, and CORS
"""

import os
from typing import Callable

from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from bondtrader.utils.rate_limiter import get_api_rate_limiter

# Rate limiter
rate_limiter = get_api_rate_limiter()

# API key authentication (optional)
security = HTTPBearer(auto_error=False)
API_KEY = os.getenv("API_KEY", None)
REQUIRE_API_KEY = os.getenv("REQUIRE_API_KEY", "false").lower() == "true"


def verify_api_key(credentials: HTTPAuthorizationCredentials = None) -> bool:
    """Verify API key if authentication is enabled"""
    if not REQUIRE_API_KEY:
        return True
    if not API_KEY:
        from fastapi import HTTPException

        raise HTTPException(status_code=500, detail="API key authentication required but not configured")
    if not credentials:
        from fastapi import HTTPException

        raise HTTPException(status_code=401, detail="API key required")
    if credentials.credentials != API_KEY:
        from fastapi import HTTPException

        raise HTTPException(status_code=403, detail="Invalid API key")
    return True


def get_client_ip(request: Request) -> str:
    """Extract client IP address from request"""
    if request.client:
        return request.client.host
    return "unknown"


async def rate_limit_middleware(request: Request, call_next: Callable):
    """Rate limiting middleware"""
    client_ip = get_client_ip(request)
    allowed, error = rate_limiter.is_allowed(client_ip)
    if not allowed:
        return JSONResponse(
            status_code=429,
            content={"detail": error},
            headers={"X-RateLimit-Limit": str(rate_limiter.max_requests), "X-RateLimit-Remaining": "0"},
        )
    response = await call_next(request)
    # Add rate limit headers
    remaining = rate_limiter.get_remaining(client_ip)
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    response.headers["X-RateLimit-Limit"] = str(rate_limiter.max_requests)
    return response


def setup_cors(app):
    """Setup CORS middleware"""
    allowed_origins = os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:8000,http://localhost:8501").split(",")
    if os.getenv("CORS_ALLOW_ALL", "false").lower() == "true":
        # Only allow wildcard in development with explicit flag
        allowed_origins = ["*"]

    from fastapi import FastAPI

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
