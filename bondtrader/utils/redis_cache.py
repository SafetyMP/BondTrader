"""
Redis Caching Implementation
Provides distributed caching for production scalability

CRITICAL: Required for production performance in Fortune 10 financial institutions
"""

from functools import wraps
from typing import Any, Callable, Optional, TypeVar

# Try to import Redis
try:
    import redis
    from redis.exceptions import ConnectionError, RedisError

    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

from bondtrader.utils.utils import logger

T = TypeVar("T")


class RedisCache:
    """
    Redis-based caching for frequently accessed data.

    CRITICAL: Provides distributed caching for production scalability.
    """

    def __init__(
        self, host: str = "localhost", port: int = 6379, db: int = 0, password: Optional[str] = None
    ):
        """
        Initialize Redis cache.

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Optional Redis password
        """
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self._client: Optional[Any] = None
        self._connected = False

        if HAS_REDIS:
            try:
                self._client = redis.Redis(
                    host=host,
                    port=port,
                    db=db,
                    password=password,
                    decode_responses=True,
                    socket_connect_timeout=5,
                )
                # Test connection
                self._client.ping()
                self._connected = True
                logger.info(f"Connected to Redis at {host}:{port}")
            except (ConnectionError, RedisError) as e:
                logger.warning(f"Redis not available: {e}. Caching disabled.")
                self._connected = False
        else:
            logger.warning("Redis not installed. Install with: pip install redis")
            self._connected = False

    def is_available(self) -> bool:
        """Check if Redis is available"""
        if not self._connected:
            return False
        try:
            self._client.ping()
            return True
        except Exception:
            self._connected = False
            return False

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        if not self.is_available():
            return None

        try:
            import json

            value = self._client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.warning(f"Error getting from Redis cache: {e}")
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (None = no expiration)

        Returns:
            True if set successfully
        """
        if not self.is_available():
            return False

        try:
            import json

            serialized = json.dumps(value, default=str)
            if ttl:
                self._client.setex(key, ttl, serialized)
            else:
                self._client.set(key, serialized)
            return True
        except Exception as e:
            logger.warning(f"Error setting Redis cache: {e}")
            return False

    def delete(self, key: str) -> bool:
        """
        Delete key from cache.

        Args:
            key: Cache key

        Returns:
            True if deleted
        """
        if not self.is_available():
            return False

        try:
            self._client.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Error deleting from Redis cache: {e}")
            return False

    def clear_pattern(self, pattern: str) -> int:
        """
        Clear all keys matching pattern.

        Args:
            pattern: Redis key pattern (e.g., "bond:*")

        Returns:
            Number of keys deleted
        """
        if not self.is_available():
            return 0

        try:
            keys = self._client.keys(pattern)
            if keys:
                return self._client.delete(*keys)
            return 0
        except Exception as e:
            logger.warning(f"Error clearing Redis cache pattern: {e}")
            return 0

    def get_or_set(self, key: str, loader: Callable[[], T], ttl: Optional[int] = 3600) -> T:
        """
        Get value from cache or load using provided function.

        Args:
            key: Cache key
            loader: Function to load value if not in cache
            ttl: Time to live in seconds

        Returns:
            Cached or newly loaded value
        """
        # Try to get from cache
        cached = self.get(key)
        if cached is not None:
            return cached

        # Load value
        value = loader()

        # Cache it
        self.set(key, value, ttl=ttl)

        return value


# Global Redis cache instance
_redis_cache: Optional[RedisCache] = None


def get_redis_cache() -> RedisCache:
    """Get or create global Redis cache instance"""
    global _redis_cache
    if _redis_cache is None:
        import os

        host = os.getenv("REDIS_HOST", "localhost")
        port = int(os.getenv("REDIS_PORT", "6379"))
        password = os.getenv("REDIS_PASSWORD")
        _redis_cache = RedisCache(host=host, port=port, password=password)
    return _redis_cache


def cache_result(key_prefix: str, ttl: int = 3600):
    """
    Decorator to cache function results in Redis.

    Args:
        key_prefix: Prefix for cache keys
        ttl: Time to live in seconds

    Example:
        @cache_result("bond_valuation", ttl=3600)
        def calculate_valuation(bond_id: str):
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            import hashlib
            import json

            cache_key = f"{key_prefix}:{func.__name__}:{hashlib.md5(json.dumps((args, kwargs), sort_keys=True).encode()).hexdigest()}"

            # Try to get from cache
            redis_cache = get_redis_cache()
            cached = redis_cache.get(cache_key)
            if cached is not None:
                return cached

            # Execute function
            result = func(*args, **kwargs)

            # Cache result
            redis_cache.set(cache_key, result, ttl=ttl)

            return result

        return wrapper

    return decorator
