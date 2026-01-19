"""
Unit tests for Redis cache utilities
"""

import pytest
from unittest.mock import MagicMock, patch


@pytest.mark.unit
class TestRedisCache:
    """Test RedisCache class"""

    @patch("bondtrader.utils.redis_cache.HAS_REDIS", True)
    @patch("bondtrader.utils.redis_cache.redis")
    def test_redis_cache_creation(self, mock_redis):
        """Test creating Redis cache"""
        from bondtrader.utils.redis_cache import RedisCache

        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_redis.Redis.return_value = mock_client

        cache = RedisCache(host="localhost", port=6379)
        assert cache.host == "localhost"
        assert cache.port == 6379

    @patch("bondtrader.utils.redis_cache.HAS_REDIS", False)
    def test_redis_cache_no_redis(self):
        """Test Redis cache when Redis not available"""
        from bondtrader.utils.redis_cache import RedisCache

        cache = RedisCache()
        assert cache.is_available() is False

    @patch("bondtrader.utils.redis_cache.HAS_REDIS", True)
    @patch("bondtrader.utils.redis_cache.redis")
    def test_redis_cache_get_set(self, mock_redis):
        """Test getting and setting values"""
        from bondtrader.utils.redis_cache import RedisCache
        import json

        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_client.get.return_value = json.dumps({"key": "value"})
        mock_client.set.return_value = True
        mock_redis.Redis.return_value = mock_client

        cache = RedisCache()
        
        # Test set
        result = cache.set("test_key", {"key": "value"}, ttl=3600)
        assert result is True

        # Test get
        value = cache.get("test_key")
        assert value == {"key": "value"}

    @patch("bondtrader.utils.redis_cache.HAS_REDIS", True)
    @patch("bondtrader.utils.redis_cache.redis")
    def test_redis_cache_delete(self, mock_redis):
        """Test deleting from cache"""
        from bondtrader.utils.redis_cache import RedisCache

        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_client.delete.return_value = 1
        mock_redis.Redis.return_value = mock_client

        cache = RedisCache()
        result = cache.delete("test_key")
        assert result is True

    @patch("bondtrader.utils.redis_cache.HAS_REDIS", True)
    @patch("bondtrader.utils.redis_cache.redis")
    def test_redis_cache_clear_pattern(self, mock_redis):
        """Test clearing cache by pattern"""
        from bondtrader.utils.redis_cache import RedisCache

        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_client.keys.return_value = ["bond:1", "bond:2"]
        mock_client.delete.return_value = 2
        mock_redis.Redis.return_value = mock_client

        cache = RedisCache()
        result = cache.clear_pattern("bond:*")
        assert result >= 0

    @patch("bondtrader.utils.redis_cache.HAS_REDIS", True)
    @patch("bondtrader.utils.redis_cache.redis")
    def test_redis_cache_ttl(self, mock_redis):
        """Test setting TTL"""
        from bondtrader.utils.redis_cache import RedisCache
        import json

        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_client.set.return_value = True
        mock_redis.Redis.return_value = mock_client

        cache = RedisCache()
        result = cache.set("test_key", {"key": "value"}, ttl=3600)
        assert result is True

    def test_redis_cache_is_available_false(self):
        """Test is_available returns False when not connected"""
        from bondtrader.utils.redis_cache import RedisCache

        with patch("bondtrader.utils.redis_cache.HAS_REDIS", False):
            cache = RedisCache()
            assert cache.is_available() is False