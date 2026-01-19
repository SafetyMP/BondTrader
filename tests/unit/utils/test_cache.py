"""
Unit tests for cache utilities
"""

import pytest

from bondtrader.utils.cache import ModelCache, cache_model, clear_bond_cache, get_cached_bond


@pytest.mark.unit
class TestModelCache:
    """Test ModelCache functionality"""

    def test_get_nonexistent(self):
        """Test getting non-existent cached model"""
        ModelCache.clear()
        result = ModelCache.get("nonexistent")
        assert result is None

    def test_set_and_get(self):
        """Test setting and getting cached model"""
        ModelCache.clear()
        test_model = {"type": "random_forest", "accuracy": 0.95}
        ModelCache.set("test_model", test_model)
        result = ModelCache.get("test_model")
        assert result == test_model

    def test_cache_clear(self):
        """Test clearing cache"""
        ModelCache.set("test_model", {"test": "data"})
        ModelCache.clear()
        assert ModelCache.get("test_model") is None

    def test_get_or_load_cache_hit(self):
        """Test get_or_load with cache hit"""
        ModelCache.clear()
        test_model = {"type": "random_forest"}
        ModelCache.set("test_model", test_model)

        loader_called = []

        def loader():
            loader_called.append(True)
            return {"loaded": "model"}

        result = ModelCache.get_or_load("test_model", loader)
        assert result == test_model
        assert len(loader_called) == 0  # Loader should not be called

    def test_get_or_load_cache_miss(self):
        """Test get_or_load with cache miss"""
        ModelCache.clear()

        loader_called = []

        def loader():
            loader_called.append(True)
            return {"loaded": "model"}

        result = ModelCache.get_or_load("test_model", loader)
        assert result == {"loaded": "model"}
        assert len(loader_called) == 1  # Loader should be called
        # Model should now be cached
        assert ModelCache.get("test_model") == {"loaded": "model"}

    def test_cache_size_limit(self):
        """Test cache size limit"""
        ModelCache.clear()
        ModelCache._cache_size_limit = 2  # Set small limit

        # Fill cache
        ModelCache.set("model1", {"id": 1})
        ModelCache.set("model2", {"id": 2})
        # Add third - should remove oldest
        ModelCache.set("model3", {"id": 3})

        # model1 should be removed
        assert ModelCache.get("model1") is None
        assert ModelCache.get("model2") is not None
        assert ModelCache.get("model3") is not None

        # Restore default
        ModelCache._cache_size_limit = 10


@pytest.mark.unit
class TestCacheHelpers:
    """Test cache helper functions"""

    def test_cache_model_key(self):
        """Test generating cache key for model"""
        key = cache_model("random_forest", "enhanced", False)
        assert key == "ml_model:random_forest:enhanced"

    def test_cache_model_key_ensemble(self):
        """Test generating cache key for ensemble model"""
        key = cache_model("random_forest", "enhanced", True)
        assert key == "ml_model:random_forest:enhanced:ensemble"

    def test_get_cached_bond_none(self):
        """Test getting cached bond (returns None)"""
        clear_bond_cache()
        result = get_cached_bond("TEST-001")
        assert result is None

    def test_clear_bond_cache(self):
        """Test clearing bond cache"""
        # Call function to populate cache
        get_cached_bond("TEST-001")
        clear_bond_cache()
        # Should work without error
        assert True
