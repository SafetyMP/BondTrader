"""
Caching Utilities for Performance Optimization
Provides in-memory caching for ML models and frequently accessed data

CRITICAL: Reduces model loading time and database queries for better performance
"""

from functools import lru_cache
from typing import Any, Callable, Dict, Optional, TypeVar

T = TypeVar("T")


class ModelCache:
    """
    Cache for ML models to avoid reloading on every request.

    CRITICAL: Significantly improves performance by caching trained models in memory.
    """

    _cache: Dict[str, Any] = {}
    _cache_size_limit: int = 10  # Maximum number of models to cache

    @classmethod
    def get(cls, key: str) -> Optional[Any]:
        """Get cached model"""
        return cls._cache.get(key)

    @classmethod
    def set(cls, key: str, value: Any) -> None:
        """Set cached model with size limit"""
        # If cache is full, remove oldest entry (simple FIFO)
        if len(cls._cache) >= cls._cache_size_limit and key not in cls._cache:
            # Remove first (oldest) entry
            oldest_key = next(iter(cls._cache))
            del cls._cache[oldest_key]

        cls._cache[key] = value

    @classmethod
    def clear(cls) -> None:
        """Clear all cached models"""
        cls._cache.clear()

    @classmethod
    def get_or_load(cls, key: str, loader: Callable[[], T]) -> T:
        """
        Get model from cache or load using provided loader function.

        Args:
            key: Cache key (e.g., "ml_model:random_forest:enhanced")
            loader: Function that loads the model if not in cache

        Returns:
            Cached or newly loaded model
        """
        cached = cls.get(key)
        if cached is not None:
            return cached

        # Load model
        model = loader()
        cls.set(key, model)
        return model


def cache_model(model_type: str, feature_level: str = "basic", use_ensemble: bool = False) -> str:
    """
    Generate cache key for ML model.

    Args:
        model_type: Model type (e.g., "random_forest")
        feature_level: Feature level (e.g., "enhanced")
        use_ensemble: Whether ensemble is used

    Returns:
        Cache key string
    """
    ensemble_suffix = ":ensemble" if use_ensemble else ""
    return f"ml_model:{model_type}:{feature_level}{ensemble_suffix}"


# LRU cache for frequently accessed data
@lru_cache(maxsize=1000)
def get_cached_bond(bond_id: str) -> Optional[Any]:
    """
    Get cached bond data (decorated with LRU cache).

    Note: This is a placeholder - actual implementation would load from database.
    Use this pattern for frequently accessed data.
    """
    # In production, this would load from database
    # For now, return None (cache miss)
    return None


def clear_bond_cache() -> None:
    """Clear bond cache"""
    get_cached_bond.cache_clear()
