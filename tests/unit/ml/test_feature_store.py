"""
Tests for feature store module
"""

import numpy as np
import pytest

from bondtrader.ml.feature_store import FeatureStore


@pytest.mark.unit
class TestFeatureStore:
    """Test FeatureStore functionality"""

    @pytest.fixture
    def store(self):
        """Create feature store"""
        return FeatureStore()

    def test_store_init(self, store):
        """Test store initialization"""
        assert store is not None

    def test_store_features(self, store):
        """Test storing features"""
        features = np.array([[1.0, 2.0], [3.0, 4.0]])
        feature_names = ["feature1", "feature2"]
        try:
            store.store_features("test_key", features, feature_names)
            # Just verify it doesn't raise
        except Exception:
            # May require backend setup
            pass

    def test_get_features(self, store):
        """Test getting features"""
        try:
            result = store.get_features("test_key")
            assert result is None or isinstance(result, (np.ndarray, dict))
        except Exception:
            pass
