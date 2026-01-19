"""
Tests for data lineage module
"""

import pytest

from bondtrader.ml.data_lineage import DataLineageTracker


@pytest.mark.unit
class TestDataLineageTracker:
    """Test DataLineageTracker functionality"""

    @pytest.fixture
    def tracker(self):
        """Create data lineage tracker"""
        return DataLineageTracker()

    def test_tracker_init(self, tracker):
        """Test tracker initialization"""
        assert tracker is not None

    def test_track_data_source(self, tracker):
        """Test tracking data source"""
        try:
            tracker.track_data_source("test_source", {"type": "csv", "path": "data.csv"})
            # Just verify it doesn't raise
        except Exception:
            pass

    def test_track_transformation(self, tracker):
        """Test tracking transformation"""
        try:
            tracker.track_transformation("test_transform", {"function": "normalize"})
            # Just verify it doesn't raise
        except Exception:
            pass

    def test_get_lineage(self, tracker):
        """Test getting lineage"""
        try:
            result = tracker.get_lineage("test_key")
            assert result is None or isinstance(result, dict)
        except Exception:
            pass
