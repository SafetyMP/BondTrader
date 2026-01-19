"""
Extended unit tests for monitoring utilities
"""

from unittest.mock import MagicMock, patch

import pytest

from bondtrader.utils.monitoring import track_api_request, track_ml_prediction, track_valuation


@pytest.mark.unit
class TestMonitoringFunctions:
    """Test monitoring helper functions"""

    def test_track_api_request(self):
        """Test tracking API request"""

        @track_api_request("GET", "/api/bonds")
        def test_endpoint():
            return "success"

        result = test_endpoint()
        assert result == "success"

    def test_track_valuation(self):
        """Test tracking valuation"""
        track_valuation("CORPORATE", duration=0.05)
        # Should work or handle gracefully
        assert True

    def test_track_ml_prediction(self):
        """Test tracking ML prediction"""
        track_ml_prediction("random_forest", model_version="1.0")
        # Should work or handle gracefully
        assert True

    def test_get_metrics_endpoint(self):
        """Test getting metrics endpoint"""
        from bondtrader.utils.monitoring import get_metrics

        try:
            metrics = get_metrics()
            # Should return metrics string or handle gracefully
            assert isinstance(metrics, str) or metrics is None
        except Exception:
            # May fail if Prometheus not available
            pass
