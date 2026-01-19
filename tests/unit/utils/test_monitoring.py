"""
Tests for monitoring utilities
"""

import os
import time

import pytest

from bondtrader.utils.monitoring import (
    PROMETHEUS_AVAILABLE,
    decrement_connections,
    get_metrics,
    increment_connections,
    record_cache_hit,
    record_cache_miss,
    track_api_request,
    track_ml_prediction,
    track_risk_calculation,
    track_valuation,
)


class TestMonitoringDecorators:
    """Test monitoring decorators"""

    def test_track_api_request_success(self):
        """Test API request tracking on success"""

        @track_api_request("GET", "/api/test")
        def test_endpoint():
            return {"status": "ok"}

        result = test_endpoint()
        assert result["status"] == "ok"

        # Metrics should be recorded (if prometheus available)
        metrics = get_metrics()
        assert isinstance(metrics, str)

    def test_track_api_request_error(self):
        """Test API request tracking on error"""

        @track_api_request("GET", "/api/test")
        def test_endpoint():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            test_endpoint()

        # Metrics should still be recorded
        metrics = get_metrics()
        assert isinstance(metrics, str)

    def test_track_valuation(self):
        """Test bond valuation tracking"""

        @track_valuation("CORPORATE")
        def calculate_value(bond):
            return 1000.0

        result = calculate_value(None)
        assert result == 1000.0

        metrics = get_metrics()
        assert isinstance(metrics, str)

    def test_track_ml_prediction(self):
        """Test ML prediction tracking"""

        @track_ml_prediction("random_forest", "v1.0")
        def predict(bond):
            return 950.0

        result = predict(None)
        assert result == 950.0

        metrics = get_metrics()
        assert isinstance(metrics, str)


class TestMonitoringFunctions:
    """Test monitoring utility functions"""

    def test_track_risk_calculation(self):
        """Test risk calculation tracking"""
        track_risk_calculation("var")
        track_risk_calculation("credit")

        metrics = get_metrics()
        assert isinstance(metrics, str)

    def test_cache_tracking(self):
        """Test cache hit/miss tracking"""
        record_cache_hit("calculation")
        record_cache_miss("calculation")
        record_cache_hit("data")

        metrics = get_metrics()
        assert isinstance(metrics, str)

    def test_connection_tracking(self):
        """Test connection tracking"""
        increment_connections()
        increment_connections()
        decrement_connections()

        metrics = get_metrics()
        assert isinstance(metrics, str)

    def test_get_metrics(self):
        """Test getting metrics"""
        metrics = get_metrics()
        assert isinstance(metrics, str)
        assert len(metrics) > 0


class TestMonitoringAvailability:
    """Test monitoring availability"""

    def test_prometheus_availability(self):
        """Test that we handle prometheus availability"""
        # Should work whether prometheus is available or not
        metrics = get_metrics()
        assert isinstance(metrics, str)

    def test_metrics_always_available(self):
        """Test that metrics functions work even without prometheus"""
        # These should not raise exceptions
        track_risk_calculation("test")
        record_cache_hit("test")
        record_cache_miss("test")
        increment_connections()
        decrement_connections()

        metrics = get_metrics()
        assert isinstance(metrics, str)
