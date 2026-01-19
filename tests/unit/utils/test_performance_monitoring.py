"""
Unit tests for performance monitoring utilities
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from bondtrader.utils.performance_monitoring import (
    Alert,
    PerformanceMonitor,
    PerformanceThreshold,
    get_performance_monitor,
)


@pytest.mark.unit
class TestPerformanceThreshold:
    """Test PerformanceThreshold class"""

    def test_threshold_creation(self):
        """Test creating performance threshold"""
        threshold = PerformanceThreshold("api.response_time", 1000, 5000, "greater")
        assert threshold.metric_name == "api.response_time"
        assert threshold.warning_threshold == 1000
        assert threshold.critical_threshold == 5000
        assert threshold.comparison == "greater"


@pytest.mark.unit
class TestAlert:
    """Test Alert class"""

    def test_alert_creation(self):
        """Test creating alert"""
        alert = Alert(
            severity="warning",
            metric_name="api.response_time",
            value=1500,
            threshold=1000,
            message="Response time exceeded",
            timestamp=datetime.now(),
        )
        assert alert.severity == "warning"
        assert alert.metric_name == "api.response_time"
        assert alert.value == 1500

    def test_alert_to_dict(self):
        """Test converting alert to dictionary"""
        alert = Alert(
            severity="critical",
            metric_name="api.error_rate",
            value=6.0,
            threshold=5.0,
            message="Error rate too high",
            timestamp=datetime.now(),
        )
        alert_dict = alert.to_dict()
        assert alert_dict["severity"] == "critical"
        assert alert_dict["metric_name"] == "api.error_rate"
        assert "timestamp" in alert_dict


@pytest.mark.unit
class TestPerformanceMonitor:
    """Test PerformanceMonitor class"""

    def test_performance_monitor_creation(self):
        """Test creating performance monitor"""
        monitor = PerformanceMonitor()
        assert monitor is not None
        assert len(monitor.thresholds) > 0  # Should have default thresholds

    def test_add_threshold(self):
        """Test adding threshold"""
        monitor = PerformanceMonitor()
        monitor.add_threshold("custom.metric", 100, 500, "greater")
        assert "custom.metric" in monitor.thresholds
        threshold = monitor.thresholds["custom.metric"]
        assert threshold.warning_threshold == 100
        assert threshold.critical_threshold == 500

    def test_check_threshold_warning(self):
        """Test checking threshold - warning level"""
        monitor = PerformanceMonitor()
        monitor.add_threshold("test.metric", 100, 500, "greater")

        alert = monitor.check_threshold("test.metric", 150)
        assert alert is not None
        assert alert.severity == "warning"
        assert alert.value == 150

    def test_check_threshold_critical(self):
        """Test checking threshold - critical level"""
        monitor = PerformanceMonitor()
        monitor.add_threshold("test.metric", 100, 500, "greater")

        alert = monitor.check_threshold("test.metric", 600)
        assert alert is not None
        assert alert.severity == "critical"

    def test_check_threshold_no_alert(self):
        """Test checking threshold - no alert"""
        monitor = PerformanceMonitor()
        monitor.add_threshold("test.metric", 100, 500, "greater")

        alert = monitor.check_threshold("test.metric", 50)
        assert alert is None

    def test_check_threshold_less_comparison(self):
        """Test threshold with 'less' comparison"""
        monitor = PerformanceMonitor()
        monitor.add_threshold("test.throughput", 10, 5, "less")

        alert = monitor.check_threshold("test.throughput", 3)
        assert alert is not None
        assert alert.severity == "critical"

    def test_get_recent_alerts(self):
        """Test getting recent alerts"""
        monitor = PerformanceMonitor()
        monitor.add_threshold("test.metric", 100, 500, "greater")

        # Generate some alerts
        monitor.check_threshold("test.metric", 150)
        monitor.check_threshold("test.metric", 600)

        recent = monitor.get_recent_alerts(minutes=60)
        assert len(recent) >= 2

    def test_get_critical_alerts(self):
        """Test getting critical alerts"""
        monitor = PerformanceMonitor()
        monitor.add_threshold("test.metric", 100, 500, "greater")

        # Generate warning and critical alerts
        monitor.check_threshold("test.metric", 150)  # Warning
        monitor.check_threshold("test.metric", 600)  # Critical

        critical = monitor.get_critical_alerts()
        assert len(critical) >= 1
        assert all(alert.severity == "critical" for alert in critical)

    @patch("bondtrader.utils.performance_monitoring.get_metrics")
    def test_get_performance_summary(self, mock_get_metrics):
        """Test getting performance summary"""
        mock_metrics = MagicMock()
        mock_metrics.get_metrics.return_value = {"api.response_time": 1000}
        mock_get_metrics.return_value = mock_metrics

        monitor = PerformanceMonitor()
        monitor.add_threshold("api.response_time", 1000, 5000, "greater")
        monitor.check_threshold("api.response_time", 1500)

        summary = monitor.get_performance_summary()
        assert "metrics" in summary
        assert "recent_alerts" in summary
        assert "thresholds" in summary


@pytest.mark.unit
class TestPerformanceMonitorFunctions:
    """Test performance monitor helper functions"""

    def test_get_performance_monitor(self):
        """Test getting global performance monitor"""
        monitor1 = get_performance_monitor()
        monitor2 = get_performance_monitor()
        # Should return same instance (singleton)
        assert monitor1 is monitor2
