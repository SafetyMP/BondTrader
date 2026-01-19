"""
Tests for observability functionality
"""

import pytest

from bondtrader.core.observability import get_metrics, trace


@pytest.mark.unit
class TestMetrics:
    """Test metrics functionality"""

    def test_get_metrics_singleton(self):
        """Test get_metrics returns singleton"""
        metrics1 = get_metrics()
        metrics2 = get_metrics()
        assert metrics1 is metrics2

    def test_increment_counter(self):
        """Test incrementing counter"""
        metrics = get_metrics()
        metrics.increment("test.counter")
        # Just verify it doesn't raise

    def test_increment_with_tags(self):
        """Test incrementing counter with tags"""
        metrics = get_metrics()
        metrics.increment("test.counter", tags={"env": "test"})
        # Just verify it doesn't raise

    def test_histogram(self):
        """Test recording histogram"""
        metrics = get_metrics()
        metrics.histogram("test.histogram", 42.5)
        # Just verify it doesn't raise

    def test_gauge(self):
        """Test recording gauge"""
        metrics = get_metrics()
        metrics.gauge("test.gauge", 100)
        # Just verify it doesn't raise

    def test_get_metrics(self):
        """Test getting metrics data"""
        metrics = get_metrics()
        metrics.increment("test.counter")
        metrics.gauge("test.gauge", 100)
        metrics.histogram("test.histogram", 42.5)
        
        all_metrics = metrics.get_metrics()
        assert "counters" in all_metrics
        assert "gauges" in all_metrics
        assert "histograms" in all_metrics

    def test_reset_metrics(self):
        """Test resetting metrics"""
        metrics = get_metrics()
        metrics.increment("test.counter")
        metrics.reset()
        all_metrics = metrics.get_metrics()
        assert len(all_metrics["counters"]) == 0


@pytest.mark.unit
class TestTraceDecorator:
    """Test trace decorator"""

    def test_trace_decorator_success(self):
        """Test trace decorator with successful function"""
        @trace
        def test_func():
            return 42

        result = test_func()
        assert result == 42

    def test_trace_decorator_with_args(self):
        """Test trace decorator with arguments"""
        @trace
        def test_func(x, y):
            return x + y

        result = test_func(2, 3)
        assert result == 5

    def test_trace_decorator_with_kwargs(self):
        """Test trace decorator with keyword arguments"""
        @trace
        def test_func(x=1, y=2):
            return x + y

        result = test_func(x=3, y=4)
        assert result == 7

    def test_trace_decorator_error(self):
        """Test trace decorator with error"""
        @trace
        def test_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            test_func()
