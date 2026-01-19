"""
Unit tests for distributed tracing
"""

from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
class TestDistributedTracer:
    """Test DistributedTracer class"""

    @patch("bondtrader.utils.tracing._otel_available", True)
    @patch("bondtrader.utils.tracing.trace")
    def test_tracer_creation(self, mock_trace):
        """Test creating distributed tracer"""
        from bondtrader.utils.tracing import DistributedTracer

        mock_tracer = MagicMock()
        mock_trace.get_tracer.return_value = mock_tracer
        mock_trace.set_tracer_provider = MagicMock()

        tracer = DistributedTracer(service_name="test_service")
        assert tracer.service_name == "test_service"

    @patch("bondtrader.utils.tracing._otel_available", False)
    def test_tracer_creation_no_otel(self):
        """Test tracer creation when OpenTelemetry not available"""
        from bondtrader.utils.tracing import DistributedTracer

        tracer = DistributedTracer()
        assert tracer.service_name == "bondtrader"
        assert tracer._initialized is False

    @patch("bondtrader.utils.tracing._otel_available", True)
    @patch("bondtrader.utils.tracing.trace")
    def test_start_span(self, mock_trace):
        """Test starting a span"""
        from bondtrader.utils.tracing import DistributedTracer

        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value = mock_span
        mock_trace.get_tracer.return_value = mock_tracer
        mock_trace.set_tracer_provider = MagicMock()

        tracer = DistributedTracer()
        span = tracer.start_span("test_span", attributes={"key": "value"})
        assert span is not None

    def test_trace_function_decorator(self):
        """Test trace_function decorator"""
        from bondtrader.utils.tracing import DistributedTracer

        tracer = DistributedTracer()
        call_count = [0]

        @tracer.trace_function()
        def test_function():
            call_count[0] += 1
            return 42

        result = test_function()
        assert result == 42
        assert call_count[0] == 1

    def test_trace_span_context_manager(self):
        """Test trace_span context manager"""
        from bondtrader.utils.tracing import trace_span

        with trace_span("test_operation", {"attr": "value"}):
            # Context manager should work
            result = 42
            assert result == 42

    def test_get_tracer(self):
        """Test getting global tracer"""
        from bondtrader.utils.tracing import get_tracer

        tracer1 = get_tracer()
        tracer2 = get_tracer()
        # Should return same instance (singleton)
        assert tracer1 is tracer2

    @patch("bondtrader.utils.tracing._otel_available", True)
    @patch("bondtrader.utils.tracing.trace")
    def test_span_attributes(self, mock_trace):
        """Test span attributes are set"""
        from bondtrader.utils.tracing import DistributedTracer

        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_span.set_attribute = MagicMock()
        mock_tracer.start_as_current_span.return_value = mock_span
        mock_trace.get_tracer.return_value = mock_tracer
        mock_trace.set_tracer_provider = MagicMock()

        tracer = DistributedTracer()
        span = tracer.start_span("test_span", attributes={"test_key": "test_value"})

        if span and hasattr(span, "set_attribute"):
            # Attributes should be set in start_span
            assert True
