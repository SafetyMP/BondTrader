"""
Distributed Tracing with OpenTelemetry
Provides request tracing and performance profiling

CRITICAL: Required for production debugging and performance analysis
"""

import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar

from bondtrader.utils.utils import logger

T = TypeVar("T")

# OpenTelemetry availability
try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.trace import Status, StatusCode

    _otel_available = True
except ImportError:
    _otel_available = False
    logger.warning(
        "OpenTelemetry not available. Install with: pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp"
    )


class DistributedTracer:
    """
    Distributed Tracing Manager

    Provides OpenTelemetry-based distributed tracing for request tracking.
    """

    def __init__(self, service_name: str = "bondtrader", endpoint: Optional[str] = None):
        """
        Initialize distributed tracer.

        Args:
            service_name: Service name for traces
            endpoint: OTLP endpoint URL (optional, uses environment if not provided)
        """
        self.service_name = service_name
        self.tracer = None
        self._initialized = False

        if _otel_available:
            self._initialize_tracer(endpoint)
        else:
            logger.warning("OpenTelemetry not available. Using fallback tracing.")

    def _initialize_tracer(self, endpoint: Optional[str] = None):
        """Initialize OpenTelemetry tracer"""
        try:
            resource = Resource.create({"service.name": self.service_name})
            provider = TracerProvider(resource=resource)

            # Use endpoint from parameter or environment
            if endpoint is None:
                import os

                endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")

            # Add OTLP exporter if endpoint is configured
            if endpoint and endpoint != "disabled":
                exporter = OTLPSpanExporter(endpoint=endpoint)
                provider.add_span_processor(BatchSpanProcessor(exporter))

            trace.set_tracer_provider(provider)
            self.tracer = trace.get_tracer(self.service_name)
            self._initialized = True
            logger.info(f"Distributed tracing initialized for {self.service_name}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenTelemetry tracer: {e}")
            self._initialized = False

    def start_span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """
        Start a new trace span.

        Args:
            name: Span name
            attributes: Optional span attributes

        Returns:
            Span context manager
        """
        if self._initialized and self.tracer:

            @contextmanager
            def span_with_attributes():
                with self.tracer.start_as_current_span(name) as span:
                    if attributes:
                        for key, value in attributes.items():
                            try:
                                span.set_attribute(key, str(value))
                            except AttributeError:
                                # Try to get current span if set_attribute not available
                                current_span = trace.get_current_span()
                                if current_span and hasattr(current_span, "set_attribute"):
                                    current_span.set_attribute(key, str(value))
                    yield span

            return span_with_attributes()
        else:
            # Fallback: return no-op context manager
            return _NoOpSpan()

    def trace_function(self, name: Optional[str] = None, attributes: Optional[Dict[str, Any]] = None):
        """
        Decorator to trace function execution.

        Args:
            name: Optional span name (defaults to function name)
            attributes: Optional span attributes

        Usage:
            @tracer.trace_function()
            def my_function():
                ...
        """

        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            span_name = name or f"{func.__module__}.{func.__name__}"

            @wraps(func)
            def wrapper(*args, **kwargs) -> T:
                start_time = time.time()

                with self.start_span(span_name, attributes=attributes):
                    try:
                        result = func(*args, **kwargs)
                        duration = time.time() - start_time

                        # Add duration attribute
                        if self._initialized and self.tracer:
                            span = trace.get_current_span()
                            if span:
                                span.set_attribute("duration_ms", duration * 1000)
                                span.set_attribute("status", "success")

                        return result
                    except Exception as e:
                        duration = time.time() - start_time

                        # Mark span as error
                        if self._initialized and self.tracer:
                            span = trace.get_current_span()
                            if span:
                                span.set_attribute("duration_ms", duration * 1000)
                                span.set_attribute("status", "error")
                                span.set_attribute("error", str(e))
                                span.set_status(Status(StatusCode.ERROR, str(e)))

                        raise

            return wrapper

        return decorator


class _NoOpSpan:
    """No-op span for when OpenTelemetry is not available"""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def set_attribute(self, key: str, value: Any):
        pass

    def set_status(self, status):
        pass


# Global tracer instance
_tracer: Optional[DistributedTracer] = None


def get_tracer(service_name: str = "bondtrader", endpoint: Optional[str] = None) -> DistributedTracer:
    """Get or create global tracer instance"""
    global _tracer
    if _tracer is None:
        _tracer = DistributedTracer(service_name=service_name, endpoint=endpoint)
    return _tracer


@contextmanager
def trace_span(name: str, attributes: Optional[Dict[str, Any]] = None):
    """
    Context manager for tracing operations.

    Usage:
        with trace_span("calculate_valuation", {"bond_id": "BOND123"}):
            result = valuator.calculate_fair_value(bond)
    """
    tracer = get_tracer()
    with tracer.start_span(name, attributes=attributes):
        yield
