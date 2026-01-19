"""
Observability Module
Metrics, Tracing, and Monitoring following industry best practices
"""

import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, Optional

from bondtrader.utils.utils import logger


class Metrics:
    """
    Simple metrics collection
    In production, integrate with Prometheus, StatsD, or similar
    """

    def __init__(self):
        self._counters: Dict[str, int] = {}
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, list] = {}

    def increment(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric"""
        key = self._key_with_tags(name, tags)
        self._counters[key] = self._counters.get(key, 0) + value
        logger.debug(f"METRIC: {key} += {value}")

    def gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set a gauge metric"""
        key = self._key_with_tags(name, tags)
        self._gauges[key] = value
        logger.debug(f"METRIC: {key} = {value}")

    def histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a histogram value"""
        key = self._key_with_tags(name, tags)
        if key not in self._histograms:
            self._histograms[key] = []
        self._histograms[key].append(value)
        logger.debug(f"METRIC: {key} recorded {value}")

    def _key_with_tags(self, name: str, tags: Optional[Dict[str, str]]) -> str:
        """Create metric key with tags"""
        if not tags:
            return name
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}[{tag_str}]"

    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics"""
        return {
            "counters": self._counters.copy(),
            "gauges": self._gauges.copy(),
            "histograms": {
                k: {
                    "count": len(v),
                    "min": min(v) if v else None,
                    "max": max(v) if v else None,
                    "avg": sum(v) / len(v) if v else None,
                }
                for k, v in self._histograms.items()
            },
        }

    def reset(self):
        """Reset all metrics"""
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()


# Global metrics instance
_metrics: Optional[Metrics] = None


def get_metrics() -> Metrics:
    """Get global metrics instance"""
    global _metrics
    if _metrics is None:
        _metrics = Metrics()
    return _metrics


def trace(func: Callable = None, name: Optional[str] = None):
    """
    Decorator for distributed tracing
    Records function execution time and logs trace spans

    Usage:
        @trace
        def my_function():
            ...

        @trace(name="custom_operation")
        def another_function():
            ...
    """

    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def wrapper(*args, **kwargs):
            operation_name = name or f"{f.__module__}.{f.__name__}"
            start_time = time.time()

            # Create trace span
            span_id = f"{int(time.time() * 1000000)}"
            logger.info(f"TRACE START: {operation_name}", extra={"span_id": span_id, "operation": operation_name})

            try:
                result = f(*args, **kwargs)
                duration = time.time() - start_time

                # Record metrics
                get_metrics().histogram(f"{operation_name}.duration", duration)
                get_metrics().increment(f"{operation_name}.success")

                logger.info(
                    f"TRACE END: {operation_name}",
                    extra={
                        "span_id": span_id,
                        "operation": operation_name,
                        "duration_ms": duration * 1000,
                        "status": "success",
                    },
                )

                return result
            except Exception as e:
                duration = time.time() - start_time

                # Record metrics
                get_metrics().histogram(f"{operation_name}.duration", duration)
                get_metrics().increment(f"{operation_name}.error")

                logger.error(
                    f"TRACE ERROR: {operation_name}",
                    extra={
                        "span_id": span_id,
                        "operation": operation_name,
                        "duration_ms": duration * 1000,
                        "status": "error",
                        "error": str(e),
                    },
                    exc_info=True,
                )
                raise

        return wrapper

    if func is None:
        return decorator
    return decorator(func)


@contextmanager
def trace_context(operation_name: str):
    """
    Context manager for tracing operations

    Usage:
        with trace_context("calculate_valuation"):
            result = valuator.calculate_fair_value(bond)
    """
    start_time = time.time()
    span_id = f"{int(time.time() * 1000000)}"

    logger.info(f"TRACE START: {operation_name}", extra={"span_id": span_id, "operation": operation_name})

    try:
        yield span_id
        duration = time.time() - start_time
        get_metrics().histogram(f"{operation_name}.duration", duration)
        get_metrics().increment(f"{operation_name}.success")

        logger.info(
            f"TRACE END: {operation_name}",
            extra={"span_id": span_id, "operation": operation_name, "duration_ms": duration * 1000, "status": "success"},
        )
    except Exception as e:
        duration = time.time() - start_time
        get_metrics().histogram(f"{operation_name}.duration", duration)
        get_metrics().increment(f"{operation_name}.error")

        logger.error(
            f"TRACE ERROR: {operation_name}",
            extra={
                "span_id": span_id,
                "operation": operation_name,
                "duration_ms": duration * 1000,
                "status": "error",
                "error": str(e),
            },
            exc_info=True,
        )
        raise
