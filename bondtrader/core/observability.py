"""
Observability Module
Metrics, Tracing, and Monitoring following industry best practices
"""

import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

from bondtrader.utils.utils import logger


class Metrics:
    """
    Metrics collection with business KPIs.

    CRITICAL: Tracks both technical and business metrics for Fortune 10 financial institutions.
    """

    def __init__(self):
        self._counters: Dict[str, int] = {}
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, list] = {}

        # Business metrics tracking
        self._trading_volume: float = 0.0
        self._total_pnl: float = 0.0
        self._portfolio_values: List[float] = []
        self._risk_metrics: Dict[str, float] = {}

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

    def track_trading_volume(self, volume: float, bond_type: Optional[str] = None):
        """
        Track trading volume (business metric).

        Args:
            volume: Trading volume in dollars
            bond_type: Optional bond type for categorization
        """
        self._trading_volume += volume
        self.increment("business.trading_volume", value=int(volume), tags={"bond_type": bond_type} if bond_type else None)
        self.gauge("business.total_trading_volume", self._trading_volume)

    def track_pnl(self, pnl: float, trade_id: Optional[str] = None):
        """
        Track profit and loss (business metric).

        Args:
            pnl: Profit/loss in dollars
            trade_id: Optional trade identifier
        """
        self._total_pnl += pnl
        self.gauge("business.total_pnl", self._total_pnl)
        self.histogram("business.trade_pnl", pnl, tags={"trade_id": trade_id} if trade_id else None)

    def track_portfolio_value(self, value: float):
        """
        Track portfolio value (business metric).

        Args:
            value: Portfolio value in dollars
        """
        self._portfolio_values.append(value)
        # Keep only last 1000 values to prevent memory issues
        if len(self._portfolio_values) > 1000:
            self._portfolio_values = self._portfolio_values[-1000:]

        self.gauge("business.portfolio.total_value", value)
        self.histogram("business.portfolio.value", value)

    def track_risk_metric(self, metric_name: str, value: float):
        """
        Track risk metric (business metric).

        Args:
            metric_name: Name of risk metric (e.g., "var_95", "credit_risk")
            value: Metric value
        """
        self._risk_metrics[metric_name] = value
        self.gauge(f"business.risk.{metric_name}", value)

    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics including business KPIs"""
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
            "business": {
                "total_trading_volume": self._trading_volume,
                "total_pnl": self._total_pnl,
                "current_portfolio_value": self._portfolio_values[-1] if self._portfolio_values else None,
                "risk_metrics": self._risk_metrics.copy(),
            },
        }

    def reset(self):
        """Reset all metrics"""
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()
        self._trading_volume = 0.0
        self._total_pnl = 0.0
        self._portfolio_values.clear()
        self._risk_metrics.clear()


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
