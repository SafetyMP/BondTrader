"""
Monitoring and Metrics Utilities
Provides Prometheus metrics integration
"""

import os
import time
from datetime import datetime
from functools import wraps
from typing import Any, Dict, Optional

try:
    from prometheus_client import Counter, Gauge, Histogram, generate_latest, start_http_server

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

    # Create dummy classes for when prometheus_client is not available
    class Counter:
        def __init__(self, *args, **kwargs):
            pass

        def inc(self, *args, **kwargs):
            pass

        def labels(self, *args, **kwargs):
            return self

    class Histogram:
        def __init__(self, *args, **kwargs):
            pass

        def observe(self, *args, **kwargs):
            pass

        def labels(self, *args, **kwargs):
            return self

        def time(self):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class Gauge:
        def __init__(self, *args, **kwargs):
            pass

        def set(self, *args, **kwargs):
            pass

        def inc(self, *args, **kwargs):
            pass

        def dec(self, *args, **kwargs):
            pass


# Define metrics
if PROMETHEUS_AVAILABLE:
    # API metrics
    api_requests_total = Counter(
        "bondtrader_api_requests_total",
        "Total number of API requests",
        ["method", "endpoint", "status"],
    )

    api_request_duration = Histogram(
        "bondtrader_api_request_duration_seconds",
        "API request duration in seconds",
        ["method", "endpoint"],
    )

    # Bond valuation metrics
    bond_valuations_total = Counter("bondtrader_valuations_total", "Total number of bond valuations", ["bond_type"])

    valuation_duration = Histogram(
        "bondtrader_valuation_duration_seconds", "Bond valuation duration in seconds", ["bond_type"]
    )

    # ML metrics
    ml_predictions_total = Counter(
        "bondtrader_ml_predictions_total",
        "Total number of ML predictions",
        ["model_type", "model_version"],
    )

    ml_prediction_duration = Histogram(
        "bondtrader_ml_prediction_duration_seconds",
        "ML prediction duration in seconds",
        ["model_type"],
    )

    ml_training_duration = Histogram(
        "bondtrader_ml_training_duration_seconds",
        "ML model training duration in seconds",
        ["model_type"],
    )

    # Risk metrics
    risk_calculations_total = Counter("bondtrader_risk_calculations_total", "Total number of risk calculations", ["risk_type"])

    # System metrics
    active_connections = Gauge("bondtrader_active_connections", "Number of active connections")

    cache_hits = Counter("bondtrader_cache_hits_total", "Total number of cache hits", ["cache_type"])

    cache_misses = Counter("bondtrader_cache_misses_total", "Total number of cache misses", ["cache_type"])
else:
    # Dummy metrics when prometheus_client is not available
    api_requests_total = Counter()
    api_request_duration = Histogram()
    bond_valuations_total = Counter()
    valuation_duration = Histogram()
    ml_predictions_total = Counter()
    ml_prediction_duration = Histogram()
    ml_training_duration = Histogram()
    risk_calculations_total = Counter()
    active_connections = Gauge()
    cache_hits = Counter()
    cache_misses = Counter()


def start_metrics_server(port: int = 8001):
    """
    Start Prometheus metrics server

    Args:
        port: Port to expose metrics on
    """
    if not PROMETHEUS_AVAILABLE:
        return

    try:
        start_http_server(port)
        print(f"Prometheus metrics server started on port {port}")
    except Exception as e:
        print(f"Failed to start metrics server: {e}")


def track_api_request(method: str, endpoint: str):
    """
    Decorator to track API requests

    Usage:
        @track_api_request("GET", "/api/bonds")
        def get_bonds():
            ...
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "200"

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = "500"
                raise
            finally:
                duration = time.time() - start_time
                api_request_duration.labels(method=method, endpoint=endpoint).observe(duration)
                api_requests_total.labels(method=method, endpoint=endpoint, status=status).inc()

        return wrapper

    return decorator


def track_valuation(bond_type: str):
    """
    Decorator to track bond valuations

    Usage:
        @track_valuation("CORPORATE")
        def calculate_value(bond):
            ...
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                bond_valuations_total.labels(bond_type=bond_type).inc()
                valuation_duration.labels(bond_type=bond_type).observe(duration)

        return wrapper

    return decorator


def track_ml_prediction(model_type: str, model_version: str = "unknown"):
    """
    Decorator to track ML predictions

    Usage:
        @track_ml_prediction("random_forest", "v1.0")
        def predict(bond):
            ...
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                ml_predictions_total.labels(model_type=model_type, model_version=model_version).inc()
                ml_prediction_duration.labels(model_type=model_type).observe(duration)

        return wrapper

    return decorator


def track_risk_calculation(risk_type: str):
    """
    Track risk calculation

    Args:
        risk_type: Type of risk calculation (e.g., 'var', 'credit', 'liquidity')
    """
    risk_calculations_total.labels(risk_type=risk_type).inc()


def record_cache_hit(cache_type: str):
    """Record a cache hit"""
    cache_hits.labels(cache_type=cache_type).inc()


def record_cache_miss(cache_type: str):
    """Record a cache miss"""
    cache_misses.labels(cache_type=cache_type).inc()


def increment_connections():
    """Increment active connections counter"""
    active_connections.inc()


def decrement_connections():
    """Decrement active connections counter"""
    active_connections.dec()


def get_metrics() -> str:
    """
    Get Prometheus metrics in text format

    Returns:
        Metrics in Prometheus text format
    """
    if PROMETHEUS_AVAILABLE:
        return generate_latest().decode("utf-8")
    return "# Prometheus client not available\n"


# Auto-start metrics server if enabled
if os.getenv("ENABLE_METRICS", "false").lower() == "true":
    metrics_port = int(os.getenv("METRICS_PORT", "8001"))
    start_metrics_server(metrics_port)
