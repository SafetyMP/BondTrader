"""
Business Metrics API Endpoints
Provides business KPIs and metrics for dashboards

CRITICAL: Required for business intelligence and monitoring
"""

from typing import Optional

from fastapi import APIRouter, Query

from bondtrader.core.observability import get_metrics
from bondtrader.utils.performance_monitoring import get_performance_monitor

router = APIRouter(tags=["Metrics"])


@router.get("/metrics/business")
async def get_business_metrics():
    """
    Get business metrics (trading volume, P&L, portfolio value).

    Returns:
        Dictionary with business KPIs
    """
    metrics = get_metrics()
    all_metrics = metrics.get_metrics()

    return {
        "trading_volume": all_metrics.get("business", {}).get("total_trading_volume", 0),
        "total_pnl": all_metrics.get("business", {}).get("total_pnl", 0),
        "portfolio_value": all_metrics.get("business", {}).get("current_portfolio_value"),
        "risk_metrics": all_metrics.get("business", {}).get("risk_metrics", {}),
    }


@router.get("/metrics/performance")
async def get_performance_metrics():
    """
    Get performance metrics and alerts.

    Returns:
        Dictionary with performance metrics and recent alerts
    """
    monitor = get_performance_monitor()
    summary = monitor.get_performance_summary()

    return summary


@router.get("/metrics/all")
async def get_all_metrics():
    """
    Get all metrics (business, technical, performance).

    Returns:
        Dictionary with all metrics
    """
    metrics = get_metrics()
    performance_monitor = get_performance_monitor()

    return {
        "business": metrics.get_metrics().get("business", {}),
        "technical": {
            "counters": metrics.get_metrics().get("counters", {}),
            "gauges": metrics.get_metrics().get("gauges", {}),
            "histograms": metrics.get_metrics().get("histograms", {}),
        },
        "performance": performance_monitor.get_performance_summary(),
    }
