"""
System Routes
Health check and root endpoints with comprehensive monitoring

CRITICAL: Enhanced health checks for production monitoring
"""

from datetime import datetime

from fastapi import APIRouter

from bondtrader.utils.health import get_health_checker
from bondtrader.utils.pool_monitoring import get_pool_monitor

router = APIRouter(tags=["System"])


@router.get("/health")
async def health_check():
    """
    Comprehensive health check endpoint.

    CRITICAL: Checks database, Redis, and external APIs.
    Returns detailed health status for production monitoring.

    Returns:
        - **status**: Overall health status (healthy/degraded/unhealthy/critical)
        - **timestamp**: Current server timestamp
        - **components**: Health status of individual components
    """
    health_checker = get_health_checker()
    health_status = health_checker.get_health_status()
    return health_status


@router.get("/health/simple")
async def simple_health_check():
    """
    Simple health check endpoint (for load balancers).

    Returns basic status without detailed component checks.
    """
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@router.get("/health/detailed")
async def detailed_health_check():
    """
    Detailed health check with pool monitoring.

    Returns comprehensive health status including connection pool statistics.
    """
    health_checker = get_health_checker()
    health_status = health_checker.get_health_status()

    # Add pool monitoring
    try:
        from bondtrader.core.container import get_container

        container = get_container()
        database = container.get_database()

        pool_monitor = get_pool_monitor()
        pool_health = pool_monitor.check_pool_health(database)
        health_status["connection_pool"] = pool_health
    except Exception as e:
        health_status["connection_pool"] = {"error": str(e)}

    return health_status


@router.get("/")
async def root():
    """
    API root endpoint

    Returns basic API information and available endpoints.
    This is a good starting point for exploring the API.
    """
    return {
        "name": "BondTrader API",
        "version": "1.0.0",
        "description": "RESTful API for bond trading, valuation, and arbitrage detection",
        "endpoints": {
            "health": "/health",
            "bonds": "/bonds",
            "valuation": "/bonds/{bond_id}/valuation",
            "arbitrage": "/arbitrage/opportunities",
            "ml": "/ml/predict/{bond_id}",
            "risk": "/risk/{bond_id}",
        },
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc",
            "openapi_json": "/openapi.json",
        },
    }
