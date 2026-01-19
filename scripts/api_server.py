"""
FastAPI REST API Server for BondTrader
Provides RESTful API endpoints for bond valuation, arbitrage detection, and ML predictions

Refactored into modular structure with separate routes, models, and middleware
"""

import os

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from bondtrader.config import get_config
from bondtrader.core.container import get_container
from bondtrader.utils.error_handling import PRODUCTION_MODE, sanitize_error_message
from scripts.api.middleware import rate_limit_middleware, setup_cors, verify_api_key
from scripts.api.routes import arbitrage, bonds, metrics, ml, risk, system, valuation

# Initialize FastAPI app
app = FastAPI(
    title="BondTrader API",
    description="""
    ## BondTrader RESTful API

    Comprehensive API for bond trading, valuation, arbitrage detection, and risk management.

    ### Features

    * **Bond Management**: Create, retrieve, and list bonds
    * **Valuation**: Calculate fair value, YTM, duration, and convexity
    * **Arbitrage Detection**: Identify mispriced bonds and profit opportunities
    * **Machine Learning**: ML-enhanced bond price predictions
    * **Risk Management**: VaR calculations and credit risk analysis

    ### Authentication

    API keys may be required for certain endpoints. Contact your administrator for access.

    ### Rate Limiting

    API requests are rate-limited to ensure fair usage. Check response headers for rate limit information.
    """,
    version="1.0.0",
    contact={
        "name": "BondTrader Support",
        "email": "support@bondtrader.local",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
    servers=[
        {"url": "http://localhost:8000", "description": "Development server"},
        {"url": "https://api.bondtrader.local", "description": "Production server"},
    ],
    tags_metadata=[
        {
            "name": "Health",
            "description": "Health check and system status endpoints",
        },
        {
            "name": "Bonds",
            "description": "Bond CRUD operations and management",
        },
        {
            "name": "Valuation",
            "description": "Bond valuation and financial metrics calculations",
        },
        {
            "name": "Arbitrage",
            "description": "Arbitrage opportunity detection and analysis",
        },
        {
            "name": "Machine Learning",
            "description": "ML-enhanced predictions and model management",
        },
        {
            "name": "Risk",
            "description": "Risk metrics including VaR and credit risk",
        },
    ],
)

# Setup middleware
setup_cors(app)
app.middleware("http")(rate_limit_middleware)


# CRITICAL: Global exception handler for error sanitization
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler with error sanitization.

    CRITICAL: In production, sanitizes error messages to prevent information leakage.
    """
    import os

    production = os.getenv("PRODUCTION_MODE", "false").lower() == "true"

    # Log full error internally
    from bondtrader.utils.utils import logger

    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    # Sanitize error message
    sanitized_message = sanitize_error_message(exc, production=production)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": sanitized_message},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Validation error handler.

    Validation errors are user-facing and don't need sanitization.
    """
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors()},
    )


# Initialize service container (singleton)
container = get_container()
config = container.config

# Database path from environment or config
db_path = os.getenv("BOND_DB_PATH", os.path.join(config.data_dir, "bonds.db"))
container.get_database(db_path=db_path)

# Register routes
app.include_router(system.router)
app.include_router(bonds.router, prefix="/bonds")
app.include_router(valuation.router, prefix="")
app.include_router(arbitrage.router, prefix="/arbitrage")
app.include_router(ml.router, prefix="/ml")
app.include_router(risk.router, prefix="/risk")
app.include_router(metrics.router, prefix="/metrics")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
