"""
FastAPI REST API Server for BondTrader
Provides RESTful API endpoints for bond valuation, arbitrage detection, and ML predictions
"""

import os
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException, Path, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from bondtrader.config import get_config
from bondtrader.core.arbitrage_detector import ArbitrageDetector
from bondtrader.core.bond_models import Bond, BondType
from bondtrader.core.bond_valuation import BondValuator
from bondtrader.data.data_persistence_enhanced import EnhancedBondDatabase
from bondtrader.ml.ml_adjuster_enhanced import EnhancedMLBondAdjuster
from bondtrader.risk.risk_management import RiskManager
from bondtrader.utils.rate_limiter import RateLimiter, get_api_rate_limiter

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

# CORS middleware - configure allowed origins from environment
allowed_origins = os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:8000,http://localhost:8501").split(",")
if os.getenv("CORS_ALLOW_ALL", "false").lower() == "true":
    # Only allow wildcard in development with explicit flag
    allowed_origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Initialize core services
config = get_config()
valuator = BondValuator(risk_free_rate=config.default_risk_free_rate)
arbitrage_detector = ArbitrageDetector(valuator=valuator)
risk_manager = RiskManager(valuator=valuator)

# Database path from environment or config
db_path = os.getenv("BOND_DB_PATH", os.path.join(config.data_dir, "bonds.db"))
db = EnhancedBondDatabase(db_path=db_path)

# Rate limiter
rate_limiter = get_api_rate_limiter()

# API key authentication (optional)
security = HTTPBearer(auto_error=False)
API_KEY = os.getenv("API_KEY", None)
REQUIRE_API_KEY = os.getenv("REQUIRE_API_KEY", "false").lower() == "true"


def verify_api_key(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> bool:
    """Verify API key if authentication is enabled"""
    if not REQUIRE_API_KEY:
        return True
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API key authentication required but not configured")
    if not credentials:
        raise HTTPException(status_code=401, detail="API key required")
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return True


def get_client_ip(request: Request) -> str:
    """Extract client IP address from request"""
    if request.client:
        return request.client.host
    return "unknown"


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware"""
    client_ip = get_client_ip(request)
    allowed, error = rate_limiter.is_allowed(client_ip)
    if not allowed:
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=429,
            content={"detail": error},
            headers={"X-RateLimit-Limit": str(rate_limiter.max_requests), "X-RateLimit-Remaining": "0"},
        )
    response = await call_next(request)
    # Add rate limit headers
    remaining = rate_limiter.get_remaining(client_ip)
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    response.headers["X-RateLimit-Limit"] = str(rate_limiter.max_requests)
    return response


# ML model (lazy loaded)
ml_adjuster: Optional[EnhancedMLBondAdjuster] = None


# Pydantic models for request/response
class BondCreate(BaseModel):
    """Bond creation request model"""

    bond_id: str = Field(..., description="Unique bond identifier", example="BOND-001")
    bond_type: str = Field(
        ...,
        description="Type of bond",
        example="CORPORATE",
        enum=["ZERO_COUPON", "FIXED_RATE", "TREASURY", "CORPORATE", "MUNICIPAL", "HIGH_YIELD", "FLOATING_RATE"],
    )
    face_value: float = Field(..., gt=0, description="Face value (par value) of the bond", example=1000.0)
    coupon_rate: float = Field(..., ge=0, le=1, description="Annual coupon rate as decimal (e.g., 0.05 for 5%)", example=0.05)
    maturity_date: str = Field(
        ..., description="Maturity date in ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)", example="2029-12-31"
    )
    issue_date: str = Field(
        ..., description="Issue date in ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)", example="2024-01-01"
    )
    current_price: float = Field(..., gt=0, description="Current market price of the bond", example=950.0)
    credit_rating: Optional[str] = Field(None, description="Credit rating (e.g., AAA, AA, BBB)", example="BBB")
    issuer: Optional[str] = Field(None, description="Bond issuer name", example="Example Corp")
    frequency: int = Field(default=2, ge=1, le=12, description="Coupon payment frequency per year", example=2)
    callable: bool = Field(default=False, description="Whether the bond is callable", example=False)
    convertible: bool = Field(default=False, description="Whether the bond is convertible", example=False)

    class Config:
        schema_extra = {
            "example": {
                "bond_id": "BOND-001",
                "bond_type": "CORPORATE",
                "face_value": 1000.0,
                "coupon_rate": 0.05,
                "maturity_date": "2029-12-31",
                "issue_date": "2024-01-01",
                "current_price": 950.0,
                "credit_rating": "BBB",
                "issuer": "Example Corp",
                "frequency": 2,
                "callable": False,
                "convertible": False,
            }
        }


class BondResponse(BaseModel):
    """Bond response model"""

    bond_id: str = Field(..., description="Unique bond identifier", example="BOND-001")
    bond_type: str = Field(..., description="Type of bond", example="CORPORATE")
    face_value: float = Field(..., description="Face value of the bond", example=1000.0)
    coupon_rate: float = Field(..., description="Annual coupon rate as decimal", example=0.05)
    current_price: float = Field(..., description="Current market price", example=950.0)
    credit_rating: Optional[str] = Field(None, description="Credit rating", example="BBB")
    issuer: Optional[str] = Field(None, description="Bond issuer", example="Example Corp")

    class Config:
        schema_extra = {
            "example": {
                "bond_id": "BOND-001",
                "bond_type": "CORPORATE",
                "face_value": 1000.0,
                "coupon_rate": 0.05,
                "current_price": 950.0,
                "credit_rating": "BBB",
                "issuer": "Example Corp",
            }
        }


class ValuationResponse(BaseModel):
    """Bond valuation metrics response"""

    bond_id: str = Field(..., description="Bond identifier", example="BOND-001")
    fair_value: float = Field(..., description="Calculated fair value using DCF", example=975.50)
    yield_to_maturity: float = Field(..., description="Yield to maturity as decimal", example=0.0523)
    duration: float = Field(..., description="Macaulay duration in years", example=4.5)
    convexity: float = Field(..., description="Convexity measure", example=22.3)
    market_price: float = Field(..., description="Current market price", example=950.0)
    mismatch_percentage: float = Field(..., description="Percentage difference between market and fair value", example=-2.61)

    class Config:
        schema_extra = {
            "example": {
                "bond_id": "BOND-001",
                "fair_value": 975.50,
                "yield_to_maturity": 0.0523,
                "duration": 4.5,
                "convexity": 22.3,
                "market_price": 950.0,
                "mismatch_percentage": -2.61,
            }
        }


class ArbitrageOpportunity(BaseModel):
    """Arbitrage opportunity response"""

    bond_id: str = Field(..., description="Bond identifier", example="BOND-001")
    market_price: float = Field(..., description="Current market price", example=950.0)
    fair_value: float = Field(..., description="Calculated fair value", example=975.50)
    profit: float = Field(..., description="Potential profit per bond", example=25.50)
    profit_percentage: float = Field(..., description="Profit as percentage", example=2.68)
    recommendation: str = Field(..., description="Trading recommendation", example="BUY")
    arbitrage_type: str = Field(..., description="Type of arbitrage opportunity", example="UNDERVALUED")

    class Config:
        schema_extra = {
            "example": {
                "bond_id": "BOND-001",
                "market_price": 950.0,
                "fair_value": 975.50,
                "profit": 25.50,
                "profit_percentage": 2.68,
                "recommendation": "BUY",
                "arbitrage_type": "UNDERVALUED",
            }
        }


class MLPredictionResponse(BaseModel):
    """ML-enhanced bond prediction response"""

    bond_id: str = Field(..., description="Bond identifier", example="BOND-001")
    theoretical_fair_value: float = Field(..., description="Theoretical fair value from DCF", example=975.50)
    ml_adjusted_fair_value: float = Field(..., description="ML-adjusted fair value", example=980.25)
    adjustment_factor: float = Field(..., description="ML adjustment factor", example=1.0049)
    ml_confidence: float = Field(..., description="ML model confidence score (0-1)", example=0.85)

    class Config:
        schema_extra = {
            "example": {
                "bond_id": "BOND-001",
                "theoretical_fair_value": 975.50,
                "ml_adjusted_fair_value": 980.25,
                "adjustment_factor": 1.0049,
                "ml_confidence": 0.85,
            }
        }


class RiskMetricsResponse(BaseModel):
    """Risk metrics response"""

    bond_id: str = Field(..., description="Bond identifier", example="BOND-001")
    var_historical: Optional[float] = Field(None, description="Value at Risk (Historical method)", example=45.2)
    var_parametric: Optional[float] = Field(None, description="Value at Risk (Parametric method)", example=42.8)
    var_monte_carlo: Optional[float] = Field(None, description="Value at Risk (Monte Carlo method)", example=46.1)
    credit_risk: dict = Field(
        ..., description="Credit risk metrics", example={"default_probability": 0.02, "expected_loss": 5.0}
    )

    class Config:
        schema_extra = {
            "example": {
                "bond_id": "BOND-001",
                "var_historical": 45.2,
                "var_parametric": 42.8,
                "var_monte_carlo": 46.1,
                "credit_risk": {"default_probability": 0.02, "expected_loss": 5.0, "credit_spread": 0.015},
            }
        }


@app.get("/health", tags=["System"])
async def health_check():
    """
    Health check endpoint

    Returns the current health status of the API server.
    Use this endpoint to monitor API availability and uptime.

    Returns:
        - **status**: Health status (always "healthy" if endpoint is reachable)
        - **timestamp**: Current server timestamp in ISO format
    """
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/", tags=["System"])
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


@app.post("/bonds", response_model=BondResponse, status_code=201, tags=["Bonds"], dependencies=[Depends(verify_api_key)])
async def create_bond(bond: BondCreate):
    """
    Create a new bond

    Creates a new bond record in the system. The bond will be stored in the database
    and can be used for valuation, arbitrage detection, and risk analysis.

    **Request Body:**
    - All fields are required except `credit_rating`, `issuer`, `callable`, and `convertible`
    - `maturity_date` and `issue_date` should be in ISO format (YYYY-MM-DD)
    - `coupon_rate` should be a decimal (e.g., 0.05 for 5%)

    **Returns:**
    - Created bond object with assigned ID

    **Errors:**
    - `400 Bad Request`: Invalid bond data or validation error
    - `500 Internal Server Error`: Database or processing error
    """
    try:
        # Validate and convert string dates to datetime
        try:
            maturity = datetime.fromisoformat(bond.maturity_date.replace("Z", "+00:00"))
        except (ValueError, AttributeError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid maturity_date format: {e}")

        try:
            issue = datetime.fromisoformat(bond.issue_date.replace("Z", "+00:00"))
        except (ValueError, AttributeError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid issue_date format: {e}")

        # Validate dates are logical
        if issue >= maturity:
            raise HTTPException(status_code=400, detail="Issue date must be before maturity date")

        # Create Bond object
        bond_obj = Bond(
            bond_id=bond.bond_id,
            bond_type=BondType[bond.bond_type.upper()],
            face_value=bond.face_value,
            coupon_rate=bond.coupon_rate,
            maturity_date=maturity,
            issue_date=issue,
            current_price=bond.current_price,
            credit_rating=bond.credit_rating or "BBB",
            issuer=bond.issuer or "Unknown",
            frequency=bond.frequency,
            callable=bond.callable,
            convertible=bond.convertible,
        )

        # Save to database
        db.save_bond(bond_obj)

        return BondResponse(
            bond_id=bond_obj.bond_id,
            bond_type=bond_obj.bond_type.name,
            face_value=bond_obj.face_value,
            coupon_rate=bond_obj.coupon_rate,
            current_price=bond_obj.current_price,
            credit_rating=bond_obj.credit_rating,
            issuer=bond_obj.issuer,
        )
    except HTTPException:
        raise
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid bond data: {str(e)}")
    except Exception as e:
        # Log unexpected errors but don't expose internal details
        import logging

        logger = logging.getLogger(__name__)
        logger.error(f"Unexpected error creating bond: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/bonds", response_model=List[BondResponse], tags=["Bonds"], dependencies=[Depends(verify_api_key)])
async def list_bonds(
    skip: int = Query(0, ge=0, description="Number of records to skip (for pagination)", example=0),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return", example=100),
    issuer: Optional[str] = Query(None, description="Filter by issuer name (case-insensitive partial match)", example="Corp"),
    bond_type: Optional[str] = Query(None, description="Filter by bond type", example="CORPORATE"),
):
    """
    List bonds with optional filtering

    Retrieves a paginated list of bonds with optional filtering by issuer and bond type.

    **Query Parameters:**
    - `skip`: Number of records to skip (default: 0)
    - `limit`: Maximum number of records to return (default: 100, max: 1000)
    - `issuer`: Filter by issuer name (partial match, case-insensitive)
    - `bond_type`: Filter by bond type (exact match)

    **Returns:**
    - List of bond objects matching the criteria

    **Example:**
    ```
    GET /bonds?skip=0&limit=50&bond_type=CORPORATE&issuer=Example
    ```
    """
    try:
        bonds = db.load_all_bonds()

        # Apply filters
        if issuer:
            bonds = [b for b in bonds if b.issuer and issuer.lower() in b.issuer.lower()]
        if bond_type:
            bonds = [b for b in bonds if b.bond_type.name.lower() == bond_type.lower()]

        # Paginate
        bonds = bonds[skip : skip + limit]

        return [
            BondResponse(
                bond_id=b.bond_id,
                bond_type=b.bond_type.name,
                face_value=b.face_value,
                coupon_rate=b.coupon_rate,
                current_price=b.current_price,
                credit_rating=b.credit_rating,
                issuer=b.issuer,
            )
            for b in bonds
        ]
    except HTTPException:
        raise
    except Exception as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.error(f"Error listing bonds: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/bonds/{bond_id}", response_model=BondResponse, tags=["Bonds"], dependencies=[Depends(verify_api_key)])
async def get_bond(bond_id: str):
    """
    Get a specific bond by ID

    Retrieves detailed information about a specific bond.

    **Path Parameters:**
    - `bond_id`: Unique identifier of the bond

    **Returns:**
    - Bond object with all details

    **Errors:**
    - `404 Not Found`: Bond with the specified ID does not exist
    """
    bond = db.load_bond(bond_id)
    if not bond:
        raise HTTPException(status_code=404, detail=f"Bond {bond_id} not found")

    return BondResponse(
        bond_id=bond.bond_id,
        bond_type=bond.bond_type.name,
        face_value=bond.face_value,
        coupon_rate=bond.coupon_rate,
        current_price=bond.current_price,
        credit_rating=bond.credit_rating,
        issuer=bond.issuer,
    )


@app.get(
    "/bonds/{bond_id}/valuation", response_model=ValuationResponse, tags=["Valuation"], dependencies=[Depends(verify_api_key)]
)
async def get_valuation(bond_id: str):
    """
    Get valuation metrics for a bond

    Calculates comprehensive valuation metrics including:
    - Fair value (theoretical price)
    - Yield to Maturity (YTM)
    - Duration (price sensitivity)
    - Convexity (second-order price sensitivity)
    - Market price mismatch percentage

    **Path Parameters:**
    - `bond_id`: Unique identifier of the bond

    **Returns:**
    - Valuation metrics including fair value, YTM, duration, convexity, and mismatch

    **Errors:**
    - `404 Not Found`: Bond with the specified ID does not exist
    - `500 Internal Server Error`: Calculation error
    """
    bond = db.load_bond(bond_id)
    if not bond:
        raise HTTPException(status_code=404, detail=f"Bond {bond_id} not found")

    try:
        fair_value = valuator.calculate_fair_value(bond)
        ytm = valuator.calculate_yield_to_maturity(bond)
        duration = valuator.calculate_duration(bond, ytm)
        convexity = valuator.calculate_convexity(bond, ytm)

        mismatch = ((bond.current_price - fair_value) / fair_value) * 100

        return ValuationResponse(
            bond_id=bond.bond_id,
            fair_value=fair_value,
            yield_to_maturity=ytm,
            duration=duration,
            convexity=convexity,
            market_price=bond.current_price,
            mismatch_percentage=mismatch,
        )
    except HTTPException:
        raise
    except Exception as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.error(f"Error listing bonds: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get(
    "/arbitrage/opportunities",
    response_model=List[ArbitrageOpportunity],
    tags=["Arbitrage"],
    dependencies=[Depends(verify_api_key)],
)
async def get_arbitrage_opportunities(
    min_profit_percentage: float = Query(
        0.0, ge=0, description="Minimum profit percentage threshold (as decimal, e.g., 0.01 for 1%)", example=0.01
    ),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of opportunities to return", example=10),
):
    """
    Get arbitrage opportunities

    Identifies bonds that are mispriced relative to their theoretical fair value,
    presenting potential arbitrage opportunities.

    **Query Parameters:**
    - `min_profit_percentage`: Minimum profit percentage to include (default: 0.0)
    - `limit`: Maximum number of opportunities to return (default: 10, max: 100)

    **Returns:**
    - List of arbitrage opportunities sorted by profit percentage (descending)
    - Each opportunity includes bond ID, prices, profit, and recommendation

    **Example:**
    ```
    GET /arbitrage/opportunities?min_profit_percentage=0.02&limit=20
    ```
    """
    try:
        bonds = db.load_all_bonds()
        opportunities = arbitrage_detector.find_arbitrage_opportunities(bonds, min_profit_percentage=min_profit_percentage)

        # Sort by profit and limit
        opportunities.sort(key=lambda x: x.get("profit_percentage", 0), reverse=True)
        opportunities = opportunities[:limit]

        return [
            ArbitrageOpportunity(
                bond_id=opp["bond_id"],
                market_price=opp["market_price"],
                fair_value=opp["fair_value"],
                profit=opp["profit"],
                profit_percentage=opp["profit_percentage"],
                recommendation=opp["recommendation"],
                arbitrage_type=opp.get("arbitrage_type", "Unknown"),
            )
            for opp in opportunities
        ]
    except HTTPException:
        raise
    except Exception as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.error(f"Error listing bonds: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get(
    "/ml/predict/{bond_id}",
    response_model=MLPredictionResponse,
    tags=["Machine Learning"],
    dependencies=[Depends(verify_api_key)],
)
async def ml_predict(bond_id: str):
    """
    Get ML-adjusted prediction for a bond

    Uses machine learning models to adjust theoretical bond valuations,
    accounting for market inefficiencies and non-linear relationships.

    **Path Parameters:**
    - `bond_id`: Unique identifier of the bond

    **Returns:**
    - ML-adjusted fair value
    - Adjustment factor applied
    - Model confidence score

    **Errors:**
    - `404 Not Found`: Bond with the specified ID does not exist
    - `503 Service Unavailable`: ML model not available or not trained
    - `500 Internal Server Error`: Prediction error

    **Note:** ML models must be trained before use. See training scripts for details.
    """
    global ml_adjuster

    bond = db.load_bond(bond_id)
    if not bond:
        raise HTTPException(status_code=404, detail=f"Bond {bond_id} not found")

    # Lazy load ML model
    if ml_adjuster is None:
        try:
            ml_adjuster = EnhancedMLBondAdjuster(model_type=config.ml_model_type)
            # Try to load existing model
            import os

            model_path = os.path.join(config.model_dir, "enhanced_ml_model.joblib")
            if os.path.exists(model_path):
                ml_adjuster.load_model(model_path)
        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.error(f"ML model initialization error: {e}", exc_info=True)
            raise HTTPException(status_code=503, detail="ML model not available")

    try:
        result = ml_adjuster.predict_adjusted_value(bond)
        return MLPredictionResponse(
            bond_id=bond_id,
            theoretical_fair_value=result["theoretical_fair_value"],
            ml_adjusted_fair_value=result["ml_adjusted_fair_value"],
            adjustment_factor=result.get("adjustment_factor", 1.0),
            ml_confidence=result.get("ml_confidence", 0.0),
        )
    except HTTPException:
        raise
    except Exception as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.error(f"Error listing bonds: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get(
    "/risk/{bond_id}", response_model=RiskMetricsResponse, tags=["Risk Management"], dependencies=[Depends(verify_api_key)]
)
async def get_risk_metrics(bond_id: str):
    """
    Get risk metrics for a bond

    Calculates comprehensive risk metrics including:
    - Value at Risk (VaR) using multiple methods (Historical, Parametric, Monte Carlo)
    - Credit risk analysis including default probability and expected loss

    **Path Parameters:**
    - `bond_id`: Unique identifier of the bond

    **Returns:**
    - VaR values for different calculation methods
    - Credit risk metrics including default probability and recovery rates

    **Errors:**
    - `404 Not Found`: Bond with the specified ID does not exist
    - `500 Internal Server Error`: Risk calculation error

    **Note:** VaR calculations use a 95% confidence level and 1-day time horizon.
    """
    bond = db.load_bond(bond_id)
    if not bond:
        raise HTTPException(status_code=404, detail=f"Bond {bond_id} not found")

    try:
        # Calculate VaR (using single bond portfolio)
        var_historical = risk_manager.calculate_var([bond], [1.0], confidence_level=0.95, time_horizon=1, method="historical")
        var_parametric = risk_manager.calculate_var([bond], [1.0], confidence_level=0.95, time_horizon=1, method="parametric")
        var_monte_carlo = risk_manager.calculate_var(
            [bond], [1.0], confidence_level=0.95, time_horizon=1, method="monte_carlo"
        )

        # Credit risk
        credit_risk = risk_manager.calculate_credit_risk(bond)

        return RiskMetricsResponse(
            bond_id=bond_id,
            var_historical=var_historical.get("var_value"),
            var_parametric=var_parametric.get("var_value"),
            var_monte_carlo=var_monte_carlo.get("var_value"),
            credit_risk=credit_risk,
        )
    except HTTPException:
        raise
    except Exception as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.error(f"Error listing bonds: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
