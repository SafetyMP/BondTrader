"""
API Request/Response Models
Pydantic models for API endpoints
"""

from typing import Optional

from pydantic import BaseModel, Field, validator


class BondCreate(BaseModel):
    """
    Bond creation request model with comprehensive validation.

    CRITICAL: All inputs are validated to prevent invalid financial data.
    """

    bond_id: str = Field(
        ...,
        description="Unique bond identifier",
        example="BOND-001",
        min_length=1,
        max_length=100,
        regex="^[A-Z0-9-_]+$",  # Alphanumeric, dashes, underscores only
    )
    bond_type: str = Field(
        ...,
        description="Type of bond",
        example="CORPORATE",
        enum=[
            "ZERO_COUPON",
            "FIXED_RATE",
            "TREASURY",
            "CORPORATE",
            "MUNICIPAL",
            "HIGH_YIELD",
            "FLOATING_RATE",
        ],
    )
    face_value: float = Field(
        ...,
        gt=0,
        le=1e12,  # Maximum $1 trillion (reasonable upper bound)
        description="Face value (par value) of the bond",
        example=1000.0,
    )
    coupon_rate: float = Field(
        ...,
        ge=0,
        le=1,  # 0-100% as decimal
        description="Annual coupon rate as decimal (e.g., 0.05 for 5%)",
        example=0.05,
    )
    maturity_date: str = Field(
        ...,
        description="Maturity date in ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)",
        example="2029-12-31",
    )
    issue_date: str = Field(
        ...,
        description="Issue date in ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)",
        example="2024-01-01",
    )
    current_price: float = Field(
        ...,
        gt=0,
        le=1e12,  # Maximum $1 trillion
        description="Current market price of the bond",
        example=950.0,
    )
    credit_rating: Optional[str] = Field(
        None,
        description="Credit rating (e.g., AAA, AA, BBB)",
        example="BBB",
        regex="^[A-Z]{1,3}[+-]?$",  # Valid credit rating format
    )
    issuer: Optional[str] = Field(
        None,
        description="Bond issuer name",
        example="Example Corp",
        max_length=200,  # Prevent extremely long strings
    )
    frequency: int = Field(
        default=2,
        ge=1,
        le=12,  # 1-12 payments per year (monthly maximum)
        description="Coupon payment frequency per year",
        example=2,
    )
    callable: bool = Field(default=False, description="Whether the bond is callable", example=False)
    convertible: bool = Field(default=False, description="Whether the bond is convertible", example=False)

    @validator("maturity_date")
    def validate_maturity_date(cls, v):
        """Validate maturity date is in the future"""
        from datetime import datetime

        try:
            maturity = datetime.fromisoformat(v.replace("Z", "+00:00"))
            if maturity <= datetime.now():
                raise ValueError("Maturity date must be in the future")
            return v
        except (ValueError, AttributeError) as e:
            raise ValueError(f"Invalid maturity_date format: {e}")

    @validator("issue_date")
    def validate_issue_date(cls, v, values):
        """Validate issue date is before maturity date"""
        from datetime import datetime

        try:
            issue = datetime.fromisoformat(v.replace("Z", "+00:00"))
            if "maturity_date" in values:
                maturity = datetime.fromisoformat(values["maturity_date"].replace("Z", "+00:00"))
                if issue >= maturity:
                    raise ValueError("Issue date must be before maturity date")
            return v
        except (ValueError, AttributeError) as e:
            raise ValueError(f"Invalid issue_date format: {e}")

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
        ...,
        description="Credit risk metrics",
        example={"default_probability": 0.02, "expected_loss": 5.0},
    )

    class Config:
        schema_extra = {
            "example": {
                "bond_id": "BOND-001",
                "var_historical": 45.2,
                "var_parametric": 42.8,
                "var_monte_carlo": 46.1,
                "credit_risk": {
                    "default_probability": 0.02,
                    "expected_loss": 5.0,
                    "credit_spread": 0.015,
                },
            }
        }
