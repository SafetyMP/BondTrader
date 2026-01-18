"""
Pydantic-Enhanced Bond Models (OPTIONAL)
Optional Pydantic validation for bond data models

NOTE: This module is optional and currently not used in the codebase.
The standard bond_models.py module (using dataclasses) is the default.
This module is provided for users who want Pydantic validation.
To use it, you would need to manually import and use BondPydantic instead of Bond.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

# Optional Pydantic for validation
try:
    from pydantic import BaseModel, Field, field_validator, model_validator
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    # Fallback to simple BaseModel
    class BaseModel:
        pass


from bondtrader.core.bond_models import Bond, BondType


if HAS_PYDANTIC:

    class BondPydantic(BaseModel):
        """Pydantic-validated Bond model"""

        bond_id: str = Field(..., description="Unique bond identifier", min_length=1)
        bond_type: BondType
        face_value: float = Field(..., gt=0, description="Face value (must be positive)")
        coupon_rate: float = Field(..., ge=0, le=100, description="Annual coupon rate as percentage")
        maturity_date: datetime = Field(..., description="Maturity date")
        issue_date: datetime = Field(..., description="Issue date")
        current_price: float = Field(..., gt=0, description="Current price (must be positive)")
        credit_rating: str = Field(default="BBB", description="Credit rating")
        issuer: str = Field(default="", description="Issuer name")
        frequency: int = Field(default=2, ge=1, le=12, description="Coupon payments per year")
        callable: bool = Field(default=False, description="Callable bond flag")
        convertible: bool = Field(default=False, description="Convertible bond flag")

        @field_validator("maturity_date", "issue_date")
        @classmethod
        def validate_date(cls, v: datetime) -> datetime:
            """Validate date is datetime object"""
            if not isinstance(v, datetime):
                raise ValueError("Date must be datetime object")
            return v

        @model_validator(mode="after")
        def validate_dates(self) -> "BondPydantic":
            """Validate maturity date is after issue date"""
            if self.maturity_date <= self.issue_date:
                raise ValueError("Maturity date must be after issue date")
            return self

        @model_validator(mode="after")
        def validate_zero_coupon(self) -> "BondPydantic":
            """Validate zero coupon bonds have coupon_rate=0"""
            if self.bond_type == BondType.ZERO_COUPON and self.coupon_rate != 0:
                raise ValueError("Zero coupon bonds must have coupon_rate=0")
            return self

        def to_bond(self) -> Bond:
            """Convert to standard Bond dataclass"""
            return Bond(
                bond_id=self.bond_id,
                bond_type=self.bond_type,
                face_value=self.face_value,
                coupon_rate=self.coupon_rate,
                maturity_date=self.maturity_date,
                issue_date=self.issue_date,
                current_price=self.current_price,
                credit_rating=self.credit_rating,
                issuer=self.issuer,
                frequency=self.frequency,
                callable=self.callable,
                convertible=self.convertible,
            )

        @classmethod
        def from_bond(cls, bond: Bond) -> "BondPydantic":
            """Create from standard Bond dataclass"""
            return cls(
                bond_id=bond.bond_id,
                bond_type=bond.bond_type,
                face_value=bond.face_value,
                coupon_rate=bond.coupon_rate,
                maturity_date=bond.maturity_date,
                issue_date=bond.issue_date,
                current_price=bond.current_price,
                credit_rating=bond.credit_rating,
                issuer=bond.issuer,
                frequency=bond.frequency,
                callable=bond.callable,
                convertible=bond.convertible,
            )

else:
    # Fallback if Pydantic not available
    class BondPydantic:
        """Fallback BondPydantic (Pydantic not available)"""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Pydantic not installed. Install with: pip install pydantic. "
                "Or use the standard Bond class from bondtrader.core.bond_models"
            )
