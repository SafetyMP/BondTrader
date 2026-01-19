"""
Bond Models and Classification System
Supports various bond types with different characteristics
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


class BondType(Enum):
    """Enumeration of different bond types"""

    ZERO_COUPON = "Zero Coupon"
    FIXED_RATE = "Fixed Rate"
    FLOATING_RATE = "Floating Rate"
    TREASURY = "Treasury"
    CORPORATE = "Corporate"
    MUNICIPAL = "Municipal"
    HIGH_YIELD = "High Yield"


@dataclass
class Bond:
    """Base bond structure with all relevant attributes"""

    bond_id: str
    bond_type: BondType
    face_value: float
    coupon_rate: float  # Annual coupon rate as percentage
    maturity_date: datetime
    issue_date: datetime
    current_price: float
    credit_rating: str = "BBB"
    issuer: str = ""
    frequency: int = 2  # Coupon payments per year (semi-annual default)
    callable: bool = False
    convertible: bool = False

    def __post_init__(self):
        """Validate bond data"""
        if self.current_price <= 0:
            raise ValueError("Current price must be positive")
        if self.face_value <= 0:
            raise ValueError("Face value must be positive")

    @property
    def time_to_maturity(self) -> float:
        """Calculate time to maturity in years"""
        delta = self.maturity_date - datetime.now()
        return max(0, delta.days / 365.25)

    @property
    def years_since_issue(self) -> float:
        """Calculate years since issue date"""
        delta = datetime.now() - self.issue_date
        return delta.days / 365.25

    def get_bond_characteristics(self, current_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Extract characteristics for ML classification

        Args:
            current_time: Optional datetime to use for calculations (for consistency in batches)
                         If None, uses datetime.now()

        Returns:
            Dictionary of bond characteristics
        """
        # OPTIMIZED: Use provided current_time or calculate once
        # This avoids multiple datetime.now() calls when processing batches
        if current_time is None:
            current_time = datetime.now()

        # Calculate time-based metrics once
        time_to_maturity = max(0, (self.maturity_date - current_time).days / 365.25)
        years_since_issue = (current_time - self.issue_date).days / 365.25

        return {
            "coupon_rate": self.coupon_rate,
            "time_to_maturity": time_to_maturity,
            "credit_rating_numeric": self._rating_to_numeric(self.credit_rating),
            "current_price": self.current_price,
            "face_value": self.face_value,
            "years_since_issue": years_since_issue,
            "frequency": self.frequency,
            "callable": 1 if self.callable else 0,
            "convertible": 1 if self.convertible else 0,
        }

    @staticmethod
    def _rating_to_numeric(rating: str) -> int:
        """Convert credit rating to numeric scale"""
        rating_map = {
            "AAA": 20,
            "AA+": 19,
            "AA": 18,
            "AA-": 17,
            "A+": 16,
            "A": 15,
            "A-": 14,
            "BBB+": 13,
            "BBB": 12,
            "BBB-": 11,
            "BB+": 10,
            "BB": 9,
            "BB-": 8,
            "B+": 7,
            "B": 6,
            "B-": 5,
            "CCC+": 4,
            "CCC": 3,
            "CCC-": 2,
            "D": 1,
            "NR": 0,
        }
        return rating_map.get(rating.upper(), 12)  # Default to BBB


class BondClassifier:
    """Classifies bonds and extracts features for ML models"""

    def __init__(self):
        self.bond_type_map = {
            BondType.ZERO_COUPON: lambda b: b.coupon_rate == 0,
            BondType.FIXED_RATE: lambda b: b.coupon_rate > 0 and b.bond_type != BondType.FLOATING_RATE,
            BondType.TREASURY: lambda b: "treasury" in b.issuer.lower() or b.credit_rating in ["AAA", "AA+"],
            BondType.CORPORATE: lambda b: b.bond_type == BondType.CORPORATE or ("corp" in b.issuer.lower()),
            BondType.HIGH_YIELD: lambda b: b.credit_rating in ["BB", "BB+", "BB-", "B", "B+", "B-", "CCC"],
        }

    def classify(self, bond: Bond) -> BondType:
        """Classify bond based on characteristics"""
        # Check zero coupon first
        if bond.coupon_rate == 0:
            return BondType.ZERO_COUPON

        # Check more specific types first (TREASURY, HIGH_YIELD) before general FIXED_RATE
        # Check TREASURY
        if "treasury" in bond.issuer.lower() or bond.credit_rating in ["AAA", "AA+"]:
            return BondType.TREASURY

        # Check HIGH_YIELD
        if bond.credit_rating in ["BB", "BB+", "BB-", "B", "B+", "B-", "CCC"]:
            return BondType.HIGH_YIELD

        # Check CORPORATE
        if bond.bond_type == BondType.CORPORATE or ("corp" in bond.issuer.lower()):
            return BondType.CORPORATE

        # Check FIXED_RATE (general case)
        if bond.coupon_rate > 0 and bond.bond_type != BondType.FLOATING_RATE:
            return BondType.FIXED_RATE

        # Default classification
        return BondType.FIXED_RATE

    def extract_features(self, bonds: List[Bond]) -> np.ndarray:
        """Extract features from bonds for ML model"""
        features = []
        for bond in bonds:
            char = bond.get_bond_characteristics()
            feature_vector = [
                char["coupon_rate"],
                char["time_to_maturity"],
                char["credit_rating_numeric"],
                char["current_price"] / char["face_value"],  # Price to par ratio
                char["years_since_issue"],
                char["frequency"],
                char["callable"],
                char["convertible"],
            ]
            features.append(feature_vector)
        return np.array(features)
