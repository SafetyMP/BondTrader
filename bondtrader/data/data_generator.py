"""
Bond Data Generator
Generates synthetic bond data for testing and demonstration
"""

import random
from datetime import datetime, timedelta
from typing import List

import numpy as np

from bondtrader.core.bond_models import Bond, BondType


class BondDataGenerator:
    """Generates synthetic bond data with realistic characteristics"""

    def __init__(self, seed: int = None):
        """Initialize generator with optional seed"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def generate_bonds(self, num_bonds: int = 50) -> List[Bond]:
        """Generate a list of synthetic bonds"""
        bonds = []

        # Bond type probabilities
        type_probs = {
            BondType.ZERO_COUPON: 0.1,
            BondType.FIXED_RATE: 0.3,
            BondType.TREASURY: 0.2,
            BondType.CORPORATE: 0.3,
            BondType.HIGH_YIELD: 0.1,
        }

        # Credit ratings distribution
        ratings = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]
        rating_probs = [0.1, 0.15, 0.25, 0.25, 0.15, 0.08, 0.02]

        # Issuer names
        issuers = [
            "US Treasury",
            "Corporate Tech Inc",
            "Global Finance Corp",
            "Municipal Authority",
            "Industrial Holdings",
            "Energy Solutions Ltd",
            "Telecom Global",
            "Healthcare Systems",
            "Manufacturing Co",
            "Investment Bank PLC",
            "Retail Chain Corp",
            "Transportation Ltd",
        ]

        for i in range(num_bonds):
            # Select bond type
            bond_type = np.random.choice(list(type_probs.keys()), p=list(type_probs.values()))

            # Generate characteristics
            face_value = random.choice([1000, 5000, 10000, 25000])
            time_to_maturity = random.uniform(0.5, 30.0)  # 6 months to 30 years
            maturity_date = datetime.now() + timedelta(days=int(time_to_maturity * 365.25))
            issue_date = datetime.now() - timedelta(days=random.randint(30, 3650))  # Up to 10 years ago

            # Coupon rate based on bond type
            if bond_type == BondType.ZERO_COUPON:
                coupon_rate = 0.0
            elif bond_type == BondType.TREASURY:
                coupon_rate = random.uniform(1.5, 4.5)
            elif bond_type == BondType.HIGH_YIELD:
                coupon_rate = random.uniform(6.0, 12.0)
            else:
                coupon_rate = random.uniform(2.0, 8.0)

            # Select credit rating
            rating = np.random.choice(ratings, p=rating_probs)

            # Assign issuer based on bond type
            if bond_type == BondType.TREASURY:
                issuer = "US Treasury"
            elif bond_type == BondType.CORPORATE:
                issuer = random.choice([iss for iss in issuers if iss != "US Treasury"])
            else:
                issuer = random.choice(issuers)

            # Generate current market price (with some variance from fair value)
            # We'll add randomness later, but base it on face value
            base_price_ratio = random.uniform(0.85, 1.15)  # Price can be 85% to 115% of face
            current_price = face_value * base_price_ratio

            # Frequency (usually semi-annual)
            frequency = 2

            # Additional characteristics
            callable = random.random() < 0.2  # 20% are callable
            convertible = random.random() < 0.1  # 10% are convertible

            # Create bond ID
            bond_id = f"BOND-{i+1:04d}-{bond_type.value[:3].upper()}-{rating}"

            try:
                bond = Bond(
                    bond_id=bond_id,
                    bond_type=bond_type,
                    face_value=face_value,
                    coupon_rate=coupon_rate,
                    maturity_date=maturity_date,
                    issue_date=issue_date,
                    current_price=current_price,
                    credit_rating=rating,
                    issuer=issuer,
                    frequency=frequency,
                    callable=callable,
                    convertible=convertible,
                )
                bonds.append(bond)
            except ValueError as e:
                # Skip invalid bonds
                continue

        return bonds

    def add_price_noise(self, bonds: List[Bond], noise_level: float = 0.05) -> List[Bond]:
        """
        Add random noise to bond prices to create more realistic market conditions

        Args:
            bonds: List of bonds
            noise_level: Standard deviation of noise as fraction of price (default 5%)

        Returns:
            List of bonds with adjusted prices
        """
        from bondtrader.core.bond_valuation import BondValuator

        valuator = BondValuator()

        for bond in bonds:
            # Calculate fair value
            fair_value = valuator.calculate_fair_value(bond)

            # Add noise
            noise = np.random.normal(0, noise_level * fair_value)
            bond.current_price = max(fair_value * 0.5, fair_value + noise)  # Ensure positive

        return bonds
