"""
Generate Sample Dataset for Streamlit Dashboard
Generates synthetic bond data and saves it as CSV for dashboard visualization

This is a standalone script that generates sample bond data without requiring
the full bondtrader module dependencies.
"""

import random
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


class BondType(Enum):
    """Enumeration of different bond types"""

    ZERO_COUPON = "Zero Coupon"
    FIXED_RATE = "Fixed Rate"
    FLOATING_RATE = "Floating Rate"
    TREASURY = "Treasury"
    CORPORATE = "Corporate"
    MUNICIPAL = "Municipal"
    HIGH_YIELD = "High Yield"


class BondDataGenerator:
    """Generates synthetic bond data with realistic characteristics"""

    def __init__(self, seed: Optional[int] = None) -> None:
        """
        Initialize generator with optional seed

        Args:
            seed: Random seed for reproducibility (optional)
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def generate_bonds_dict(self, num_bonds: int = 50) -> list[dict]:
        """Generate a list of synthetic bonds as dictionaries"""
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

            # Generate current market price (with some variance from face value)
            base_price_ratio = random.uniform(0.85, 1.15)  # Price can be 85% to 115% of face
            current_price = face_value * base_price_ratio

            # Frequency (usually semi-annual)
            frequency = 2

            # Additional characteristics
            callable = random.random() < 0.2  # 20% are callable
            convertible = random.random() < 0.1  # 10% are convertible

            # Create bond ID
            bond_id = f"BOND-{i+1:04d}-{bond_type.value[:3].upper().replace(' ', '')}-{rating}"

            # Calculate time to maturity in years
            delta = maturity_date - datetime.now()
            time_to_maturity_years = max(0, delta.days / 365.25)

            # Calculate years since issue
            delta_issue = datetime.now() - issue_date
            years_since_issue = delta_issue.days / 365.25

            bond_dict = {
                "bond_id": bond_id,
                "bond_type": bond_type.value,
                "issuer": issuer,
                "credit_rating": rating,
                "face_value": face_value,
                "coupon_rate": coupon_rate,
                "maturity_date": maturity_date.strftime("%Y-%m-%d"),
                "issue_date": issue_date.strftime("%Y-%m-%d"),
                "current_price": round(current_price, 2),
                "time_to_maturity": round(time_to_maturity_years, 4),
                "years_since_issue": round(years_since_issue, 4),
                "frequency": frequency,
                "callable": callable,
                "convertible": convertible,
            }

            # Validate bond data
            if bond_dict["current_price"] > 0 and bond_dict["face_value"] > 0:
                bonds.append(bond_dict)

        return bonds


def generate_sample_dataset(
    num_bonds: int = 200,
    output_dir: str = "sample_data",
    filename: str = "sample_bonds.csv",
    seed: int = 42,
) -> str:
    """
    Generate sample bond dataset and save to CSV

    Args:
        num_bonds: Number of bonds to generate (default: 200)
        output_dir: Output directory for the dataset (default: "sample_data")
        filename: Output filename (default: "sample_bonds.csv")
        seed: Random seed for reproducibility (default: 42)

    Returns:
        Path to the generated CSV file
    """
    print(f"Generating {num_bonds} sample bonds...")

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Generate bonds
    generator = BondDataGenerator(seed=seed)
    bonds = generator.generate_bonds_dict(num_bonds=num_bonds)

    print(f"Generated {len(bonds)} bonds")

    # Convert to DataFrame
    df = pd.DataFrame(bonds)

    # Save to CSV
    output_file = output_path / filename
    df.to_csv(output_file, index=False)

    print(f"\nSample dataset saved to: {output_file}")
    print(f"\nDataset summary:")
    print(f"  - Total bonds: {len(df)}")
    print(f"  - Bond types:")
    for bond_type, count in df["bond_type"].value_counts().items():
        print(f"      {bond_type}: {count}")
    print(f"  - Credit ratings:")
    for rating, count in df["credit_rating"].value_counts().items():
        print(f"      {rating}: {count}")
    print(f"  - Average price: ${df['current_price'].mean():,.2f}")
    print(f"  - Price range: ${df['current_price'].min():,.2f} - ${df['current_price'].max():,.2f}")
    print(f"  - Average time to maturity: {df['time_to_maturity'].mean():.2f} years")
    print(f"  - Average coupon rate: {df['coupon_rate'].mean():.2f}%")

    return str(output_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate sample bond dataset for Streamlit dashboard")
    parser.add_argument(
        "--num-bonds",
        type=int,
        default=200,
        help="Number of bonds to generate (default: 200)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="sample_data",
        help="Output directory (default: sample_data)",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="sample_bonds.csv",
        help="Output filename (default: sample_bonds.csv)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    output_file = generate_sample_dataset(
        num_bonds=args.num_bonds,
        output_dir=args.output_dir,
        filename=args.filename,
        seed=args.seed,
    )

    print(f"\n✓ Success! Dataset saved to: {output_file}")
