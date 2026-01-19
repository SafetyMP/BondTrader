"""
Historical Bond Data Fetcher
Fetches historical bond data from FRED and FINRA APIs
and converts it to Bond objects for training and evaluation
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env file if it exists
try:
    from dotenv import load_dotenv

    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    # python-dotenv not installed, try to load .env manually
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()

from bondtrader.config import get_config
from bondtrader.core.bond_models import Bond, BondType
from bondtrader.data.market_data import FINRADataProvider, MarketDataManager
from bondtrader.utils.utils import logger


def fetch_historical_treasury_bonds(start_date: datetime, end_date: datetime, save_path: Optional[str] = None) -> List[Bond]:
    """
    Fetch historical Treasury bond data from FRED and convert to Bond objects

    Args:
        start_date: Start date for historical data (e.g., 1980-01-01)
        end_date: End date for historical data (e.g., 1999-12-31)
        save_path: Optional path to save the data as CSV

    Returns:
        List of Bond objects created from historical data
    """
    config = get_config()

    if not config.fred_api_key:
        logger.warning(
            "FRED_API_KEY not found. Generating synthetic Treasury data instead. "
            "To use real FRED data, set FRED_API_KEY in your .env file."
        )
        # Generate synthetic data instead
        return _generate_synthetic_treasury_bonds(start_date, end_date, save_path)

    print(f"Fetching historical Treasury data from {start_date.date()} to {end_date.date()}...")

    manager = MarketDataManager()

    # Fetch historical Treasury yields
    treasury_data = manager.fetch_historical_treasury_data(
        start_date=start_date, end_date=end_date, maturities=["GS1", "GS2", "GS5", "GS10", "GS30"]  # 1, 2, 5, 10, 30 year
    )

    if treasury_data is None or treasury_data.empty:
        print("No Treasury data retrieved from FRED")
        return []

    print(f"Retrieved {len(treasury_data)} data points")
    print(f"Date range: {treasury_data.index.min()} to {treasury_data.index.max()}")
    print(f"Available maturities: {list(treasury_data.columns)}")

    # Convert to Bond objects
    bonds = []

    # For each date, create synthetic bonds based on Treasury yields
    for date, row in treasury_data.iterrows():
        # Create bonds for each available maturity
        for maturity_col in treasury_data.columns:
            if pd.isna(row[maturity_col]):
                continue

            yield_rate = row[maturity_col]

            # Extract maturity in years from column name (e.g., 'GS10' -> 10)
            try:
                maturity_years = int(maturity_col.replace("GS", ""))
            except ValueError:
                continue

            # Create a synthetic Treasury bond
            # Use the yield as the coupon rate (simplified assumption)
            coupon_rate = yield_rate * 100  # Convert to percentage

            # Calculate dates
            issue_date = date - pd.Timedelta(days=365 * maturity_years)  # Assume issued maturity_years ago
            maturity_date = date + pd.Timedelta(days=365 * maturity_years)

            # Calculate bond price from yield
            # Simplified: price = face_value for par bonds, adjust based on yield
            face_value = 1000.0
            current_price = face_value  # Simplified - in reality would calculate from yield

            bond_id = f"TREASURY-{date.strftime('%Y%m%d')}-{maturity_years}YR"

            try:
                bond = Bond(
                    bond_id=bond_id,
                    bond_type=BondType.TREASURY,
                    face_value=face_value,
                    coupon_rate=coupon_rate,
                    maturity_date=maturity_date.to_pydatetime(),
                    issue_date=issue_date.to_pydatetime(),
                    current_price=current_price,
                    credit_rating="AAA",
                    issuer="US Treasury",
                    frequency=2,  # Semi-annual
                    callable=False,
                    convertible=False,
                )
                bonds.append(bond)
            except Exception as e:
                logger.warning(f"Error creating bond {bond_id}: {e}")
                continue

    print(f"Created {len(bonds)} Bond objects from historical data")

    # Save to CSV if requested
    if save_path:
        save_bonds_to_csv(bonds, save_path)

    return bonds


def save_bonds_to_csv(bonds: List[Bond], filepath: str):
    """Save bonds to CSV file"""
    data = []
    for bond in bonds:
        data.append(
            {
                "bond_id": bond.bond_id,
                "bond_type": bond.bond_type.value,
                "face_value": bond.face_value,
                "coupon_rate": bond.coupon_rate,
                "maturity_date": bond.maturity_date.isoformat(),
                "issue_date": bond.issue_date.isoformat(),
                "current_price": bond.current_price,
                "credit_rating": bond.credit_rating,
                "issuer": bond.issuer,
                "frequency": bond.frequency,
                "callable": bond.callable,
                "convertible": bond.convertible,
                "time_to_maturity": bond.time_to_maturity,
                "years_since_issue": bond.years_since_issue,
            }
        )

    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Saved {len(bonds)} bonds to {filepath}")


def _generate_synthetic_treasury_bonds(
    start_date: datetime, end_date: datetime, save_path: Optional[str] = None
) -> List[Bond]:
    """
    Generate synthetic Treasury bond data when API is not available
    """
    from datetime import timedelta

    import numpy as np

    print("Generating synthetic Treasury bond data...")

    bonds = []
    current_date = start_date

    # Generate monthly data points
    while current_date <= end_date:
        # Simulate Treasury yields (realistic ranges for 2010-2020)
        base_rate = 0.02 + np.random.uniform(0, 0.03)  # 2-5% range

        # Create bonds for different maturities
        for maturity_years in [1, 2, 5, 10, 30]:
            # Yield curve: longer maturities have higher yields
            yield_rate = base_rate + (maturity_years / 30) * 0.01
            coupon_rate = yield_rate * 100  # Convert to percentage

            # Calculate dates
            issue_date = current_date - timedelta(days=365 * maturity_years // 2)
            maturity_date = current_date + timedelta(days=365 * maturity_years)

            face_value = 1000.0
            current_price = face_value * (1 + np.random.uniform(-0.05, 0.05))  # ±5% variation

            bond_id = f"SYNTH-TREASURY-{current_date.strftime('%Y%m%d')}-{maturity_years}YR"

            try:
                bond = Bond(
                    bond_id=bond_id,
                    bond_type=BondType.TREASURY,
                    face_value=face_value,
                    coupon_rate=coupon_rate,
                    maturity_date=maturity_date,
                    issue_date=issue_date,
                    current_price=current_price,
                    credit_rating="AAA",
                    issuer="US Treasury",
                    frequency=2,
                    callable=False,
                    convertible=False,
                )
                bonds.append(bond)
            except Exception as e:
                logger.warning(f"Error creating synthetic bond {bond_id}: {e}")
                continue

        # Move to next month
        if current_date.month == 12:
            current_date = current_date.replace(year=current_date.year + 1, month=1)
        else:
            current_date = current_date.replace(month=current_date.month + 1)

    print(f"Generated {len(bonds)} synthetic Treasury bonds")

    if save_path:
        save_bonds_to_csv(bonds, save_path)

    return bonds


def fetch_finra_bonds(start_date: datetime, end_date: datetime, save_path: Optional[str] = None) -> List[Bond]:
    """
    Fetch historical bond data from FINRA TRACE and convert to Bond objects

    Args:
        start_date: Start date for historical data (must be 2002 or later)
        end_date: End date for historical data
        save_path: Optional path to save the data as CSV

    Returns:
        List of Bond objects created from FINRA data
    """
    config = get_config()

    if not config.finra_api_key or not config.finra_api_password:
        logger.warning(
            "FINRA API credentials not found. Generating synthetic corporate bond data instead. "
            "To use real FINRA data, set FINRA_API_KEY and FINRA_API_PASSWORD in your .env file."
        )
        # Generate synthetic corporate bond data instead
        return _generate_synthetic_corporate_bonds(start_date, end_date, save_path)

    if start_date.year < 2002:
        logger.warning(
            f"FINRA TRACE data is only available from 2002 onward. "
            f"Requested start date: {start_date.year}. Skipping FINRA fetch."
        )
        return []

    print(f"Fetching FINRA TRACE data from {start_date.date()} to {end_date.date()}...")

    finra = FINRADataProvider()

    # Fetch historical bond transaction data
    finra_data = finra.fetch_historical_bond_data(start_date=start_date, end_date=end_date)

    if finra_data is None or finra_data.empty:
        print("No FINRA data retrieved")
        return []

    print(f"Retrieved {len(finra_data)} FINRA records")
    print(f"Available columns: {list(finra_data.columns)}")

    # Convert to Bond objects
    # Note: FINRA data structure may vary - this is a generic conversion
    bonds = []

    # Common FINRA TRACE fields (adjust based on actual API response)
    # Typical fields: executionDate, cusip, price, yield, quantity, etc.
    for idx, row in finra_data.iterrows():
        try:
            # Extract bond information from FINRA data
            # Adjust field names based on actual FINRA API response structure
            bond_id = row.get("cusip", f"FINRA-{idx}")
            execution_date = pd.to_datetime(row.get("executionDate", row.get("date", start_date)))

            # Price and yield
            price = float(row.get("price", row.get("tradePrice", 1000.0)))
            yield_rate = float(row.get("yield", row.get("yieldToMaturity", 0.05)))

            # Maturity information (if available)
            maturity_date_str = row.get("maturityDate", None)
            if maturity_date_str:
                maturity_date = pd.to_datetime(maturity_date_str).to_pydatetime()
            else:
                # Estimate maturity if not provided (default to 10 years)
                maturity_date = execution_date + pd.Timedelta(days=365 * 10)
                maturity_date = maturity_date.to_pydatetime()

            # Issue date (estimate if not provided)
            issue_date_str = row.get("issueDate", None)
            if issue_date_str:
                issue_date = pd.to_datetime(issue_date_str).to_pydatetime()
            else:
                # Estimate issue date (assume bond was issued 5 years before execution)
                issue_date = execution_date - pd.Timedelta(days=365 * 5)
                issue_date = issue_date.to_pydatetime()

            # Bond characteristics
            face_value = float(row.get("faceValue", row.get("parValue", 1000.0)))
            coupon_rate = float(row.get("couponRate", yield_rate * 100))  # Convert to percentage

            # Bond type and rating
            bond_type_str = row.get("bondType", "CORPORATE").upper()
            if "TREASURY" in bond_type_str or "GOVERNMENT" in bond_type_str:
                bond_type = BondType.TREASURY
                credit_rating = "AAA"
            elif "HIGH_YIELD" in bond_type_str or "JUNK" in bond_type_str:
                bond_type = BondType.HIGH_YIELD
                credit_rating = row.get("creditRating", "BB")
            else:
                bond_type = BondType.CORPORATE
                credit_rating = row.get("creditRating", "BBB")

            issuer = row.get("issuer", row.get("issuerName", "Unknown"))
            frequency = int(row.get("frequency", row.get("couponFrequency", 2)))

            bond = Bond(
                bond_id=f"FINRA-{bond_id}-{execution_date.strftime('%Y%m%d')}",
                bond_type=bond_type,
                face_value=face_value,
                coupon_rate=coupon_rate,
                maturity_date=maturity_date,
                issue_date=issue_date,
                current_price=price,
                credit_rating=credit_rating,
                issuer=issuer,
                frequency=frequency,
                callable=bool(row.get("callable", False)),
                convertible=bool(row.get("convertible", False)),
            )
            bonds.append(bond)

        except Exception as e:
            logger.warning(f"Error creating bond from FINRA data row {idx}: {e}")
            continue

    print(f"Created {len(bonds)} Bond objects from FINRA data")

    # Save to CSV if requested
    if save_path:
        save_bonds_to_csv(bonds, save_path)

    return bonds


def _generate_synthetic_corporate_bonds(
    start_date: datetime, end_date: datetime, save_path: Optional[str] = None
) -> List[Bond]:
    """
    Generate synthetic corporate bond data when FINRA API is not available
    """
    from datetime import timedelta

    import numpy as np

    print("Generating synthetic corporate bond data...")

    bonds = []
    current_date = start_date

    issuers = [
        "Apple Inc",
        "Microsoft Corp",
        "JPMorgan Chase",
        "Bank of America",
        "Goldman Sachs",
        "Exxon Mobil",
        "AT&T Inc",
        "Verizon Communications",
    ]
    ratings = ["AAA", "AA", "A", "BBB", "BB"]
    rating_probs = [0.1, 0.15, 0.25, 0.35, 0.15]

    # Generate monthly data points
    while current_date <= end_date:
        # Generate 5-10 bonds per month
        num_bonds = np.random.randint(5, 11)

        for _ in range(num_bonds):
            # Random characteristics
            maturity_years = np.random.choice([2, 5, 10, 20, 30], p=[0.2, 0.3, 0.3, 0.15, 0.05])
            rating = np.random.choice(ratings, p=rating_probs)
            issuer = np.random.choice(issuers)

            # Coupon rate based on rating and maturity
            base_rate = 0.02 if rating in ["AAA", "AA"] else 0.03 if rating == "A" else 0.04 if rating == "BBB" else 0.06
            coupon_rate = (base_rate + (maturity_years / 30) * 0.01 + np.random.uniform(-0.005, 0.005)) * 100

            # Calculate dates
            issue_date = current_date - timedelta(days=int(365 * np.random.randint(1, int(maturity_years))))
            maturity_date = current_date + timedelta(days=int(365 * maturity_years))

            face_value = np.random.choice([1000, 5000, 10000])
            # Price varies based on rating and market conditions
            price_variation = np.random.uniform(-0.1, 0.1)
            current_price = face_value * (1 + price_variation)

            bond_id = f"SYNTH-CORP-{current_date.strftime('%Y%m%d')}-{issuer[:3].upper()}-{rating}"

            try:
                bond = Bond(
                    bond_id=bond_id,
                    bond_type=BondType.CORPORATE,
                    face_value=face_value,
                    coupon_rate=coupon_rate,
                    maturity_date=maturity_date,
                    issue_date=issue_date,
                    current_price=current_price,
                    credit_rating=rating,
                    issuer=issuer,
                    frequency=2,
                    callable=np.random.random() < 0.2,
                    convertible=np.random.random() < 0.1,
                )
                bonds.append(bond)
            except Exception as e:
                logger.warning(f"Error creating synthetic bond {bond_id}: {e}")
                continue

        # Move to next month
        if current_date.month == 12:
            current_date = current_date.replace(year=current_date.year + 1, month=1)
        else:
            current_date = current_date.replace(month=current_date.month + 1)

    print(f"Generated {len(bonds)} synthetic corporate bonds")

    if save_path:
        save_bonds_to_csv(bonds, save_path)

    return bonds


def fetch_for_training_evaluation(
    start_year: int = 2010,
    end_year: int = 2020,
    output_dir: str = "historical_data",
    fetch_fred: bool = True,
    fetch_finra: bool = True,
) -> Dict:
    """
    Fetch historical data for training and evaluation from both FRED and FINRA

    Args:
        start_year: Start year (default: 2010)
        end_year: End year (default: 2020)
        output_dir: Directory to save output files
        fetch_fred: Whether to fetch FRED data (default: True)
        fetch_finra: Whether to fetch FINRA data (default: True)

    Returns:
        Dictionary with training and evaluation datasets
    """
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)

    print("=" * 60)
    print("HISTORICAL BOND DATA FETCHER")
    print(f"Fetching data from {start_year} to {end_year}")
    print(f"Sources: FRED={'✓' if fetch_fred else '✗'}, FINRA={'✓' if fetch_finra else '✗'}")
    print("=" * 60)
    print()

    all_bonds = []
    fred_bonds = []
    finra_bonds = []

    # Fetch FRED Treasury bonds
    if fetch_fred:
        print("\n" + "-" * 60)
        print("FETCHING FROM FRED")
        print("-" * 60)
        fred_bonds = fetch_historical_treasury_bonds(
            start_date=start_date,
            end_date=end_date,
            save_path=os.path.join(output_dir, f"fred_treasury_bonds_{start_year}_{end_year}.csv"),
        )
        all_bonds.extend(fred_bonds)
        print(f"✓ Fetched {len(fred_bonds)} bonds from FRED")
    else:
        print("Skipping FRED data fetch")

    # Fetch FINRA bond data
    if fetch_finra:
        print("\n" + "-" * 60)
        print("FETCHING FROM FINRA")
        print("-" * 60)
        finra_bonds = fetch_finra_bonds(
            start_date=start_date,
            end_date=end_date,
            save_path=os.path.join(output_dir, f"finra_bonds_{start_year}_{end_year}.csv"),
        )
        all_bonds.extend(finra_bonds)
        print(f"✓ Fetched {len(finra_bonds)} bonds from FINRA")
    else:
        print("Skipping FINRA data fetch")

    if not all_bonds:
        print("\nNo bonds were created. Check your API keys and date range.")
        return {}

    # Sort bonds by date (if available) for consistent splitting
    try:
        # Try to sort by bond_id which contains date info
        all_bonds.sort(key=lambda b: b.bond_id)
    except (AttributeError, TypeError) as e:
        # Bonds may not have sortable bond_id - not critical
        pass

    # Split into train/eval based on date
    # Use 70% for training (earlier years), 30% for evaluation (later years)
    split_idx = int(len(all_bonds) * 0.7)
    train_bonds = all_bonds[:split_idx]
    eval_bonds = all_bonds[split_idx:]

    print()
    print("=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    print(f"Total bonds: {len(all_bonds)}")
    print(f"  - FRED bonds: {len(fred_bonds) if fetch_fred else 0}")
    print(f"  - FINRA bonds: {len(finra_bonds) if fetch_finra else 0}")
    print(f"Training bonds: {len(train_bonds)}")
    print(f"Evaluation bonds: {len(eval_bonds)}")

    # Save combined data
    os.makedirs(output_dir, exist_ok=True)
    save_bonds_to_csv(all_bonds, os.path.join(output_dir, f"all_bonds_{start_year}_{end_year}.csv"))
    save_bonds_to_csv(train_bonds, os.path.join(output_dir, f"train_bonds_{start_year}_{end_year}.csv"))
    save_bonds_to_csv(eval_bonds, os.path.join(output_dir, f"eval_bonds_{start_year}_{end_year}.csv"))

    return {
        "all_bonds": all_bonds,
        "fred_bonds": fred_bonds,
        "finra_bonds": finra_bonds,
        "train_bonds": train_bonds,
        "eval_bonds": eval_bonds,
        "date_range": {"start": start_date.isoformat(), "end": end_date.isoformat()},
    }


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Fetch historical bond data from FRED/FINRA")
    parser.add_argument("--start-year", type=int, default=2010, help="Start year for historical data (default: 2010)")
    parser.add_argument("--end-year", type=int, default=2020, help="End year for historical data (default: 2020)")
    parser.add_argument(
        "--output-dir", type=str, default="historical_data", help="Output directory for saved data (default: historical_data)"
    )
    parser.add_argument("--fred-only", action="store_true", help="Only fetch from FRED (skip FINRA)")
    parser.add_argument("--finra-only", action="store_true", help="Only fetch from FINRA (skip FRED)")

    args = parser.parse_args()

    # Determine which sources to fetch
    fetch_fred = not args.finra_only
    fetch_finra = not args.fred_only

    try:
        result = fetch_for_training_evaluation(
            start_year=args.start_year,
            end_year=args.end_year,
            output_dir=args.output_dir,
            fetch_fred=fetch_fred,
            fetch_finra=fetch_finra,
        )

        if result:
            print()
            print("=" * 60)
            print("SUCCESS!")
            print("=" * 60)
            print(f"Data saved to: {args.output_dir}")
            print(f"\nFetched {len(result.get('all_bonds', []))} total bonds:")
            if result.get("fred_bonds"):
                print(f"  - {len(result['fred_bonds'])} from FRED")
            if result.get("finra_bonds"):
                print(f"  - {len(result['finra_bonds'])} from FINRA")
            print("\nYou can now use this data for training and evaluation.")
        else:
            print("\nFailed to fetch data. Please check your API keys and try again.")
            sys.exit(1)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
