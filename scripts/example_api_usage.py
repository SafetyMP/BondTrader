"""
Example: Using FRED and FINRA APIs
Demonstrates how to make API calls to fetch historical bond data
"""

import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bondtrader.config import get_config
from bondtrader.data.market_data import FINRADataProvider, FREDDataProvider, MarketDataManager


def example_fred_usage():
    """Example: Using FRED API to fetch historical Treasury data"""
    print("=" * 60)
    print("EXAMPLE: FRED API Usage")
    print("=" * 60)

    config = get_config()

    if not config.fred_api_key:
        print("ERROR: FRED_API_KEY not set in environment variables")
        print("Please set it in your .env file")
        return

    # Initialize FRED provider
    fred = FREDDataProvider()

    # Example 1: Fetch current risk-free rate
    print("\n1. Fetching current risk-free rate...")
    current_rate = fred.fetch_risk_free_rate()
    if current_rate:
        print(f"   Current 10-year Treasury rate: {current_rate:.4f} ({current_rate*100:.2f}%)")

    # Example 2: Fetch historical risk-free rate
    print("\n2. Fetching historical risk-free rate (Jan 1, 1990)...")
    historical_date = datetime(1990, 1, 1)
    historical_rate = fred.fetch_risk_free_rate(historical_date)
    if historical_rate:
        print(f"   10-year Treasury rate on {historical_date.date()}: {historical_rate:.4f} ({historical_rate*100:.2f}%)")

    # Example 3: Fetch historical Treasury data (1980s-1990s)
    print("\n3. Fetching historical Treasury data (1980-1999)...")
    start_date = datetime(1980, 1, 1)
    end_date = datetime(1999, 12, 31)

    treasury_data = fred.fetch_historical_treasury_data(
        start_date=start_date, end_date=end_date, maturities=["GS1", "GS2", "GS5", "GS10", "GS30"]
    )

    if treasury_data is not None and not treasury_data.empty:
        print(f"   Retrieved {len(treasury_data)} data points")
        print(f"   Date range: {treasury_data.index.min().date()} to {treasury_data.index.max().date()}")
        print(f"   Available maturities: {list(treasury_data.columns)}")
        print("\n   Sample data (first 5 rows):")
        print(treasury_data.head())
        print("\n   Sample data (last 5 rows):")
        print(treasury_data.tail())
    else:
        print("   No data retrieved")

    # Example 4: Fetch yield curve
    print("\n4. Fetching yield curve (Jan 1, 1985)...")
    curve_date = datetime(1985, 1, 1)
    yield_curve = fred.fetch_yield_curve(curve_date)
    if yield_curve:
        print(f"   Maturities (years): {yield_curve['maturities']}")
        print(f"   Yields: {[f'{y*100:.2f}%' for y in yield_curve['yields']]}")


def example_finra_usage():
    """Example: Using FINRA API (Note: 1980s-1990s data not available)"""
    print("\n" + "=" * 60)
    print("EXAMPLE: FINRA API Usage")
    print("=" * 60)

    config = get_config()

    if not config.finra_api_key or not config.finra_api_password:
        print("WARNING: FINRA API credentials not set in environment variables")
        print("FINRA API requires both FINRA_API_KEY and FINRA_API_PASSWORD")
        print("Note: FINRA TRACE data is only available from 2002 onward (example uses 2010-2020)")
        return

    # Initialize FINRA provider
    finra = FINRADataProvider()

    # Example: Try to fetch data (will warn if date range is too early)
    print("\n1. Attempting to fetch FINRA data for 2010-2020...")
    start_date = datetime(2010, 1, 1)
    end_date = datetime(2020, 12, 31)

    bond_data = finra.fetch_historical_bond_data(start_date=start_date, end_date=end_date)

    if bond_data is not None:
        print(f"   Retrieved {len(bond_data)} records")
        print(f"   Columns: {list(bond_data.columns)}")
        print("\n   Sample data:")
        print(bond_data.head())
    else:
        print("   No data retrieved (may require proper FINRA entitlements)")


def example_market_data_manager():
    """Example: Using MarketDataManager (high-level interface)"""
    print("\n" + "=" * 60)
    print("EXAMPLE: MarketDataManager Usage")
    print("=" * 60)

    manager = MarketDataManager()

    # Example 1: Get risk-free rate
    print("\n1. Getting current risk-free rate...")
    rate = manager.get_risk_free_rate()
    print(f"   Risk-free rate: {rate:.4f} ({rate*100:.2f}%)")

    # Example 2: Get yield curve
    print("\n2. Getting current yield curve...")
    curve = manager.get_yield_curve()
    if curve:
        print(f"   Maturities: {curve['maturities']}")
        print(f"   Yields: {[f'{y*100:.2f}%' for y in curve['yields']]}")

    # Example 3: Fetch historical data
    print("\n3. Fetching historical Treasury data...")
    start_date = datetime(1985, 1, 1)
    end_date = datetime(1995, 12, 31)

    historical_data = manager.fetch_historical_treasury_data(start_date=start_date, end_date=end_date)

    if historical_data is not None and not historical_data.empty:
        print(f"   Retrieved {len(historical_data)} data points")
        print(f"   Date range: {historical_data.index.min().date()} to {historical_data.index.max().date()}")


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("BONDTRADER API USAGE EXAMPLES")
    print("=" * 60)

    # Check configuration
    config = get_config()
    print(f"\nConfiguration:")
    print(f"  FRED API Key: {'✓ Set' if config.fred_api_key else '✗ Not set'}")
    print(f"  FINRA API Key: {'✓ Set' if config.finra_api_key else '✗ Not set'}")
    print(f"  FINRA API Password: {'✓ Set' if config.finra_api_password else '✗ Not set'}")

    # Run examples
    example_fred_usage()
    example_finra_usage()
    example_market_data_manager()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)
    print("\nFor more information, see:")
    print("  - docs/guides/HISTORICAL_DATA_FETCHING.md")
    print("  - scripts/fetch_historical_data.py")


if __name__ == "__main__":
    main()
