"""
FINRA API Troubleshooting Script
Tests FINRA API authentication and data fetching
"""

import base64
import os
import sys
from datetime import datetime
from pathlib import Path

import requests

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env file
try:
    from dotenv import load_dotenv

    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()

from bondtrader.config import get_config
from bondtrader.data.market_data import FINRADataProvider


def test_finra_authentication():
    """Test FINRA OAuth2 authentication"""
    print("=" * 70)
    print("TESTING FINRA AUTHENTICATION")
    print("=" * 70)

    config = get_config()

    if not config.finra_api_key or not config.finra_api_password:
        print("ERROR: FINRA API credentials not found in config")
        print(f"  FINRA_API_KEY: {'Set' if config.finra_api_key else 'Not set'}")
        print(f"  FINRA_API_PASSWORD: {'Set' if config.finra_api_password else 'Not set'}")
        return None

    print(f"✓ API Key found: {config.finra_api_key[:10]}...")
    print(f"✓ API Password found: {'*' * len(config.finra_api_password)}")

    # Test authentication
    auth_url = "https://ews.fip.finra.org/fip/rest/ews/oauth2/access_token"

    try:
        credentials = f"{config.finra_api_key}:{config.finra_api_password}"
        b64_credentials = base64.b64encode(credentials.encode()).decode()

        headers = {"Authorization": f"Basic {b64_credentials}", "Content-Type": "application/x-www-form-urlencoded"}

        # FINRA requires grant_type as query parameter per their documentation
        auth_url_with_grant = f"{auth_url}?grant_type=client_credentials"

        print(f"\nRequesting access token from: {auth_url}")
        print(f"Using grant_type: client_credentials (in URL query parameter)")

        response = requests.post(auth_url_with_grant, headers=headers, timeout=10)

        print(f"Response Status: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")

        if response.status_code == 200:
            data = response.json()
            print(f"\n✓ Authentication successful!")
            print(f"  Access Token: {data.get('access_token', 'N/A')[:20]}...")
            print(f"  Token Type: {data.get('token_type', 'N/A')}")
            print(f"  Expires In: {data.get('expires_in', 'N/A')} seconds")
            return data.get("access_token")
        else:
            print(f"\n✗ Authentication failed!")
            print(f"  Status Code: {response.status_code}")
            print(f"  Response: {response.text}")
            return None

    except Exception as e:
        print(f"\n✗ Error during authentication: {e}")
        import traceback

        traceback.print_exc()
        return None


def test_finra_endpoints(access_token):
    """Test different FINRA API endpoints"""
    if not access_token:
        print("\nSkipping endpoint tests (no access token)")
        return

    print("\n" + "=" * 70)
    print("TESTING FINRA API ENDPOINTS")
    print("=" * 70)

    api_base = "https://api.finra.org"
    headers = {"Authorization": f"Bearer {access_token}"}

    # Test different endpoint patterns
    endpoints_to_test = [
        "/data/group/fixedIncomeMarket/name/traceHistoric",
        "/data/group/fixedIncomeMarket/name/trace",
        "/data/group/fixedIncomeMarket/name/traceDailyAggregates",
        "/data/group/fixedIncomeMarket/name/treasuryDailyAggregates",
        "/data/group/fixedIncomeMarket/datasets",
        "/data/group/fixedIncomeMarket",
        "/data/catalog",
    ]

    start_date = datetime(2010, 1, 1)
    end_date = datetime(2010, 1, 31)  # Test with small date range

    for endpoint in endpoints_to_test:
        url = f"{api_base}{endpoint}"
        print(f"\nTesting: {endpoint}")

        try:
            # Try with date parameters
            params = {
                "startDate": start_date.strftime("%Y-%m-%d"),
                "endDate": end_date.strftime("%Y-%m-%d"),
                "limit": 10,
                "offset": 0,
            }

            response = requests.get(url, headers=headers, params=params, timeout=30)
            print(f"  Status: {response.status_code}")

            if response.status_code == 200:
                try:
                    data = response.json()
                    print(f"  ✓ Success! Response type: {type(data)}")
                    if isinstance(data, dict):
                        print(f"  Keys: {list(data.keys())[:10]}")
                        if "data" in data:
                            print(f"  Data records: {len(data.get('data', []))}")
                    elif isinstance(data, list):
                        print(f"  List length: {len(data)}")
                except (ValueError, KeyError, TypeError) as e:
                    # Fallback to showing raw response text
                    print(f"  Response (first 200 chars): {response.text[:200]}")
            elif response.status_code == 401:
                print(f"  ✗ Unauthorized - token may be invalid or expired")
            elif response.status_code == 404:
                print(f"  ✗ Not Found - endpoint doesn't exist")
            elif response.status_code == 403:
                print(f"  ✗ Forbidden - may need different permissions")
            else:
                print(f"  ✗ Error: {response.status_code}")
                print(f"  Response: {response.text[:200]}")

        except Exception as e:
            print(f"  ✗ Exception: {e}")


def test_finra_provider():
    """Test using the FINRADataProvider class"""
    print("\n" + "=" * 70)
    print("TESTING FINRADataProvider CLASS")
    print("=" * 70)

    try:
        finra = FINRADataProvider()

        print("\n1. Testing access token retrieval...")
        token = finra._get_access_token()
        if token:
            print(f"   ✓ Got access token: {token[:20]}...")
        else:
            print("   ✗ Failed to get access token")
            return

        print("\n2. Testing historical data fetch...")
        start_date = datetime(2010, 1, 1)
        end_date = datetime(2010, 1, 31)

        data = finra.fetch_historical_bond_data(start_date, end_date)

        if data is not None and not data.empty:
            print(f"   ✓ Successfully fetched {len(data)} records")
            print(f"   Columns: {list(data.columns)}")
            print(f"   Sample data:")
            print(data.head())
        else:
            print("   ✗ No data returned")
            print("   This could mean:")
            print("     - Endpoint doesn't exist")
            print("     - Date range has no data")
            print("     - API structure is different than expected")

    except Exception as e:
        print(f"✗ Error testing provider: {e}")
        import traceback

        traceback.print_exc()


def main():
    """Main troubleshooting function"""
    print("FINRA API TROUBLESHOOTING")
    print("=" * 70)

    # Step 1: Test authentication
    access_token = test_finra_authentication()

    # Step 2: Test endpoints
    if access_token:
        test_finra_endpoints(access_token)

    # Step 3: Test provider class
    test_finra_provider()

    print("\n" + "=" * 70)
    print("TROUBLESHOOTING COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Check FINRA API documentation for correct endpoints")
    print("2. Verify your API credentials have the right permissions")
    print("3. Check if your account has access to TRACE data")
    print("4. Contact FINRA support if authentication works but endpoints fail")


if __name__ == "__main__":
    main()
