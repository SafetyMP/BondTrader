"""
Market Data Integration Module
Provides interfaces for fetching real market data from various sources
"""

# yfinance is optional - import only when needed to avoid Python 3.9 compatibility issues
# (yfinance 0.2.0+ uses Python 3.10+ union syntax which causes TypeError on Python 3.9)
import base64
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import requests

try:
    # Check Python version - yfinance 0.2.0+ requires Python 3.10+
    if sys.version_info < (3, 10):
        HAS_YFINANCE = False
        yf = None
    else:
        import yfinance as yf

        HAS_YFINANCE = True
except (ImportError, TypeError, SyntaxError, Exception):
    # Catch all exceptions during import (yfinance may fail during module init)
    HAS_YFINANCE = False
    yf = None

from bondtrader.config import get_config
from bondtrader.core.bond_models import Bond, BondType
from bondtrader.utils.utils import logger


class MarketDataProvider:
    """Base class for market data providers"""

    def fetch_bond_data(self, bond_id: str) -> Optional[Dict]:
        """Fetch bond data from market source"""
        raise NotImplementedError

    def fetch_yield_curve(self) -> Optional[Dict]:
        """Fetch yield curve data"""
        raise NotImplementedError


class TreasuryDataProvider(MarketDataProvider):
    """Fetch US Treasury data from public sources"""

    def __init__(self):
        """Initialize Treasury data provider"""
        self.base_url = "https://www.quandl.com/api/v3/datasets"
        # Note: In production, would use API keys

    def fetch_treasury_rates(self) -> Optional[Dict]:
        """Fetch current Treasury rates"""
        try:
            # Simplified - in production would fetch from actual API
            # Using FRED or Treasury website data
            logger.info("Fetching Treasury rates (simulated)")

            # Return sample structure
            return {
                "1_month": 0.02,
                "3_month": 0.025,
                "6_month": 0.03,
                "1_year": 0.035,
                "2_year": 0.04,
                "5_year": 0.045,
                "10_year": 0.047,
                "30_year": 0.05,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Error fetching Treasury rates: {e}")
            return None

    def fetch_yield_curve(self) -> Optional[Dict]:
        """Fetch Treasury yield curve"""
        rates = self.fetch_treasury_rates()
        if rates:
            return {
                "maturities": [0.083, 0.25, 0.5, 1, 2, 5, 10, 30],
                "yields": [
                    rates["1_month"],
                    rates["3_month"],
                    rates["6_month"],
                    rates["1_year"],
                    rates["2_year"],
                    rates["5_year"],
                    rates["10_year"],
                    rates["30_year"],
                ],
                "timestamp": datetime.now().isoformat(),
            }
        return None


class YahooFinanceProvider(MarketDataProvider):
    """Fetch bond data from Yahoo Finance"""

    def fetch_bond_data(self, symbol: str) -> Optional[Dict]:
        """Fetch bond data from Yahoo Finance"""
        try:
            # Note: Yahoo Finance bond data access is limited
            # This is a placeholder for actual implementation
            logger.info(f"Fetching data for {symbol} from Yahoo Finance")

            # In production, would use actual Yahoo Finance API or library
            # For now, return None to indicate not implemented
            return None
        except Exception as e:
            logger.error(f"Error fetching from Yahoo Finance: {e}")
            return None


class FREDDataProvider(MarketDataProvider):
    """Fetch economic data from FRED (Federal Reserve Economic Data)"""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize FRED provider"""
        config = get_config()
        self.api_key = api_key or config.fred_api_key
        self.base_url = "https://api.stlouisfed.org/fred/series/observations"

        if not self.api_key:
            logger.warning("FRED API key not provided. Some functions may not work.")

    def fetch_risk_free_rate(self, date: Optional[datetime] = None) -> Optional[float]:
        """Fetch risk-free rate (10-year Treasury) for a specific date or current"""
        try:
            if not self.api_key:
                logger.warning("FRED API key not available")
                return None

            # Use GS10 (10-year Treasury constant maturity) series
            series_id = "GS10"
            end_date = date.strftime("%Y-%m-%d") if date else datetime.now().strftime("%Y-%m-%d")

            params = {
                "series_id": series_id,
                "api_key": self.api_key,
                "file_type": "json",
                "observation_end": end_date,
                "limit": 1,
                "sort_order": "desc",
                "units": "lin",  # Linear units (not transformed)
            }

            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if "observations" in data and len(data["observations"]) > 0:
                # Get the most recent observation
                obs = data["observations"][0]
                if obs.get("value") != ".":  # FRED uses "." for missing values
                    rate = float(obs["value"]) / 100.0  # Convert percentage to decimal
                    logger.info(f"Fetched risk-free rate from FRED: {rate:.4f}")
                    return rate

            logger.warning("No valid data found in FRED response")
            return None
        except Exception as e:
            logger.error(f"Error fetching risk-free rate from FRED: {e}")
            return None

    def fetch_historical_treasury_data(
        self, start_date: datetime, end_date: datetime, maturities: Optional[List[str]] = None
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical Treasury bond yields from FRED

        Args:
            start_date: Start date for historical data
            end_date: End date for historical data
            maturities: List of maturity series to fetch. Default includes common maturities.
                       Options: 'GS1MO', 'GS3MO', 'GS6MO', 'GS1', 'GS2', 'GS5', 'GS7', 'GS10', 'GS20', 'GS30'

        Returns:
            DataFrame with dates as index and maturity columns
        """
        if not self.api_key:
            logger.error("FRED API key not available")
            return None

        if maturities is None:
            # Default to common maturities available in 1980s-1990s
            maturities = ["GS1", "GS2", "GS5", "GS10", "GS30"]  # 1, 2, 5, 10, 30 year

        try:
            all_data = {}

            for series_id in maturities:
                params = {
                    "series_id": series_id,
                    "api_key": self.api_key,
                    "file_type": "json",
                    "observation_start": start_date.strftime("%Y-%m-%d"),
                    "observation_end": end_date.strftime("%Y-%m-%d"),
                    "sort_order": "asc",
                }

                logger.info(f"Fetching {series_id} from FRED...")
                response = requests.get(self.base_url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()

                if "observations" in data:
                    # Convert to DataFrame
                    obs_list = []
                    for obs in data["observations"]:
                        if obs.get("value") != ".":  # Skip missing values
                            obs_list.append(
                                {
                                    "date": pd.to_datetime(obs["date"]),
                                    series_id: float(obs["value"]) / 100.0,  # Convert % to decimal
                                }
                            )

                    if obs_list:
                        df_series = pd.DataFrame(obs_list).set_index("date")
                        all_data[series_id] = df_series[series_id]
                        logger.info(f"Fetched {len(obs_list)} observations for {series_id}")

            if all_data:
                # Combine all series into one DataFrame
                result_df = pd.DataFrame(all_data)
                result_df.index.name = "date"
                logger.info(f"Successfully fetched historical Treasury data: {result_df.shape}")
                return result_df
            else:
                logger.warning("No data retrieved from FRED")
                return None

        except Exception as e:
            logger.error(f"Error fetching historical Treasury data from FRED: {e}")
            return None

    def fetch_yield_curve(self, date: Optional[datetime] = None) -> Optional[Dict]:
        """Fetch yield curve data for a specific date or current"""
        if not self.api_key:
            logger.warning("FRED API key not available")
            return None

        try:
            # Fetch multiple maturities
            maturities_map = {
                "1_month": "GS1MO",
                "3_month": "GS3MO",
                "6_month": "GS6MO",
                "1_year": "GS1",
                "2_year": "GS2",
                "5_year": "GS5",
                "7_year": "GS7",
                "10_year": "GS10",
                "20_year": "GS20",
                "30_year": "GS30",
            }

            target_date = date or datetime.now()
            end_date = target_date.strftime("%Y-%m-%d")

            yields = {}
            for maturity_name, series_id in maturities_map.items():
                params = {
                    "series_id": series_id,
                    "api_key": self.api_key,
                    "file_type": "json",
                    "observation_end": end_date,
                    "limit": 1,
                    "sort_order": "desc",
                }

                response = requests.get(self.base_url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if "observations" in data and len(data["observations"]) > 0:
                        obs = data["observations"][0]
                        if obs.get("value") != ".":
                            yields[maturity_name] = float(obs["value"]) / 100.0

            if yields:
                # Convert to standard format
                maturity_years = {
                    "1_month": 1 / 12,
                    "3_month": 0.25,
                    "6_month": 0.5,
                    "1_year": 1.0,
                    "2_year": 2.0,
                    "5_year": 5.0,
                    "7_year": 7.0,
                    "10_year": 10.0,
                    "20_year": 20.0,
                    "30_year": 30.0,
                }

                maturities = []
                yield_values = []
                for name, years in maturity_years.items():
                    if name in yields:
                        maturities.append(years)
                        yield_values.append(yields[name])

                return {
                    "maturities": maturities,
                    "yields": yield_values,
                    "timestamp": target_date.isoformat(),
                }

            return None
        except Exception as e:
            logger.error(f"Error fetching yield curve from FRED: {e}")
            return None


class FINRADataProvider(MarketDataProvider):
    """Fetch bond data from FINRA (Financial Industry Regulatory Authority)"""

    def __init__(self, api_key: Optional[str] = None, api_password: Optional[str] = None):
        """Initialize FINRA provider with OAuth2 authentication"""
        config = get_config()
        self.api_key = api_key or config.finra_api_key
        self.api_password = api_password or config.finra_api_password
        self.auth_url = "https://ews.fip.finra.org/fip/rest/ews/oauth2/access_token"
        self.api_base = "https://api.finra.org"
        self._access_token: Optional[str] = None
        self._token_expires_at: Optional[datetime] = None

        if not self.api_key or not self.api_password:
            logger.warning("FINRA API credentials not provided. Some functions may not work.")

    def _get_access_token(self) -> Optional[str]:
        """Get OAuth2 access token, refreshing if needed"""
        # Check if we have a valid token
        if self._access_token and self._token_expires_at:
            if datetime.now() < self._token_expires_at - timedelta(seconds=60):  # Refresh 1 min early
                return self._access_token

        if not self.api_key or not self.api_password:
            logger.error("FINRA API credentials not available")
            return None

        try:
            # Create Basic Auth header
            credentials = f"{self.api_key}:{self.api_password}"
            b64_credentials = base64.b64encode(credentials.encode()).decode()

            headers = {"Authorization": f"Basic {b64_credentials}", "Content-Type": "application/x-www-form-urlencoded"}

            # FINRA requires grant_type as query parameter per their documentation
            auth_url_with_grant = f"{self.auth_url}?grant_type=client_credentials"

            logger.info("Requesting FINRA access token...")
            response = requests.post(auth_url_with_grant, headers=headers, timeout=10)
            response.raise_for_status()

            data = response.json()
            self._access_token = data.get("access_token")
            expires_in = data.get("expires_in", 3600)  # Default to 1 hour

            # Ensure expires_in is an integer
            if isinstance(expires_in, str):
                try:
                    expires_in = int(expires_in)
                except ValueError:
                    expires_in = 3600

            # Set expiration time
            self._token_expires_at = datetime.now() + timedelta(seconds=int(expires_in))

            logger.info("Successfully obtained FINRA access token")
            return self._access_token

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 400:
                error_data = e.response.json() if e.response.text else {}
                error_msg = error_data.get("error_message", "Unknown error")
                if "Invalid Credentials" in error_msg or "invalid_client" in error_data.get("error", ""):
                    logger.error(
                        f"FINRA API authentication failed: Invalid credentials. "
                        f"Please verify FINRA_API_KEY and FINRA_API_PASSWORD in your .env file. "
                        f"Note: FINRA API may require account activation or special permissions."
                    )
                else:
                    logger.error(f"Error obtaining FINRA access token: {error_msg}")
            else:
                logger.error(f"Error obtaining FINRA access token: {e}")
            return None
        except Exception as e:
            logger.error(f"Error obtaining FINRA access token: {e}")
            return None

    def fetch_bond_data(self, bond_id: str) -> Optional[Dict]:
        """
        Fetch bond data from FINRA

        Note: FINRA TRACE data is only available from 2002 onward.
        For 1980s-1990s data, use FRED instead.
        """
        token = self._get_access_token()
        if not token:
            logger.warning("Cannot fetch FINRA data: no access token")
            return None

        try:
            headers = {"Authorization": f"Bearer {token}"}

            # Example endpoint - adjust based on actual FINRA API structure
            # This is a placeholder as FINRA API structure may vary
            url = f"{self.api_base}/data/group/fixedIncomeMarket/name/treasuryDailyAggregates"

            params = {"limit": 1000, "offset": 0}

            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            logger.info(f"Fetched data from FINRA for bond {bond_id}")
            return data

        except Exception as e:
            logger.error(f"Error fetching bond data from FINRA: {e}")
            return None

    def fetch_historical_bond_data(
        self, start_date: datetime, end_date: datetime, cusip: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical bond transaction data from FINRA

        Uses available FINRA fixed income datasets:
        - treasuryDailyAggregates: Treasury bond trading data
        - corporateDebtMarketBreadth: Corporate bond market data

        Args:
            start_date: Start date for historical data
            end_date: End date for historical data
            cusip: Optional CUSIP identifier to filter by specific bond

        Returns:
            DataFrame with transaction data, or None if data not available
        """
        token = self._get_access_token()
        if not token:
            return None

        try:
            headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}

            all_data = []

            # Try datasets - use Mock versions if credentials are for mock environment
            # FINRA returns 403 if trying to access production datasets with mock credentials
            datasets_to_try = [
                "treasuryDailyAggregatesMock",  # Mock version for testing
                "treasuryMonthlyAggregatesMock",
                "treasuryWeeklyAggregatesMock",
                "corporateMarketBreadthMock",
                "agencyMarketBreadthMock",
                # Try production versions as fallback (will fail if mock credentials)
                "treasuryDailyAggregates",
                "treasuryMonthlyAggregates",
                "corporateDebtMarketBreadth",
                "agencyMarketBreadth",
            ]

            for dataset_name in datasets_to_try:
                try:
                    url = f"{self.api_base}/data/group/fixedIncomeMarket/name/{dataset_name}"

                    # Build filters for date range
                    # FINRA uses compareFilters format: field,compareType,value
                    params = {
                        "limit": 10000,
                    }

                    # Add date filter - FINRA datasets typically use tradeDate
                    # Format: field,compareType,value (comma-separated for multiple)
                    trade_date_gte = f"tradeDate,GTE,{start_date.strftime('%Y-%m-%d')}"
                    trade_date_lte = f"tradeDate,LTE,{end_date.strftime('%Y-%m-%d')}"
                    params["compareFilters"] = f"{trade_date_gte};{trade_date_lte}"

                    logger.info(f"Fetching FINRA data from {dataset_name}...")
                    response = requests.get(url, headers=headers, params=params, timeout=60)

                    if response.status_code == 200:
                        data = response.json()

                        # Handle different response formats
                        if isinstance(data, dict):
                            if "data" in data:
                                records = data["data"]
                            elif "results" in data:
                                records = data["results"]
                            else:
                                records = [data]
                        elif isinstance(data, list):
                            records = data
                        else:
                            continue

                        if records and len(records) > 0:
                            df = pd.DataFrame(records)
                            # Add dataset name for tracking
                            df["_dataset"] = dataset_name
                            all_data.append(df)
                            logger.info(f"  ✓ Fetched {len(df)} records from {dataset_name}")
                    elif response.status_code == 403:
                        logger.warning(f"  ✗ Access forbidden for {dataset_name} - may need permissions")
                    elif response.status_code == 404:
                        logger.warning(f"  ✗ Dataset {dataset_name} not found")
                    else:
                        logger.warning(f"  ✗ Error {response.status_code} for {dataset_name}")

                except Exception as e:
                    logger.warning(f"  ✗ Error fetching {dataset_name}: {e}")
                    continue

            if all_data:
                # Combine all datasets
                combined_df = pd.concat(all_data, ignore_index=True)
                logger.info(f"Fetched {len(combined_df)} total records from FINRA")
                return combined_df
            else:
                logger.warning("No data retrieved from any FINRA dataset")
                return None

        except Exception as e:
            logger.error(f"Error fetching historical bond data from FINRA: {e}")
            return None

    def fetch_yield_curve(self) -> Optional[Dict]:
        """Fetch yield curve data from FINRA (if available)"""
        # FINRA may not provide yield curve data directly
        # This is a placeholder for future implementation
        logger.warning("FINRA yield curve not yet implemented")
        return None


class MarketDataManager:
    """Manages multiple market data sources"""

    def __init__(self):
        """Initialize market data manager"""
        config = get_config()
        self.providers = {
            "treasury": TreasuryDataProvider(),
            "yfinance": YahooFinanceProvider(),
            "fred": FREDDataProvider(),
            "finra": FINRADataProvider(),
        }

    def get_risk_free_rate(self, date: Optional[datetime] = None) -> float:
        """Get risk-free rate for a specific date or current"""
        rate = self.providers["fred"].fetch_risk_free_rate(date)
        if rate:
            return rate
        # Fallback
        return 0.03

    def get_yield_curve(self, date: Optional[datetime] = None) -> Optional[Dict]:
        """Get yield curve for a specific date or current"""
        curve = self.providers["fred"].fetch_yield_curve(date)
        if curve:
            return curve
        # Fallback to treasury provider
        return self.providers["treasury"].fetch_yield_curve()

    def fetch_historical_treasury_data(
        self, start_date: datetime, end_date: datetime, maturities: Optional[List[str]] = None
    ) -> Optional[pd.DataFrame]:
        """Fetch historical Treasury data from FRED"""
        return self.providers["fred"].fetch_historical_treasury_data(start_date, end_date, maturities)

    def convert_market_data_to_bond(self, market_data: Dict, bond_id: str) -> Optional[Bond]:
        """Convert market data to Bond object"""
        try:
            # This would parse market data into Bond format
            # Simplified implementation
            logger.info(f"Converting market data to Bond: {bond_id}")
            return None  # Placeholder
        except Exception as e:
            logger.error(f"Error converting market data: {e}")
            return None
