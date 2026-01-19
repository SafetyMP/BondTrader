# Historical Bond Data Fetching Guide

This guide explains how to fetch historical bond data from FRED and FINRA APIs for training and evaluation.

## Overview

The BondTrader system supports fetching historical bond data from:
- **FRED (Federal Reserve Economic Data)**: Treasury bond yields from 1980s-1990s
- **FINRA (Financial Industry Regulatory Authority)**: TRACE bond transaction data (2002+)

## Prerequisites

1. **FRED API Key**: Get a free API key from [FRED API Key Registration](https://fred.stlouisfed.org/docs/api/api_key.html)
2. **FINRA API Credentials**: Requires FINRA Entitlement Agreement and API Console access
3. **Environment Variables**: Set in your `.env` file:
   ```env
   FRED_API_KEY=your_fred_api_key_here
   FINRA_API_KEY=your_finra_client_id
   FINRA_API_PASSWORD=your_finra_client_secret
   ```

## Quick Start

### Fetch Historical Data (1980s-1990s)

Use the provided script to fetch Treasury bond data:

```bash
python scripts/fetch_historical_data.py --start-year 1980 --end-year 1999
```

This will:
1. Fetch historical Treasury yields from FRED
2. Convert the data to Bond objects
3. Split into training (70%) and evaluation (30%) sets
4. Save to CSV files in `historical_data/` directory

### Command Line Options

```bash
python scripts/fetch_historical_data.py \
    --start-year 1980 \
    --end-year 1999 \
    --output-dir historical_data
```

- `--start-year`: Start year (default: 1980)
- `--end-year`: End year (default: 1999)
- `--output-dir`: Output directory (default: historical_data)

## Programmatic Usage

### Using FRED API

```python
from datetime import datetime
from bondtrader.data.market_data import MarketDataManager

# Initialize manager (uses API keys from config)
manager = MarketDataManager()

# Fetch historical Treasury data
start_date = datetime(1980, 1, 1)
end_date = datetime(1999, 12, 31)

treasury_data = manager.fetch_historical_treasury_data(
    start_date=start_date,
    end_date=end_date,
    maturities=['GS1', 'GS2', 'GS5', 'GS10', 'GS30']  # 1, 2, 5, 10, 30 year
)

print(treasury_data.head())
```

### Using FINRA API

```python
from datetime import datetime
from bondtrader.data.market_data import FINRADataProvider

# Initialize provider (uses API keys from config)
finra = FINRADataProvider()

# Fetch historical bond data (2002+ only)
start_date = datetime(2010, 1, 1)
end_date = datetime(2020, 12, 31)

bond_data = finra.fetch_historical_bond_data(
    start_date=start_date,
    end_date=end_date
)

if bond_data is not None:
    print(bond_data.head())
else:
    print("FINRA data not available for this date range")
```

### Fetch Risk-Free Rate

```python
from datetime import datetime
from bondtrader.data.market_data import MarketDataManager

manager = MarketDataManager()

# Current risk-free rate
current_rate = manager.get_risk_free_rate()
print(f"Current risk-free rate: {current_rate:.4f}")

# Historical risk-free rate
historical_date = datetime(1990, 1, 1)
historical_rate = manager.get_risk_free_rate(historical_date)
print(f"Risk-free rate on {historical_date.date()}: {historical_rate:.4f}")
```

### Fetch Yield Curve

```python
from datetime import datetime
from bondtrader.data.market_data import MarketDataManager

manager = MarketDataManager()

# Current yield curve
curve = manager.get_yield_curve()
print(f"Maturities: {curve['maturities']}")
print(f"Yields: {curve['yields']}")

# Historical yield curve
historical_date = datetime(1985, 6, 15)
historical_curve = manager.get_yield_curve(historical_date)
```

## Data Format

### Treasury Data (FRED)

The FRED provider returns a pandas DataFrame with:
- **Index**: Date (datetime)
- **Columns**: Treasury series (GS1, GS2, GS5, GS10, GS30)
- **Values**: Yields as decimals (e.g., 0.05 for 5%)

### Bond Objects

The fetched data is converted to `Bond` objects with:
- `bond_id`: Unique identifier
- `bond_type`: BondType.TREASURY
- `face_value`: $1000 (standard)
- `coupon_rate`: Annual coupon rate as percentage
- `maturity_date`: Calculated maturity date
- `issue_date`: Calculated issue date
- `current_price`: Bond price
- `credit_rating`: "AAA" for Treasury bonds
- `issuer`: "US Treasury"

## Available FRED Series

Common Treasury series available:

| Series ID | Description | Available From |
|-----------|-------------|----------------|
| GS1MO | 1-Month Treasury | 2001-07-31 |
| GS3MO | 3-Month Treasury | 1982-01-04 |
| GS6MO | 6-Month Treasury | 1982-01-04 |
| GS1 | 1-Year Treasury | 1962-01-02 |
| GS2 | 2-Year Treasury | 1976-06-01 |
| GS5 | 5-Year Treasury | 1962-01-02 |
| GS7 | 7-Year Treasury | 1969-07-01 |
| GS10 | 10-Year Treasury | 1962-01-02 |
| GS20 | 20-Year Treasury | 1993-10-01 |
| GS30 | 30-Year Treasury | 1977-02-15 |

**Note**: GS30 was discontinued Feb 2002 - Feb 2006, but available for 1980s-1990s.

## FINRA Data Limitations

**Important**: FINRA TRACE data is only available from **2002 onward**. 

For 1980s-1990s data:
- ✅ Use **FRED** for Treasury yields
- ❌ FINRA TRACE data not available for this period

## Integration with Training Pipeline

After fetching historical data, you can use it with the training pipeline:

```python
from bondtrader.data.training_data_generator import TrainingDataGenerator
from scripts.fetch_historical_data import fetch_historical_treasury_bonds
from datetime import datetime

# Fetch historical bonds
bonds = fetch_historical_treasury_bonds(
    start_date=datetime(1980, 1, 1),
    end_date=datetime(1999, 12, 31)
)

# Use with training generator
generator = TrainingDataGenerator(seed=42)

# Generate comprehensive dataset using historical bonds
# (You may need to adapt the generator to accept pre-fetched bonds)
```

## Troubleshooting

### FRED API Issues

**Error: "FRED API key not available"**
- Check that `FRED_API_KEY` is set in your `.env` file
- Verify the API key is valid at [FRED API Key Page](https://fred.stlouisfed.org/docs/api/api_key.html)

**Error: "No data retrieved from FRED"**
- Check the date range - some series may not have data for all dates
- Verify the series ID is correct
- Check FRED API status

### FINRA API Issues

**Error: "FINRA API credentials not available"**
- Ensure both `FINRA_API_KEY` and `FINRA_API_PASSWORD` are set
- Verify you have FINRA Entitlement Agreement in place
- Check that MFA is enabled on your FINRA account

**Error: "FINRA TRACE data is only available from 2002 onward"**
- This is expected - TRACE system started in 2002
- Use FRED for earlier data

### Rate Limiting

Both APIs have rate limits:
- **FRED**: 120 requests per minute (free tier)
- **FINRA**: Varies by subscription level

If you hit rate limits, add delays between requests or use batch processing.

## Example Output

After running the fetch script, you'll get:

```
historical_data/
├── treasury_bonds_1980_1999.csv      # All bonds
├── train_bonds_1980_1999.csv         # Training set (70%)
└── eval_bonds_1980_1999.csv          # Evaluation set (30%)
```

Each CSV contains:
- bond_id, bond_type, face_value, coupon_rate
- maturity_date, issue_date, current_price
- credit_rating, issuer, frequency
- callable, convertible, time_to_maturity, years_since_issue

## Next Steps

1. Review the fetched data quality
2. Integrate with your training pipeline
3. Use for model evaluation and backtesting
4. Combine with synthetic data if needed

For more information, see:
- [FRED API Documentation](https://fred.stlouisfed.org/docs/api/)
- [FINRA API Documentation](https://developer.finra.org/docs)
- [Training Data Guide](./TRAINING_DATA.md)
