# Sample Bond Dataset

This directory contains sample bond datasets generated for use with the Streamlit dashboard.

## Files

- `sample_bonds.csv` - Sample bond dataset with 200 bonds containing various bond types, credit ratings, and characteristics

## Dataset Structure

The CSV file contains the following columns:

- `bond_id`: Unique identifier for each bond
- `bond_type`: Type of bond (Fixed Rate, Corporate, Treasury, Zero Coupon, High Yield)
- `issuer`: Issuer name
- `credit_rating`: Credit rating (AAA, AA, A, BBB, BB, B, CCC)
- `face_value`: Face/par value of the bond
- `coupon_rate`: Annual coupon rate (percentage)
- `maturity_date`: Bond maturity date (YYYY-MM-DD)
- `issue_date`: Bond issue date (YYYY-MM-DD)
- `current_price`: Current market price
- `time_to_maturity`: Time to maturity in years
- `years_since_issue`: Years since the bond was issued
- `frequency`: Coupon payment frequency (typically 2 for semi-annual)
- `callable`: Whether the bond is callable (True/False)
- `convertible`: Whether the bond is convertible (True/False)

## Generating New Sample Data

To generate a new sample dataset with custom parameters:

```bash
python3 scripts/generate_sample_data.py --num-bonds 300 --output-dir sample_data --filename sample_bonds.csv --seed 42
```

### Parameters

- `--num-bonds`: Number of bonds to generate (default: 200)
- `--output-dir`: Output directory (default: sample_data)
- `--filename`: Output filename (default: sample_bonds.csv)
- `--seed`: Random seed for reproducibility (default: 42)

## Usage with Streamlit Dashboard

The dashboard currently generates bonds on-the-fly using `BondDataGenerator`. To use this pre-generated dataset, you would need to modify the dashboard to load from CSV instead. However, the dashboard's current implementation works well with the generator for dynamic data.

This dataset is useful for:
- Testing and development
- Static analysis
- Documentation and demonstrations
- Importing into other tools

## Statistics

The current dataset (200 bonds) contains:
- Multiple bond types with realistic distributions
- Various credit ratings
- Diverse maturity dates and time to maturity
- Realistic pricing ranges (85% to 115% of face value)
- Mix of callable and convertible bonds
