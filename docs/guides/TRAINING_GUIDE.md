# Training Models Guide

This guide explains how to train models with your historical bond data.

## Quick Start

### Step 1: Fetch Historical Data (if not done already)

First, fetch historical data from FRED and FINRA:

```bash
python scripts/fetch_historical_data.py --start-year 2010 --end-year 2020
```

This will create CSV files in `historical_data/`:
- `train_bonds_2010_2020.csv` - Training set
- `eval_bonds_2010_2020.csv` - Evaluation set
- `all_bonds_2010_2020.csv` - All bonds combined

### Step 2: Train Models with Historical Data

**Option A: Direct Training (Recommended)**

Train models directly from CSV files:

```bash
python scripts/train_with_historical_data.py
```

This will:
- Load bonds from `historical_data/train_bonds_2010_2020.csv`
- Train all ML models (Basic, Enhanced, Advanced, AutoML)
- Save models to `trained_models/`

**Option B: Create Dataset File First**

Create a dataset file compatible with the full training pipeline:

```bash
python scripts/train_with_historical_data.py --create-dataset
```

Then use the comprehensive training script:

```bash
python scripts/train_all_models.py --dataset-path training_data/historical_training_dataset.joblib
```

### Step 3: Train with Custom Data

If you have data in a different location:

```bash
# Use specific files
python scripts/train_with_historical_data.py \
    --train-file path/to/train.csv \
    --eval-file path/to/eval.csv \
    --model-dir my_models

# Or use a single file with all bonds
python scripts/train_with_historical_data.py \
    --all-file historical_data/all_bonds_2010_2020.csv
```

## Training Options

### Available Models

The training script trains these models:

1. **Basic ML Adjuster** - Random Forest model
2. **Enhanced ML Adjuster** - With hyperparameter tuning
3. **Advanced ML Adjuster** - Ensemble model (Random Forest + Gradient Boosting + Neural Network)
4. **AutoML** - Automated model selection

### Command Line Arguments

```bash
python scripts/train_with_historical_data.py [OPTIONS]

Options:
  --data-dir DIR          Directory with CSV files (default: historical_data)
  --train-file FILE       Training CSV file (default: train_bonds_2010_2020.csv)
  --eval-file FILE        Evaluation CSV file (default: eval_bonds_2010_2020.csv)
  --all-file FILE         Use all bonds from single file
  --model-dir DIR         Directory to save models (default: trained_models)
  --create-dataset        Create dataset file for train_all_models.py
```

## Output

After training, you'll have:

- **Trained Models** in `trained_models/`:
  - `ml_adjuster.joblib`
  - `enhanced_ml_adjuster.joblib`
  - `advanced_ml_adjuster.joblib`
  - `automl_adjuster.joblib`

- **Training Metrics** displayed in console:
  - R² scores (train/test)
  - RMSE (Root Mean Squared Error)
  - MAE (Mean Absolute Error)
  - Cross-validation scores

## Using Trained Models

Load and use trained models:

```python
import joblib
from bondtrader.core.bond_models import Bond, BondType
from datetime import datetime, timedelta

# Load model
model = joblib.load('trained_models/enhanced_ml_adjuster.joblib')

# Create a bond
bond = Bond(
    bond_id="TEST-001",
    bond_type=BondType.CORPORATE,
    face_value=1000,
    coupon_rate=5.0,
    maturity_date=datetime.now() + timedelta(days=1825),
    issue_date=datetime.now() - timedelta(days=365),
    current_price=950,
    credit_rating="BBB",
    issuer="Example Corp"
)

# Predict adjusted value
prediction = model.predict_adjusted_value(bond)
print(f"ML Adjusted Value: {prediction.get('ml_adjusted_value', 'N/A')}")
```

## Troubleshooting

### No Data Files Found

If you see "No CSV files found":
1. First run: `python scripts/fetch_historical_data.py --start-year 2010 --end-year 2020`
2. Or use synthetic data: `python scripts/train_all_models.py` (generates data automatically)

### API Key Errors

If you get FRED/FINRA API errors:
- Check your `.env` file has `FRED_API_KEY` set
- For FINRA, ensure both `FINRA_API_KEY` and `FINRA_API_PASSWORD` are set
- The script will use synthetic data if APIs are unavailable

### Memory Issues

If training fails due to memory:
- Reduce the number of bonds: Use `--all-file` with a smaller dataset
- Train models one at a time (modify script)
- Use a machine with more RAM

## Next Steps

After training:

1. **Evaluate Models**: Use `scripts/evaluate_models.py` to test performance
2. **Compare Models**: Check which model performs best on your data
3. **Deploy**: Use the best model in your trading system
4. **Monitor**: Set up drift detection to monitor model performance over time

For more information, see:
- `docs/guides/TRAINING_DATA.md` - Training data documentation
- `docs/guides/HISTORICAL_DATA_FETCHING.md` - API data fetching guide
