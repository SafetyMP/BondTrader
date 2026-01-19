# API-Based Model Training and Prediction

This guide explains how to train and evaluate models using API data from FRED and FINRA, then display results in Streamlit.

## Overview

The workflow consists of:
1. **Training** (2016-2017): Fetch bond data from APIs and train multiple ML models
2. **Fine-tuning** (2018): Fine-tune the trained models on 2018 data
3. **Prediction** (2025): Make predictions on 2025 bond data
4. **Visualization**: Display results in an interactive Streamlit dashboard

## Prerequisites

1. **API Keys** (optional but recommended):
   - FRED API Key: Get from https://fred.stlouisfed.org/docs/api/api_key.html
   - FINRA API Credentials: Get from FINRA (if available)
   
   Add to your `.env` file:
   ```
   FRED_API_KEY=your_fred_api_key
   FINRA_API_KEY=your_finra_api_key
   FINRA_API_PASSWORD=your_finra_password
   ```

2. **Dependencies**: All required packages should be in `requirements.txt`

## Usage

### Step 1: Train and Evaluate Models

Run the main training script:

```bash
python scripts/train_evaluate_with_api_data.py
```

This script will:
- Fetch Treasury bond data from FRED API for 2016-2017 (training)
- Train 4 different ML models:
  - Basic ML Adjuster
  - Enhanced ML Adjuster (with hyperparameter tuning)
  - Advanced ML Adjuster (ensemble)
  - AutoML (automated model selection)
- Fetch 2018 data and fine-tune all models
- Fetch 2025 data and make predictions
- Save models to `trained_models/api_trained_models/`
- Save predictions to `training_data/predictions/2025_predictions.csv`

**Note**: If API keys are not available, the script will generate synthetic data automatically.

### Step 2: View Results in Streamlit

Launch the Streamlit dashboard:

```bash
streamlit run scripts/streamlit_predictions_dashboard.py
```

The dashboard will display:
- Summary statistics
- Model performance metrics (MAE, RMSE, MAPE, Correlation)
- Predictions comparison charts
- Price distribution visualizations
- Detailed predictions table with filtering options

## Output Files

### Models
- `trained_models/api_trained_models/ml_adjuster.joblib`
- `trained_models/api_trained_models/enhanced_ml_adjuster.joblib`
- `trained_models/api_trained_models/advanced_ml_adjuster.joblib`
- `trained_models/api_trained_models/automl_adjuster.joblib`
- Fine-tuned versions: `*_fine_tuned.joblib`

### Predictions
- `training_data/predictions/2025_predictions.csv`

## Features

### Training Script Features
- **API Integration**: Uses FRED for Treasury data, FINRA for corporate bonds
- **Fallback**: Generates synthetic data if APIs are unavailable
- **Multiple Models**: Trains 4 different ML approaches
- **Fine-tuning**: Continues training on 2018 data
- **Comprehensive Predictions**: All models predict on 2025 data

### Dashboard Features
- **Interactive Filters**: Filter by bond type, credit rating, issuer
- **Performance Metrics**: Compare model accuracy
- **Visualizations**: Charts for predictions comparison and distributions
- **Export**: Download predictions as CSV

## Troubleshooting

### API Key Issues
If you see warnings about missing API keys, the script will automatically generate synthetic data. This is fine for testing, but for production use, obtain real API keys.

### No Data Retrieved
If APIs return no data:
- Check your API keys are correct
- Verify network connectivity
- Check date ranges (FINRA data only available from 2002+)
- The script will fall back to synthetic data generation

### Model Training Errors
If a model fails to train:
- Check that you have at least 10 bonds in the dataset
- Verify all dependencies are installed: `pip install -r requirements.txt`
- Check logs for specific error messages

## Customization

### Change Date Ranges
Edit the date ranges in `train_evaluate_with_api_data.py`:

```python
train_start = datetime(2016, 1, 1)
train_end = datetime(2017, 12, 31)
fine_tune_start = datetime(2018, 1, 1)
fine_tune_end = datetime(2018, 12, 31)
predict_start = datetime(2025, 1, 1)
predict_end = datetime(2025, 12, 31)
```

### Add More Models
Add additional models in the `train_models()` function following the existing pattern.

### Customize Dashboard
Edit `streamlit_predictions_dashboard.py` to add more visualizations or metrics.

## API Data Sources

### FRED (Federal Reserve Economic Data)
- **Treasury Yields**: 1, 2, 5, 10, 30 year maturities
- **Free API**: Requires registration for API key
- **Rate Limits**: Generous for non-commercial use

### FINRA (Financial Industry Regulatory Authority)
- **Corporate Bonds**: Transaction data from TRACE
- **Available**: 2002 onwards
- **Access**: May require special permissions/credentials

## Next Steps

1. **Evaluate Model Performance**: Review metrics in the dashboard
2. **Compare Models**: Use the comparison charts to select best model
3. **Production Deployment**: Use the best model for live predictions
4. **Continuous Training**: Retrain periodically with new data
