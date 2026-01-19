# Data Leakage Fix - ML Bond Adjuster

## Issue Identified

The machine learning models in BondTrader had a **critical data leakage problem** where the target variable was included as a feature, making the model trivially accurate but not learning meaningful patterns.

### The Problem

- **Feature**: `price_to_fair_ratio = bond.current_price / fair_value`
- **Target**: `adjustment_factor = bond.current_price / fair_value`

These are **identical**, meaning the model could simply return the feature value as the prediction without learning anything useful.

## Files Fixed

1. **`bondtrader/ml/ml_adjuster.py`**
   - Removed `price_to_fair_ratio` from `_create_features()` method
   - Added explanatory comment about why it was removed

2. **`bondtrader/ml/ml_adjuster_enhanced.py`**
   - Removed `price_to_fair_ratio` from `_create_enhanced_features()` method
   - Updated `feature_names` list to remove the feature
   - Added explanatory comments

3. **`bondtrader/ml/ml_advanced.py`**
   - Removed `price_to_fair_ratio` from `_create_advanced_features()` method
   - Updated `base_names` list to remove the feature
   - Added explanatory comments

4. **`bondtrader/data/training_data_generator.py`**
   - Removed `market_price / fair_value` from feature vector in `_create_feature_matrices()` method
   - Added explanatory comment

## What Remains (Intentionally)

The following features are **still included** and are **correct**:
- `price_to_par_ratio = current_price / face_value` - This is different from the target and is a valid bond characteristic
- All other bond characteristics (coupon rate, maturity, credit rating, YTM, duration, convexity, etc.)

## Impact

### Before Fix
- Model could achieve artificially high R² scores by simply returning the input feature
- Model wasn't learning meaningful relationships between bond characteristics and market adjustments
- Performance metrics were misleading

### After Fix
- Model must learn genuine patterns from bond characteristics
- Performance metrics will be more realistic and meaningful
- Model will be more robust and generalizable
- **Note**: Model performance metrics (R², RMSE) may decrease initially, but this reflects true predictive capability

## Model Retraining Required

⚠️ **Important**: Any models trained before this fix should be **retrained** because:
1. The feature count has changed (one feature removed)
2. Old models were trained with data leakage and won't work correctly with the new feature set
3. The scaler was fit on a different number of features

### Retraining Steps

```python
from bondtrader.ml import MLBondAdjuster
from bondtrader.data import TrainingDataGenerator

# Generate fresh training data
generator = TrainingDataGenerator(seed=42)
bonds = generator.generate_bonds_for_training(num_bonds=1000)

# Train new model
ml_adjuster = MLBondAdjuster(model_type="random_forest")
metrics = ml_adjuster.train(bonds)

# Save the retrained model
ml_adjuster.save_model("models/ml_adjuster_fixed.joblib")
```

## Validation

The fix has been validated:
- ✅ No linter errors
- ✅ Feature count is consistent across training and prediction
- ✅ Prediction logic remains correct (doesn't require market price during prediction)
- ✅ All comments explain the reasoning

## Technical Details

### Feature Engineering Philosophy

The model now learns adjustments based on:
- **Bond characteristics**: coupon rate, maturity, credit rating, face value
- **Derived metrics**: YTM, duration, convexity, modified duration, spread over risk-free rate
- **Market regime indicators**: (in enhanced/advanced versions)
- **Time features**: (in enhanced/advanced versions)

The model **does not** use:
- ❌ `price_to_fair_ratio` (data leakage - same as target)
- ✅ `price_to_par_ratio` (valid - different from target, represents bond trading level)

### Use Case

The ML model is designed to:
1. Take bond characteristics and theoretical fair value as input
2. Predict an adjustment factor based on learned market patterns
3. Return: `ml_adjusted_fair_value = theoretical_fair_value × adjustment_factor`

This allows the model to capture market inefficiencies and factors not captured by theoretical DCF models.
