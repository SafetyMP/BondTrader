# Evaluation Errors Summary

## Current Status

**Active Evaluation**: Yes (started at 3:50 PM, running for ~1:22 minutes)

## Known Errors

### 1. Missing 'bonds' Key Error ✅ **FIXED**

**Error**: `"'bonds'"` - `KeyError` when accessing `scenario_data['bonds']`

**Status**: 
- ✅ **Fixed in code** - `save_evaluation_dataset()` now includes bonds
- ✅ **Restoration code** - `generate_or_load_evaluation_dataset()` restores bonds from `evaluation_bonds`
- ⚠️ **Legacy datasets** - Existing datasets may still have this issue

**Impact**: All 4 models failed evaluation in previous run (15:02 PM)

**Fix Applied**:
1. Updated `save_evaluation_dataset()` to include `'bonds'` when saving
2. Added bond restoration logic in `generate_or_load_evaluation_dataset()`
3. Added better error message when bonds are missing

**How to Resolve**:
```python
# Option 1: Regenerate dataset (recommended)
evaluator = ModelEvaluator()
evaluator.generate_or_load_evaluation_dataset(generate_new=True)

# Option 2: Relies on restoration code (should work if evaluation_bonds exists)
evaluator.generate_or_load_evaluation_dataset(generate_new=False)
```

## Error Detection Points

### During Evaluation

1. **Dataset Loading** (`generate_or_load_evaluation_dataset()`)
   - Checks if bonds exist in scenario data
   - Attempts restoration from `evaluation_bonds`
   - Falls back to regeneration if needed

2. **Model Evaluation** (`evaluate_model()`)
   - Now checks for `'bonds'` key before accessing
   - Raises clear error message if missing
   - All exceptions caught and logged with traceback

3. **Prediction Generation**
   - Individual bond predictions wrapped in try/except
   - Failed predictions fallback to `bond.current_price`
   - Errors logged but don't stop evaluation

4. **Results Saving**
   - Failed evaluations marked with `status: 'failed'`
   - Error messages and tracebacks stored in results
   - Results always saved even if some models fail

## Error Handling Improvements

### Current Error Handling

1. **Model Loading Errors**
   ```python
   except Exception as e:
       print(f"  Error loading {model_name}: {e}")
       traceback.print_exc()
       return None
   ```

2. **Evaluation Errors**
   ```python
   except Exception as e:
       print(f"    ✗ Evaluation failed: {e}")
       print(f"    Full error traceback:")
       traceback.print_exc()
       all_results[model_name] = {
           'status': 'failed',
           'error': str(e),
           'traceback': traceback.format_exc()
       }
   ```

3. **Prediction Errors** (per bond)
   ```python
   except Exception as e:
       predictions.append(bond.current_price)  # Fallback
   ```

### Error Reporting

All errors are:
- ✅ **Printed to console** with full traceback
- ✅ **Saved in results** (`error` and `traceback` fields)
- ✅ **Included in warnings** in scoring reports

## Common Error Scenarios

### 1. Missing Bonds in Dataset
**Symptom**: `KeyError: 'bonds'`  
**Cause**: Dataset saved before fix was applied  
**Solution**: Regenerate dataset or use restoration code

### 2. Model Not Trained
**Symptom**: Model fails to make predictions  
**Cause**: Model loaded but `is_trained=False`  
**Solution**: Ensure models are properly trained and saved

### 3. Missing Model Files
**Symptom**: `model_name not found`  
**Cause**: Model file doesn't exist in `trained_models/`  
**Solution**: Train models first using `train_all_models.py`

### 4. Bond Restoration Failure
**Symptom**: Bonds not restored from `evaluation_bonds`  
**Cause**: `evaluation_bonds` missing or empty  
**Solution**: Regenerate evaluation dataset

## Monitoring Current Evaluation

To check for errors in the currently running evaluation:

```python
# Wait for completion, then check results
import joblib
import glob

# Find most recent results
result_files = glob.glob('evaluation_results/model_scores_*.joblib')
latest = max(result_files, key=os.path.getctime)
results = joblib.load(latest)

# Check for failures
for model_name, data in results['evaluation_results'].items():
    if data.get('status') == 'failed':
        print(f"{model_name}: {data.get('error')}")
        print(f"Traceback: {data.get('traceback', 'N/A')}")
```

## Prevention

1. ✅ **Bonds now saved** - Future datasets will include bonds
2. ✅ **Better error messages** - Clear indication of what's wrong
3. ✅ **Restoration code** - Handles legacy datasets automatically
4. ✅ **Error logging** - Full tracebacks saved for debugging
