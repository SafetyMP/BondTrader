# Evaluation Failure Analysis

## Root Cause

The evaluation failures were caused by a **missing `'bonds'` key** in the evaluation dataset when it was loaded from disk.

### The Problem

1. **Error**: `"'bonds'"` - A `KeyError` when trying to access `scenario_data['bonds']` during evaluation
2. **Location**: Line 616 in `evaluation_dataset_generator.py` - `bonds = scenario_data['bonds']`
3. **Why it failed**: The `save_evaluation_dataset()` function was **not saving the `'bonds'` key** when serializing the dataset

### The Bug

In `save_evaluation_dataset()` (lines 949-957), when converting the dataset to a serializable format, the function saved:
- ✅ `'scenario'`
- ✅ `'actual_prices'`
- ✅ `'fair_values'`
- ✅ `'benchmark_prices'`
- ✅ `'num_bonds'`
- ✅ `'date_range'`
- ✅ `'point_in_time'`
- ❌ **`'bonds'` - MISSING!**

This meant when the dataset was loaded back, the `'bonds'` key was missing from each scenario, causing evaluation to fail immediately.

### Dataset Structure (Before Fix)

**When Generated** (in memory):
```python
scenario_data = {
    'scenario': EvaluationScenario(...),
    'bonds': [Bond, Bond, ...],  # ✅ Present
    'actual_prices': [...],
    'fair_values': [...],
    ...
}
```

**When Saved to Disk**:
```python
scenario_data = {
    'scenario': {...},
    # 'bonds': MISSING! ❌
    'actual_prices': [...],
    'fair_values': [...],
    ...
}
```

**When Loaded from Disk**:
```python
scenario_data = {
    'scenario': {...},
    # 'bonds': STILL MISSING! ❌
    'actual_prices': [...],
    ...
}
```

### The Fix

1. **Updated `save_evaluation_dataset()`** to include `'bonds'` in the serialized data:
   ```python
   'bonds': sc_data.get('bonds', []),  # Now included!
   ```

2. **Bond restoration code** (already present in `generate_or_load_evaluation_dataset()`) will handle existing datasets that are missing bonds by:
   - Using `evaluation_bonds` from the top-level dataset (if available)
   - Recreating bonds from `actual_prices` and bond metadata
   - Falling back to regenerating the dataset if needed

### Why This Happened

- The save function was designed to only save "serializable" data
- Bond objects are complex Python objects that can be serialized with joblib, but the developer thought they needed special handling
- The bonds were included in the `'evaluation_bonds'` top-level key, but not in individual scenario data
- The evaluation code expects `scenario_data['bonds']` to be present

### Impact

- **All 4 models failed evaluation** with the same error: `"'bonds'"`
- **Evaluation time**: ~0.03 seconds (failed immediately on first scenario)
- **No results were generated** - all models marked as `status: 'failed'`

### Prevention

1. **Fixed `save_evaluation_dataset()`** to include bonds
2. **Added bond restoration** in `generate_or_load_evaluation_dataset()` to handle legacy datasets
3. **Better error handling** now prints full tracebacks for debugging

### Testing

To verify the fix:
```python
# Generate a new dataset (will include bonds)
evaluator = ModelEvaluator()
dataset = evaluator.generate_or_load_evaluation_dataset(generate_new=True)

# Check that bonds are present
for scenario_name, scenario_data in dataset['scenarios'].items():
    if scenario_name != 'benchmarks':
        assert 'bonds' in scenario_data, f"Missing bonds in {scenario_name}"
        assert len(scenario_data['bonds']) > 0, f"Empty bonds in {scenario_name}"
        print(f"✓ {scenario_name}: {len(scenario_data['bonds'])} bonds")
```

### Next Steps

1. **Regenerate the evaluation dataset** to fix existing broken datasets:
   ```python
   evaluator.generate_or_load_evaluation_dataset(generate_new=True)
   ```

2. **Or rely on restoration code** - The restoration code in `generate_or_load_evaluation_dataset()` should automatically fix existing datasets when they're loaded.

3. **Re-run evaluation** - Models should now evaluate successfully.
