# Regenerate Evaluation Dataset

## Quick Start

To regenerate the evaluation dataset, use the EvaluationDatasetGenerator directly:

```python
from bondtrader.data.evaluation_dataset_generator import EvaluationDatasetGenerator, save_evaluation_dataset

generator = EvaluationDatasetGenerator(seed=42)
dataset = generator.generate_evaluation_dataset(
    num_bonds=1000, 
    scenarios=None,  # All scenarios
    include_benchmarks=True, 
    point_in_time=True
)
save_evaluation_dataset(dataset, "evaluation_data/evaluation_dataset.joblib")
```

## What This Does

1. **Regenerates the evaluation dataset** with all bonds properly included
2. **Fixes the missing 'bonds' issue** that was causing evaluation failures
3. **Verifies the dataset** to ensure all scenarios have bonds
4. **Provides backup option** to save existing dataset before overwriting

## Usage Examples

### Basic Usage (Default: 1000 bonds, all scenarios)
```bash
python regenerate_evaluation_dataset.py
```

### Generate with More Bonds
```bash
python regenerate_evaluation_dataset.py --num-bonds 2000
```

### Generate Specific Scenarios Only
```bash
python regenerate_evaluation_dataset.py --scenarios normal_market,rate_shock_up_200bps,credit_spread_widening
```

### Backup Existing Dataset First
```bash
python regenerate_evaluation_dataset.py --backup
```

### Combined Options
```bash
python regenerate_evaluation_dataset.py --num-bonds 2000 --scenarios normal_market,rate_shock_up_200bps --backup
```

## Available Scenarios

When using `--scenarios`, you can specify:
- `normal_market` - Normal Market Conditions
- `rate_shock_up_200bps` - Interest Rate Shock +200 bps
- `rate_shock_down_200bps` - Interest Rate Shock -200 bps
- `credit_spread_widening` - Credit Spread Widening +150 bps
- `liquidity_crisis` - Liquidity Crisis (2008-style)
- `market_crash` - Market Crash
- `low_volatility` - Low Volatility Regime
- `high_volatility` - High Volatility Regime
- `recovery` - Post-Crisis Recovery

## What Gets Fixed

### Before (Broken Dataset)
```python
scenario_data = {
    'scenario': {...},
    'actual_prices': [...],
    'fair_values': [...],
    # 'bonds': MISSING! ❌
}
```

### After (Fixed Dataset)
```python
scenario_data = {
    'scenario': {...},
    'bonds': [Bond, Bond, ...],  # ✅ Present!
    'actual_prices': [...],
    'fair_values': [...],
}
```

## Time Estimate

| Bonds | Estimated Time |
|-------|----------------|
| 500   | ~1-2 minutes   |
| 1000  | ~2-4 minutes   |
| 2000  | ~4-8 minutes   |
| 5000  | ~10-20 minutes |

*Times may vary based on your system and number of scenarios*

## Verification

After regeneration, the script automatically verifies:
- ✅ All scenarios have `'bonds'` key
- ✅ Bonds are non-empty lists
- ✅ Total bond count matches expectations
- ✅ `evaluation_bonds` is available

## Troubleshooting

### Dataset Already Exists
If a dataset exists, you'll be prompted to overwrite it. Use `--backup` to save the old one first.

### Not Enough Memory
If you get memory errors, reduce `--num-bonds` (e.g., 500 instead of 1000).

### Specific Scenarios Fail
If certain scenarios fail during generation, check the console output for specific errors.

## After Regeneration

Once regeneration is complete:

1. **Re-run evaluation**:
   ```bash
   python model_scoring_evaluator.py
   ```

2. **Verify results** - Models should now evaluate successfully without the `'bonds'` error.

## Need Help?

Run with `--help` for full usage information:
```bash
python regenerate_evaluation_dataset.py --help
```
