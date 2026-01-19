# Artifact Management

This document describes how to manage binary artifacts and regenerate models in the BondTrader codebase.

## Overview

The codebase generates various binary artifacts during training and evaluation:
- **Trained models** (`.joblib`, `.pkl` files)
- **Training datasets** (`.joblib` files)
- **Evaluation datasets** (`.joblib` files)
- **Evaluation results** (`.joblib` files)
- **Checkpoints** (`.joblib` files)
- **Database files** (`.db` files)
- **Coverage reports** (`.xml` files)
- **MLflow tracking data** (`mlruns/` directory)

These artifacts are **not** committed to git (see `.gitignore`) and can be regenerated from source code.

## Quick Start

### Clear All Artifacts

```bash
# Dry run (see what would be deleted)
make clear-artifacts

# Actually delete artifacts
make clear-artifacts-force

# Or use the script directly
python scripts/clear_artifacts.py --dry-run
python scripts/clear_artifacts.py
```

### Regenerate Models

```bash
# Full refresh (datasets + models)
make refresh-models

# Or use the script directly
python scripts/refresh_models.py
```

## Commands

### Clear Artifacts

#### Using Makefile

```bash
# Dry run (recommended first)
make clear-artifacts

# Actually clear artifacts
make clear-artifacts-force
```

#### Using Script

```bash
# Dry run
python scripts/clear_artifacts.py --dry-run

# Actually clear
python scripts/clear_artifacts.py

# Verbose output
python scripts/clear_artifacts.py --verbose
```

**What gets cleared:**
- All `.joblib`, `.pkl`, `.h5`, `.hdf5`, `.pb`, `.onnx` files
- All `.db` database files
- `trained_models/` directory
- `training_data/` directory
- `evaluation_data/` directory
- `evaluation_results/` directory
- `training_checkpoints/` directory
- `mlruns/` directory (MLflow tracking)
- `htmlcov/` directory (coverage reports)
- Cache directories (`.pytest_cache`, `.mypy_cache`, `.hypothesis`)
- Coverage files (`coverage.xml`, `.coverage`)

### Refresh Models

#### Using Makefile

```bash
# Full refresh (datasets + models)
make refresh-models

# Generate datasets only
make refresh-datasets

# Train models only (use existing datasets)
make refresh-models-only
```

#### Using Script

```bash
# Full refresh
python scripts/refresh_models.py

# Only generate datasets
python scripts/refresh_models.py --datasets-only

# Only train models (use existing datasets)
python scripts/refresh_models.py --skip-training-data --skip-evaluation-data

# Use specific dataset
python scripts/refresh_models.py --dataset training_data/my_dataset.joblib
```

**What gets generated:**
1. **Training Dataset** - Fresh training data with configurable parameters
2. **Evaluation Dataset** - Fresh evaluation data with all scenarios
3. **All ML Models** - Trained from scratch:
   - `MLBondAdjuster`
   - `EnhancedMLBondAdjuster`
   - `AdvancedMLBondAdjuster`
   - `AutoMLBondAdjuster`
   - `RegimeDetector`
   - `TailRiskAnalyzer`

## Configuration

Model and dataset generation uses settings from `bondtrader/config.py`:

```python
# Training settings
training_batch_size: int = 100
training_num_bonds: int = 5000
training_time_periods: int = 60

# Paths
model_dir: str = "trained_models"
data_dir: str = "training_data"
evaluation_data_dir: str = "evaluation_data"
evaluation_results_dir: str = "evaluation_results"
checkpoint_dir: str = "training_checkpoints"
```

You can override these via environment variables (see `env.example`).

## Workflow Examples

### Clean Slate (Clear + Refresh)

```bash
# 1. Clear all artifacts
make clear-artifacts-force

# 2. Regenerate everything
make refresh-models
```

### Update Models Only

```bash
# Use existing datasets, retrain models
make refresh-models-only
```

### Generate Fresh Datasets

```bash
# Generate new datasets without retraining
make refresh-datasets
```

### Development Workflow

```bash
# Clear artifacts before committing
make clear-artifacts-force

# Regenerate models for testing
make refresh-models
```

## File Locations

### Artifacts (Cleared)

```
trained_models/          # Trained ML models
training_data/           # Training datasets
evaluation_data/         # Evaluation datasets
evaluation_results/      # Evaluation results
training_checkpoints/    # Training checkpoints
mlruns/                  # MLflow tracking data
*.db                     # Database files
coverage.xml             # Coverage reports
```

### Source Code (Preserved)

```
bondtrader/              # Source code
scripts/                 # Scripts
tests/                   # Tests
docs/                    # Documentation
```

## Troubleshooting

### Permission Errors

If you get permission errors when clearing artifacts:

```bash
# On Unix/Mac
sudo python scripts/clear_artifacts.py

# Or fix permissions
chmod -R u+w trained_models/ training_data/ evaluation_data/
```

### Out of Memory

If model training fails due to memory:

1. Reduce `training_num_bonds` in config
2. Reduce `training_time_periods` in config
3. Use smaller batch sizes

### Models Not Found

If you get "model not found" errors:

```bash
# Regenerate models
make refresh-models
```

## Best Practices

1. **Before Committing**: Clear artifacts to keep repo clean
   ```bash
   make clear-artifacts-force
   ```

2. **After Cloning**: Regenerate models for local use
   ```bash
   make refresh-models
   ```

3. **CI/CD**: Always regenerate models in CI, don't commit them

4. **Development**: Use `--skip-training-data` to speed up iteration

5. **Production**: Use versioned model storage (S3, MLflow, etc.)

## Integration with CI/CD

Example GitHub Actions workflow:

```yaml
- name: Clear artifacts
  run: make clear-artifacts-force

- name: Train models
  run: make refresh-models

- name: Run tests
  run: pytest tests/
```

## See Also

- `scripts/clear_artifacts.py` - Artifact clearing script
- `scripts/refresh_models.py` - Model regeneration script
- `bondtrader/config.py` - Configuration settings
- `.gitignore` - Ignored files/directories
