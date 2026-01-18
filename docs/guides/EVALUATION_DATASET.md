# Evaluation Dataset Generator Documentation

## Overview

The Evaluation Dataset Generator creates comprehensive, independent evaluation datasets for bond pricing models following **best practices from leading financial firms** including Bloomberg, BlackRock Aladdin, Goldman Sachs, and JPMorgan. This module is separate from training data generation and is designed specifically for **model validation, benchmarking, and performance assessment**.

## Key Features

### 1. Truly Out-of-Sample Data
- **Independent from training data**: Separate bond universe ensures no data leakage
- **Point-in-time data**: Prevents look-ahead bias (industry standard requirement)
- **Time-based validation**: Data reflects actual market conditions at specific dates

### 2. Multiple Evaluation Scenarios
The generator includes 9 comprehensive scenarios:

#### Normal Market Scenarios
- **Normal Market Conditions**: Baseline for standard evaluation
- **Post-Crisis Recovery**: Market recovery dynamics

#### Stress Scenarios (Regulatory Compliance)
- **Interest Rate Shock +200 bps**: Sudden rate increase (regulatory requirement)
- **Interest Rate Shock -200 bps**: Sudden rate decrease
- **Credit Spread Widening +150 bps**: Credit market stress
- **Liquidity Crisis**: Severe liquidity constraints (2008-style)

#### Sensitivity Scenarios
- **Low Volatility Regime**: Calm market conditions
- **High Volatility Regime**: Elevated volatility without crisis
- **Market Crash**: Extreme market dislocation

### 3. Built-in Benchmark Comparisons
Automatically generates benchmark prices from:
- **Bloomberg Terminal** (BVAL-style evaluated pricing)
- **BlackRock Aladdin** (risk-adjusted valuation)
- **Goldman Sachs** (credit research-enhanced pricing)
- **JPMorgan** (transaction cost-adjusted pricing)
- **Consensus** (weighted average of all benchmarks)

### 4. Comprehensive Evaluation Metrics
For each scenario, calculates:

#### Prediction Accuracy
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)
- R² Score
- Maximum Error

#### Financial Metrics
- Price Drift Score (normalized RMSE)
- Return Correlation
- Sharpe Ratio (simplified)

#### Benchmark Comparisons
- Drift vs. Bloomberg
- Drift vs. Aladdin
- Drift vs. Goldman
- Drift vs. JPMorgan
- Drift vs. Consensus

#### Distribution Metrics
- Prediction Bias (systematic error)
- Prediction Standard Deviation
- Actual Standard Deviation

#### Tail Risk Metrics
- Tail Loss Ratio
- Extreme Error Count

### 5. Data Quality Validation
- Missing value detection
- Infinite value checking
- Negative price validation
- Price range validation
- Distribution analysis

### 6. Audit Trail and Metadata
- Generation timestamp
- Point-in-time indicators
- Scenario documentation
- Data lineage tracking
- Compliance documentation

## Industry Best Practices Implemented

### ✅ Model Risk Management (MRM) Compliance
Follows Federal Reserve guidance on model validation:
- Independent evaluation datasets
- Stress testing requirements
- Benchmark comparisons
- Documentation and audit trails

### ✅ Point-in-Time Data (No Look-Ahead Bias)
- Data reflects information available at evaluation time
- Prevents future information leakage
- Matches Bloomberg/BlackRock data practices

### ✅ Multiple Market Regimes
- Ensures model robustness across conditions
- Regulatory stress testing requirements
- Sensitivity analysis for risk management

### ✅ Benchmark Comparison Framework
- Industry-standard benchmarks (Bloomberg, Aladdin, Goldman, JPMorgan)
- Consensus benchmarking for robustness
- Drift detection against industry standards

### ✅ Comprehensive Evaluation Metrics
- Beyond basic accuracy (RMSE, R²)
- Financial metrics (Sharpe ratio, correlation)
- Tail risk assessment
- Bias detection

## Usage

### Basic Usage: Generate Evaluation Dataset

```python
from evaluation_dataset_generator import EvaluationDatasetGenerator

# Initialize generator
generator = EvaluationDatasetGenerator(seed=42)

# Generate comprehensive evaluation dataset
evaluation_dataset = generator.generate_evaluation_dataset(
    num_bonds=2000,              # Number of bonds to evaluate
    scenarios=None,              # None = all scenarios
    include_benchmarks=True,     # Include benchmark comparisons
    point_in_time=True           # Use point-in-time data
)

# Save dataset
from evaluation_dataset_generator import save_evaluation_dataset
save_evaluation_dataset(evaluation_dataset, 'evaluation_data/evaluation_dataset.joblib')
```

### Evaluate a Model

```python
from evaluation_dataset_generator import EvaluationDatasetGenerator, load_evaluation_dataset
from ml_adjuster_enhanced import EnhancedMLBondAdjuster

# Load trained model
model = EnhancedMLBondAdjuster()
# ... train model on training data ...

# Load evaluation dataset
evaluation_dataset = load_evaluation_dataset('evaluation_data/evaluation_dataset.joblib')

# Initialize evaluation generator
generator = EvaluationDatasetGenerator()

# Evaluate model on all scenarios
evaluation_results = generator.evaluate_model(
    model=model,
    evaluation_dataset=evaluation_dataset,
    scenario_name=None  # None = evaluate all scenarios
)

# Access results
for scenario_name, metrics in evaluation_results.items():
    print(f"\n{scenario_name}:")
    print(f"  R² Score: {metrics.r2_score:.4f}")
    print(f"  RMSE: {metrics.root_mean_squared_error:.2f}")
    print(f"  Drift vs Consensus: {metrics.drift_vs_consensus.drift_score:.4f}")
    print(f"  Benchmark Correlation: {metrics.drift_vs_consensus.correlation:.4f}")
```

### Evaluate on Specific Scenario

```python
# Evaluate only on stress scenarios
evaluation_results = generator.evaluate_model(
    model=model,
    evaluation_dataset=evaluation_dataset,
    scenario_name='rate_shock_up_200bps'
)

metrics = evaluation_results['rate_shock_up_200bps']
print(f"Stress Test Performance:")
print(f"  RMSE: {metrics.root_mean_squared_error:.2f}")
print(f"  Max Error: {metrics.max_error:.2f}")
print(f"  Tail Loss Ratio: {metrics.tail_loss_ratio:.2f}")
```

### Custom Scenario Selection

```python
# Generate dataset with specific scenarios only
evaluation_dataset = generator.generate_evaluation_dataset(
    num_bonds=1000,
    scenarios=[
        'normal_market',
        'rate_shock_up_200bps',
        'liquidity_crisis',
        'market_crash'
    ],
    include_benchmarks=True
)
```

## Dataset Structure

```python
evaluation_dataset = {
    'scenarios': {
        'normal_market': {
            'scenario': EvaluationScenario(...),
            'bonds': List[Bond],
            'actual_prices': np.ndarray,
            'fair_values': np.ndarray,
            'benchmark_prices': {
                'bloomberg': List[float],
                'aladdin': List[float],
                'goldman': List[float],
                'jpmorgan': List[float],
                'consensus': List[float]
            },
            'num_bonds': int,
            'date_range': (datetime, datetime),
            'point_in_time': bool
        },
        # ... other scenarios ...
    },
    'benchmarks': {
        'bloomberg': List[float],
        'aladdin': List[float],
        # ... other benchmarks ...
    },
    'quality_report': {
        'scenario_name': {
            'num_bonds': int,
            'missing_prices': int,
            'price_range': (float, float),
            # ... other quality metrics ...
        },
        # ... other scenarios ...
    },
    'metadata': {
        'generation_timestamp': str,
        'point_in_time': bool,
        'scenarios_included': List[str],
        'evaluation_standards': List[str],
        # ... other metadata ...
    },
    'summary_statistics': {
        'total_scenarios': int,
        'total_bonds': int,
        'scenario_summary': Dict
    },
    'evaluation_bonds': List[Bond]
}
```

## Evaluation Metrics Structure

```python
EvaluationMetrics(
    # Prediction accuracy
    mean_absolute_error=float,
    root_mean_squared_error=float,
    mean_absolute_percentage_error=float,
    r2_score=float,
    max_error=float,
    
    # Financial metrics
    price_drift_score=float,
    return_correlation=float,
    sharpe_ratio=float,
    
    # Benchmark comparisons (DriftMetrics objects)
    drift_vs_bloomberg=DriftMetrics(...),
    drift_vs_aladdin=DriftMetrics(...),
    drift_vs_goldman=DriftMetrics(...),
    drift_vs_jpmorgan=DriftMetrics(...),
    drift_vs_consensus=DriftMetrics(...),
    
    # Distribution metrics
    prediction_bias=float,
    prediction_std=float,
    actual_std=float,
    
    # Tail risk
    tail_loss_ratio=float,
    extreme_error_count=int
)
```

## Scenario Details

### Normal Market Conditions
- **Purpose**: Baseline evaluation
- **Parameters**: Standard market conditions
- **Use Case**: General model performance assessment

### Interest Rate Shock +200 bps
- **Purpose**: Regulatory stress test
- **Parameters**: Risk-free rate +200 bps, elevated volatility
- **Use Case**: Model sensitivity to rate changes

### Interest Rate Shock -200 bps
- **Purpose**: Regulatory stress test (downside)
- **Parameters**: Risk-free rate -200 bps, moderate volatility
- **Use Case**: Model behavior in falling rate environment

### Credit Spread Widening
- **Purpose**: Credit market stress
- **Parameters**: Credit spreads +150 bps, high volatility
- **Use Case**: Credit risk model validation

### Liquidity Crisis
- **Purpose**: Extreme liquidity stress (2008-style)
- **Parameters**: Severe liquidity constraints, wide spreads
- **Use Case**: Liquidity risk model validation

### Market Crash
- **Purpose**: Extreme market dislocation
- **Parameters**: Maximum volatility, wide spreads, low liquidity
- **Use Case**: Worst-case scenario validation

### Low/High Volatility Regimes
- **Purpose**: Sensitivity analysis
- **Parameters**: Low/high volatility multipliers
- **Use Case**: Volatility sensitivity testing

## Best Practices for Model Evaluation

### 1. Use Separate Evaluation Dataset
- **Never use training data** for final evaluation
- Generate independent evaluation dataset
- Keep evaluation dataset separate from training pipeline

### 2. Evaluate Across All Scenarios
- Don't just evaluate on normal conditions
- Test on stress and crisis scenarios
- Identify model weaknesses in extreme conditions

### 3. Benchmark Comparison
- Always compare against industry benchmarks
- Monitor drift vs. Bloomberg, Aladdin, etc.
- Aim for low drift scores (<0.05 is excellent)

### 4. Monitor Multiple Metrics
- Don't just look at R² or RMSE
- Check bias (systematic errors)
- Assess tail risk (extreme errors)
- Evaluate correlation with benchmarks

### 5. Document Evaluation Results
- Save evaluation results with model versions
- Track performance over time
- Document assumptions and limitations

### 6. Regular Re-evaluation
- Re-evaluate models periodically
- Monitor for performance degradation
- Update evaluation dataset as market evolves

## Integration with Model Training Pipeline

```python
# 1. Train model on training data
from train_all_models import ModelTrainer
trainer = ModelTrainer(dataset_path='training_data/training_dataset.joblib')
training_results = trainer.train_all_models()

# 2. Generate evaluation dataset (separate from training)
from evaluation_dataset_generator import EvaluationDatasetGenerator
eval_generator = EvaluationDatasetGenerator()
evaluation_dataset = eval_generator.generate_evaluation_dataset(num_bonds=2000)

# 3. Evaluate trained models
for model_name, model_data in training_results.items():
    if model_data.get('status') == 'success' and 'model' in model_data:
        model = model_data['model']
        eval_results = eval_generator.evaluate_model(model, evaluation_dataset)
        
        # 4. Compare performance
        print(f"\n{model_name} Evaluation:")
        for scenario, metrics in eval_results.items():
            print(f"  {scenario}: R²={metrics.r2_score:.4f}, "
                  f"Drift={metrics.drift_vs_consensus.drift_score:.4f}")
```

## Regulatory Compliance

This evaluation framework supports regulatory compliance:

- **Model Risk Management (MRM)**: Independent validation, stress testing
- **CCAR/DFAST**: Stress scenario testing
- **Basel III**: Risk model validation
- **Volcker Rule**: Model performance monitoring

## Performance Benchmarks

Expected performance ranges (industry standards):

- **Normal Market**: R² > 0.85, Drift < 0.03
- **Stress Scenarios**: R² > 0.75, Drift < 0.05
- **Crisis Scenarios**: R² > 0.65, Drift < 0.10
- **Benchmark Correlation**: > 0.90 (excellent)

## References

This implementation follows best practices from:

- **Bloomberg**: Point-in-time data, evaluated pricing (BVAL)
- **BlackRock Aladdin**: Risk-adjusted evaluation, regime analysis
- **Goldman Sachs**: Credit research integration, stress testing
- **JPMorgan**: Transaction cost realism, execution scenarios
- **Federal Reserve**: Model Risk Management guidance
- **Basel Committee**: Risk model validation standards

## File Structure

```
BondTrader/
├── evaluation_dataset_generator.py    # Main evaluation generator
├── evaluation_data/
│   └── evaluation_dataset.joblib     # Saved evaluation dataset
└── EVALUATION_DATASET_README.md      # This documentation
```

## License

Part of the BondTrader project.
