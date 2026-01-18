# User Guide

Complete guide to using BondTrader.

## Getting Started

### Installation

See the [Main README](../../README.md) for installation instructions.

### Quick Start

1. Install dependencies
2. Run the dashboard: `streamlit run scripts/dashboard.py`
3. Explore bond valuations and arbitrage opportunities

## Dashboard Guide

### Overview Tab

- Market summary statistics
- Bond distribution charts
- Price vs fair value analysis

### Arbitrage Opportunities Tab

- List of mispriced bonds
- Profit potential analysis
- Filtering and sorting options

### Bond Comparison Tab

- Compare bonds by type, rating, or maturity
- Side-by-side analysis
- Statistical comparisons

### Bond Details Tab

- Individual bond analysis
- All valuation metrics
- Risk metrics
- ML predictions

### Portfolio Analysis Tab

- Portfolio composition
- Portfolio-level arbitrage
- Risk analysis
- Optimization suggestions

## Python API Usage

### Basic Valuation

```python
from bondtrader.core import Bond, BondType, BondValuator
from datetime import datetime, timedelta

# Create bond
bond = Bond(...)

# Value bond
valuator = BondValuator(risk_free_rate=0.03)
fair_value = valuator.calculate_fair_value(bond)
```

### ML-Enhanced Valuation

```python
from bondtrader.ml import MLBondAdjuster

# Train ML model
ml_adjuster = MLBondAdjuster()
ml_adjuster.train(bonds)

# Get ML-adjusted valuation
result = ml_adjuster.predict_adjusted_value(bond)
```

### Arbitrage Detection

```python
from bondtrader.core import ArbitrageDetector

detector = ArbitrageDetector(valuator=valuator)
opportunities = detector.find_arbitrage_opportunities(bonds)
```

## Configuration

See [Configuration Guide](../README.md#configuration) for details.

## Training Models

See [Training Data Guide](TRAINING_DATA.md) for model training.

## Advanced Features

- Portfolio Optimization
- Risk Management
- Backtesting
- Factor Models

See individual guides for detailed information.
