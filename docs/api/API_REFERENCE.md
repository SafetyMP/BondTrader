# API Reference

Complete API documentation for BondTrader.

> **Note**: This is a placeholder. Full API documentation will be generated using Sphinx.

## Core Modules

### `bondtrader.core`

#### `Bond`

Bond data model.

```python
from bondtrader.core import Bond, BondType

bond = Bond(
    bond_id="BOND-001",
    bond_type=BondType.CORPORATE,
    face_value=1000,
    coupon_rate=5.0,
    maturity_date=datetime.now() + timedelta(days=1825),
    issue_date=datetime.now() - timedelta(days=365),
    current_price=950,
    credit_rating="BBB",
    issuer="Example Corp",
    frequency=2
)
```

#### `BondValuator`

Core bond valuation engine.

```python
from bondtrader.core import BondValuator

valuator = BondValuator(risk_free_rate=0.03)
fair_value = valuator.calculate_fair_value(bond)
ytm = valuator.calculate_yield_to_maturity(bond)
duration = valuator.calculate_duration(bond)
```

#### `ArbitrageDetector`

Detects arbitrage opportunities.

```python
from bondtrader.core import ArbitrageDetector

detector = ArbitrageDetector(valuator=valuator)
opportunities = detector.find_arbitrage_opportunities([bond])
```

## Machine Learning Modules

### `bondtrader.ml`

#### `MLBondAdjuster`

Basic machine learning price adjuster.

```python
from bondtrader.ml import MLBondAdjuster

ml_adjuster = MLBondAdjuster(model_type='random_forest')
ml_adjuster.train(bonds, test_size=0.2)
result = ml_adjuster.predict_adjusted_value(bond)
```

## Configuration

### `bondtrader.config`

#### `Config`

Configuration management.

```python
from bondtrader.config import get_config, Config

config = get_config()
custom_config = Config(default_risk_free_rate=0.04)
```

---

**Full API documentation coming soon!**

For detailed usage examples, see the [Main README](../README.md) and [User Guides](../guides/).
