# Service Layer Migration Guide

This guide explains how to migrate code from direct instantiation to using the service layer and dependency injection container.

## Quick Start

**Before**: Direct instantiation bypasses service layer
```python
valuator = BondValuator()
fair_value = valuator.calculate_fair_value(bond)
```

**After**: Use service layer for all operations
```python
from bondtrader.core.container import get_container
bond_service = get_container().get_bond_service()
result = bond_service.calculate_valuation(bond_id)
if result.is_ok():
    fair_value = result.value['fair_value']
```

## Migration Patterns

### Pattern 1: Bond Operations

#### Before
```python
from bondtrader.core.bond_valuation import BondValuator
from bondtrader.data.data_persistence import EnhancedBondDatabase

db = EnhancedBondDatabase()
valuator = BondValuator()

# Save bond
db.save_bond(bond)

# Calculate valuation
fair_value = valuator.calculate_fair_value(bond)
ytm = valuator.calculate_yield_to_maturity(bond)
```

#### After
```python
from bondtrader.core.container import get_container

container = get_container()
bond_service = container.get_bond_service()

# Create bond (includes validation, audit logging)
result = bond_service.create_bond(bond)
if result.is_err():
    # Handle error
    pass

# Calculate valuation (includes audit logging, metrics)
result = bond_service.calculate_valuation(bond.bond_id)
if result.is_ok():
    valuation = result.value
    fair_value = valuation["fair_value"]
    ytm = valuation["ytm"]
```

### Pattern 2: Direct Bond Valuation (Without Repository)

If you have a `Bond` object and don't need repository lookup:

#### Before
```python
valuator = BondValuator()
fair_value = valuator.calculate_fair_value(bond)
```

#### After
```python
from bondtrader.core.container import get_container

container = get_container()
bond_service = container.get_bond_service()

# Use calculate_valuation_for_bond for Bond objects
result = bond_service.calculate_valuation_for_bond(bond)
if result.is_ok():
    valuation = result.value
    fair_value = valuation["fair_value"]
```

### Pattern 3: Batch Operations

#### Before
```python
valuator = BondValuator()
valuations = []
for bond in bonds:
    fair_value = valuator.calculate_fair_value(bond)
    valuations.append({"bond_id": bond.bond_id, "fair_value": fair_value})
```

#### After
```python
from bondtrader.core.container import get_container

container = get_container()
bond_service = container.get_bond_service()

# Batch valuation with service layer
result = bond_service.calculate_valuations_batch(bonds)
if result.is_ok():
    valuations = result.value  # List of valuation dictionaries
```

### Pattern 4: ML Modules

ML modules currently create their own `BondValuator()` instances. For now, they still work, but future refactoring should accept valuator as dependency:

#### Current (Still Works)
```python
from bondtrader.ml.ml_adjuster import MLBondAdjuster

ml_adjuster = MLBondAdjuster()  # Creates its own valuator internally
```

#### Future Pattern (Recommended)
```python
from bondtrader.core.container import get_container
from bondtrader.ml.ml_adjuster import MLBondAdjuster

container = get_container()
valuator = container.get_valuator()

# Pass valuator as dependency (requires ML module refactoring)
ml_adjuster = MLBondAdjuster(valuator=valuator)
```

### Pattern 5: API Endpoints

#### Before
```python
from bondtrader.core.bond_valuation import BondValuator
from bondtrader.data.data_persistence import EnhancedBondDatabase

valuator = BondValuator()
db = EnhancedBondDatabase()

@app.get("/bonds/{bond_id}")
async def get_bond(bond_id: str):
    bond = db.load_bond(bond_id)
    if not bond:
        raise HTTPException(status_code=404, detail="Not found")
    return bond
```

#### After
```python
from bondtrader.core.container import get_container
from bondtrader.core.exceptions import DataNotFoundError

container = get_container()
bond_service = container.get_bond_service()

@app.get("/bonds/{bond_id}")
async def get_bond(bond_id: str):
    result = bond_service.get_bond(bond_id)
    if result.is_err():
        error = result.error
        if isinstance(error, DataNotFoundError):
            raise HTTPException(status_code=404, detail=str(error))
        raise HTTPException(status_code=500, detail="Failed to retrieve bond")
    return result.value
```

## Service Container API

### Getting Services

```python
from bondtrader.core.container import get_container

container = get_container()

# Get services (singleton instances)
bond_service = container.get_bond_service()
valuator = container.get_valuator()
arbitrage_detector = container.get_arbitrage_detector()
risk_manager = container.get_risk_manager()
repository = container.get_repository()
database = container.get_database()
```

### Configuration

```python
container = get_container()
config = container.config  # Access configuration

# Override risk-free rate if needed
valuator = container.get_valuator(risk_free_rate=0.04)

# Override database path if needed
db = container.get_database(db_path="/custom/path/bonds.db")
```

## Service Layer API

### Bond CRUD Operations

```python
bond_service = container.get_bond_service()

# Create bond
result = bond_service.create_bond(bond)
if result.is_ok():
    created_bond = result.value
else:
    error = result.error  # InvalidBondError, BusinessRuleViolation, etc.

# Get bond
result = bond_service.get_bond(bond_id)
if result.is_ok():
    bond = result.value
else:
    error = result.error  # DataNotFoundError, etc.

# Find bonds with filters
filters = {"bond_type": BondType.CORPORATE, "issuer": "Example Corp"}
result = bond_service.find_bonds(filters=filters)
if result.is_ok():
    bonds = result.value

# Get bond count
result = bond_service.get_bond_count(filters=filters)
if result.is_ok():
    count = result.value
```

### Valuation Operations

```python
bond_service = container.get_bond_service()

# Calculate valuation (requires bond_id, uses repository)
result = bond_service.calculate_valuation(bond_id)
if result.is_ok():
    valuation = result.value
    # Contains: fair_value, ytm, duration, convexity, market_price, mismatch_percentage

# Calculate valuation for Bond object (no repository lookup)
result = bond_service.calculate_valuation_for_bond(bond)
if result.is_ok():
    valuation = result.value

# Batch valuation
result = bond_service.calculate_valuations_batch(bonds)
if result.is_ok():
    valuations = result.value  # List of valuation dictionaries
```

## Error Handling

The service layer uses the `Result[T, E]` pattern for explicit error handling:

```python
from bondtrader.core.result import Result
from bondtrader.core.exceptions import DataNotFoundError, ValuationError

result = bond_service.calculate_valuation(bond_id)

if result.is_ok():
    valuation = result.value
    # Use valuation
elif isinstance(result.error, DataNotFoundError):
    # Handle not found
    print(f"Bond not found: {result.error}")
elif isinstance(result.error, ValuationError):
    # Handle valuation error
    print(f"Valuation failed: {result.error}")
else:
    # Handle other errors
    print(f"Unexpected error: {result.error}")
```

## Benefits

### 1. Audit Logging
All operations through the service layer are automatically logged:
- Bond creation/updates
- Valuation calculations
- Data access

### 2. Metrics Collection
Operations are automatically metered:
- Counters for operations
- Histograms for values
- Error tracking

### 3. Consistent Error Handling
All operations return `Result[T, E]` for explicit error handling.

### 4. Shared Instances
Services are singletons, reducing memory usage and ensuring consistent configuration.

### 5. Testability
Easy to inject mocks for testing:
```python
from bondtrader.core.repository import InMemoryBondRepository
from bondtrader.core.service_layer import BondService

# Use in-memory repository for testing
test_repo = InMemoryBondRepository()
test_service = BondService(repository=test_repo)
```

## Migration Checklist

- [ ] Replace `BondValuator()` instantiations with `container.get_valuator()`
- [ ] Replace `EnhancedBondDatabase()` instantiations with `container.get_database()`
- [ ] Replace direct `db.save_bond()` with `bond_service.create_bond()`
- [ ] Replace direct `db.load_bond()` with `bond_service.get_bond()`
- [ ] Replace direct `valuator.calculate_fair_value()` with `bond_service.calculate_valuation()`
- [ ] Update error handling to use `Result` pattern
- [ ] Update API endpoints to use service layer
- [ ] Update scripts to use service layer
- [ ] Add audit logging where needed (automatic with service layer)

## Common Issues

### Issue: "I need a BondValuator for ML modules"

**Solution:** ML modules currently create their own instances. For now, this is acceptable, but future refactoring should accept valuator as a constructor parameter.

### Issue: "I have a Bond object, not a bond_id"

**Solution:** Use `bond_service.calculate_valuation_for_bond(bond)` instead of `calculate_valuation(bond_id)`.

### Issue: "I need to override configuration"

**Solution:** Pass overrides to container methods:
```python
valuator = container.get_valuator(risk_free_rate=0.04)
db = container.get_database(db_path="/custom/path")
```

## Next Steps

1. **Migrate API Server** ✅ (Completed)
2. **Migrate Training Scripts** ✅ (In Progress)
3. **Refactor ML Modules** - Accept valuator as dependency
4. **Refactor Analytics Modules** - Use shared services
5. **Add Linting Rules** - Prevent direct instantiations

## Questions?

See:
- [Architecture Documentation](ARCHITECTURE_V2.md)
- [Architecture Gap Analysis](ARCHITECTURE_GAP_ANALYSIS.md)
- [Service Layer Code](../bondtrader/core/service_layer.py)
- [Container Code](../bondtrader/core/container.py)
