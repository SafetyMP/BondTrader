# Testing Guide

This directory contains all tests for the BondTrader codebase.

## Test Structure

```
tests/
├── README.md                    # This file
├── conftest.py                  # Shared pytest fixtures
│
├── unit/                        # Unit tests (fast, isolated)
│   ├── test_bond_valuation.py
│   ├── test_arbitrage_detector.py
│   ├── test_config.py
│   └── ...
│
├── integration/                 # Integration tests (multi-module workflows)
│   └── ...
│
├── smoke/                       # Smoke tests (critical path validation)
│   └── test_critical_paths.py
│
└── fixtures/                    # Test data factories
    └── bond_factory.py
```

## Running Tests

### All Tests
```bash
pytest
```

### Unit Tests Only
```bash
pytest tests/unit -m unit
```

### Integration Tests
```bash
pytest tests/integration -m integration
```

### Smoke Tests
```bash
pytest tests/smoke -m smoke
```

### With Coverage Report
```bash
pytest --cov=bondtrader --cov-report=html
# View htmlcov/index.html for detailed coverage report
```

### Verbose Output
```bash
pytest -v
```

### Run Specific Test File
```bash
pytest tests/unit/test_bond_valuation.py
```

### Run Specific Test Function
```bash
pytest tests/unit/test_bond_valuation.py::test_calculate_fair_value
```

## Test Markers

Tests are categorized using pytest markers:

- `@pytest.mark.unit` - Unit tests (fast, isolated)
- `@pytest.mark.integration` - Integration tests (multi-module workflows)
- `@pytest.mark.smoke` - Smoke tests (critical path validation)
- `@pytest.mark.slow` - Slow tests (may take >1 second)
- `@pytest.mark.performance` - Performance/benchmark tests
- `@pytest.mark.requires_data` - Tests that require test data files
- `@pytest.mark.requires_network` - Tests that require network access

Run tests by marker:
```bash
pytest -m unit          # Run only unit tests
pytest -m "not slow"    # Run all except slow tests
pytest -m smoke         # Run smoke tests only
```

## Test Coverage

### Current Coverage
Target: 70%+ code coverage

### Generate Coverage Report
```bash
pytest --cov=bondtrader --cov-report=term-missing --cov-report=html
```

### View HTML Report
```bash
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

## Writing Tests

### Test File Naming
- Test files must start with `test_`
- Test files should be named after the module they test
- Example: `test_bond_valuation.py` tests `bond_valuation.py`

### Test Function Naming
- Test functions must start with `test_`
- Use descriptive names: `test_calculate_fair_value_with_valid_bond`
- Example: `def test_calculate_fair_value():`

### Using Fixtures

Shared fixtures are in `conftest.py`:

```python
def test_example(sample_bond, valuator):
    """Test using shared fixtures"""
    fair_value = valuator.calculate_fair_value(sample_bond)
    assert fair_value > 0
```

### Using Test Factories

Use factories from `tests/fixtures/bond_factory.py`:

```python
from tests.fixtures.bond_factory import create_test_bond

def test_example():
    bond = create_test_bond(coupon_rate=6.0)
    assert bond.coupon_rate == 6.0
```

### Adding Markers

Add markers to test functions:

```python
@pytest.mark.unit
def test_example():
    """This is a unit test"""
    pass

@pytest.mark.integration
def test_workflow():
    """This is an integration test"""
    pass
```

## CI/CD

Tests run automatically on:
- Pull requests
- Commits to main/develop branches

See `.github/workflows/ci.yml` for CI/CD configuration.

## Best Practices

1. **Isolation**: Tests should not depend on each other
2. **Deterministic**: Tests should produce same results every run
3. **Fast**: Unit tests should run in <1 second
4. **Clear**: Test names should describe what they test
5. **Coverage**: Aim for 70-80% code coverage
6. **Edge Cases**: Test boundary conditions and error cases
7. **Mocking**: Mock external dependencies (APIs, file system)

## Smoke Tests

Smoke tests validate critical functionality and must pass before deployment:
- Bond valuation works
- Arbitrage detection works
- Configuration initialization works
- Core calculations work

Run smoke tests before deployment:
```bash
pytest tests/smoke -m smoke
```

## Troubleshooting

### Import Errors
If you see import errors, ensure you're running tests from the project root:
```bash
cd /path/to/BondTrader
pytest
```

### Coverage Not Working
Ensure `pytest-cov` is installed:
```bash
pip install pytest-cov
```

### Tests Not Discovered
Check that:
- Files start with `test_`
- Functions start with `test_`
- `pytest.ini` is configured correctly

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [Test Structure Best Practices](https://docs.pytest.org/en/stable/goodpractices.html)
