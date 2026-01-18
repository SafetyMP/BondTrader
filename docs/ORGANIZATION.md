# Codebase Organization

This document describes the organization structure of the BondTrader codebase.

## Directory Structure

```
BondTrader/
├── bondtrader/              # Main package
│   ├── core/               # Core bond trading logic
│   ├── ml/                 # Machine learning modules
│   ├── risk/               # Risk management
│   ├── analytics/          # Advanced analytics
│   ├── data/               # Data handling
│   ├── utils/              # Utilities
│   ├── config.py           # Configuration
│   └── config_pydantic.py  # Optional Pydantic config
│
├── scripts/                # Executable scripts
├── tests/                  # Test suite
│   ├── unit/               # Unit tests organized by module
│   │   ├── core/          # Core module tests
│   │   ├── ml/            # ML module tests
│   │   ├── risk/          # Risk module tests
│   │   ├── analytics/     # Analytics module tests
│   │   ├── data/          # Data module tests
│   │   └── utils/         # Utils/config module tests
│   ├── integration/        # Integration tests
│   ├── smoke/              # Smoke tests
│   └── fixtures/           # Test fixtures
│
└── docs/                   # Documentation
    ├── api/                # API documentation
    ├── development/        # Development docs
    ├── guides/             # User guides
    ├── implementation/     # Implementation docs
    └── status/             # Status and status tracking docs
```

## Root Directory

The root directory contains only essential project files:
- `README.md` - Main project documentation
- `CHANGELOG.md` - Version history
- `CONTRIBUTING.md` - Contribution guidelines
- `ROADMAP.md` - Project roadmap
- `LICENSE` - License file
- `requirements.txt` - Python dependencies
- `setup.py` - Package setup
- `pytest.ini` - Pytest configuration

## Documentation Organization

### docs/api/
API reference documentation for public interfaces.

### docs/development/
Technical documentation for developers:
- Architecture and design decisions
- Implementation summaries
- Improvement recommendations
- Industry comparisons

### docs/guides/
User-facing guides:
- Quick start guide
- Training data guide
- Evaluation dataset guide
- Drift detection guide
- User guide

### docs/implementation/
Implementation and status documentation:
- Implementation guides
- Status reports
- Completion summaries
- Module optimization reviews

### docs/status/
Status tracking documents:
- Evaluation errors and optimizations
- Model tuning improvements
- Production testing readiness
- Training improvements

## Test Organization

Tests are organized by module type to match the package structure:
- `tests/unit/core/` - Tests for core modules
- `tests/unit/ml/` - Tests for ML modules
- `tests/unit/risk/` - Tests for risk modules
- `tests/unit/analytics/` - Tests for analytics modules
- `tests/unit/data/` - Tests for data modules
- `tests/unit/utils/` - Tests for utilities and config

This organization makes it easy to:
- Find tests for a specific module
- Run tests for a specific area
- Understand test coverage by module

## Benefits of This Organization

1. **Clear Separation**: Related code is grouped together
2. **Easy Navigation**: Predictable file locations
3. **Scalability**: Easy to add new modules and tests
4. **Maintainability**: Clear ownership and responsibility
5. **Professional Structure**: Follows Python best practices
