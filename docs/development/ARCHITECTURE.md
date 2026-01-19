# Architecture

System architecture and design overview.

## Overview

BondTrader is built as a modular Python package with clear separation of concerns.

## Package Structure

```
bondtrader/
├── core/           # Core bond trading logic
├── ml/             # Machine learning models
├── risk/           # Risk management
├── analytics/      # Advanced analytics
├── data/           # Data handling
├── utils/          # Utilities
└── config.py       # Configuration
```

## Core Components

### Bond Valuation Engine

- DCF calculations
- YTM calculation (Newton-Raphson)
- Duration and convexity
- Credit spread adjustments

### Machine Learning Pipeline

- Feature engineering
- Model training
- Prediction and adjustment
- Model evaluation

### Risk Management

- VaR calculations
- Credit risk analysis
- Liquidity risk
- Tail risk analysis

### Data Management

- Market data providers
- Training data generation
- Evaluation datasets
- Data persistence

## Design Patterns

- **Singleton**: Configuration management
- **Strategy**: ML model selection
- **Factory**: Data generation
- **Observer**: Drift detection

## Dependencies

- **NumPy/Pandas**: Numerical computations
- **scikit-learn**: Machine learning
- **Streamlit**: Dashboard UI
- **Plotly**: Visualizations

## Performance Considerations

- Vectorized calculations (NumPy)
- Caching for expensive operations
- Parallel processing for model evaluation
- Lazy loading where appropriate

## Extension Points

- Custom ML models
- New bond types
- Additional risk metrics
- Data providers

---

## Architecture Evolution

This document describes the original architecture. For the latest architecture following industry best practices, see [Architecture v2.0](ARCHITECTURE_V2.md).

**Key improvements in v2.0:**
- Domain-specific exception hierarchy
- Result pattern for explicit error handling
- Repository pattern for data access
- Service layer for business logic orchestration
- Audit logging for compliance
- Circuit breaker for resilience
- Observability (metrics and tracing)

For detailed implementation, see source code and [Organization Summary](ORGANIZATION_SUMMARY.md).
