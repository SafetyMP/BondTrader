# Implementation Guide - Module Optimizations

This document tracks the implementation of module optimizations across the BondTrader codebase.

## ✅ Completed

### 1. Requirements Updated
- All new dependencies added to `requirements.txt` with proper versioning
- Dependencies organized by category for maintainability

### 2. Enhanced Database Layer
- Created `bondtrader/data/data_persistence_enhanced.py` with SQLAlchemy
- Connection pooling support
- Backward compatible API

## 🚧 In Progress / To Implement

### 3. Performance Optimizations (Numba)

**Files to Update:**
- `bondtrader/risk/risk_management.py` - Monte Carlo simulations
- `bondtrader/analytics/oas_pricing.py` - Binomial tree pricing
- `bondtrader/analytics/advanced_analytics.py` - Monte Carlo scenarios

**Implementation:**
```python
try:
    from numba import jit, prange
    USE_NUMBA = True
except ImportError:
    USE_NUMBA = False
    # Fallback decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
```

### 4. ML Model Enhancements (XGBoost/LightGBM)

**Files to Update:**
- `bondtrader/ml/ml_adjuster.py` - Add XGBoost/LightGBM support
- `bondtrader/ml/ml_adjuster_enhanced.py` - Enhance with new models
- `bondtrader/ml/ml_advanced.py` - Add to ensemble

**Implementation:**
```python
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Add to model_type options
if self.model_type == "xgboost":
    self.model = XGBRegressor(...)
elif self.model_type == "lightgbm":
    self.model = LGBMRegressor(...)
```

### 5. Portfolio Optimization (CVXPY)

**Files to Update:**
- `bondtrader/analytics/portfolio_optimization.py` - Add CVXPY option

**Implementation:**
- Add `use_cvxpy` parameter to optimization methods
- Implement convex optimization using CVXPY
- Maintain backward compatibility with scipy.optimize

### 6. Hyperparameter Optimization (Optuna)

**Files to Update:**
- `bondtrader/ml/bayesian_optimization.py` - Enhance with Optuna
- `bondtrader/ml/automl.py` - Integrate Optuna

### 7. Financial Libraries (QuantLib-Python)

**Files to Update:**
- `bondtrader/core/bond_valuation.py` - Add QuantLib integration
- `bondtrader/analytics/multi_curve.py` - Use QuantLib for yield curves
- `bondtrader/analytics/oas_pricing.py` - QuantLib OAS calculations

### 8. Business Day Handling (pandas-market-calendars)

**Files to Update:**
- `bondtrader/core/bond_valuation.py` - Use market calendars
- `bondtrader/analytics/backtesting.py` - Business day calculations

### 9. Data Validation (Pydantic)

**Files to Update:**
- `bondtrader/core/bond_models.py` - Add Pydantic validation
- `bondtrader/config.py` - Use Pydantic for config

### 10. Yield Curve Fitting (statsmodels)

**Files to Update:**
- `bondtrader/analytics/advanced_analytics.py` - Use statsmodels for Nelson-Siegel
- `bondtrader/analytics/factor_models.py` - Enhance with statsmodels

## Notes

- All changes should maintain backward compatibility
- Use try/except for optional dependencies
- Add configuration flags for enabling/disabling features
- Update tests for new functionality
