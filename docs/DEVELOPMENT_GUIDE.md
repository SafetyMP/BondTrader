# Development Guide

This guide consolidates key development documentation for BondTrader.

## Architecture

- [Architecture Overview](development/ARCHITECTURE.md) - System architecture
- [Architecture v2.0](development/ARCHITECTURE_V2.md) - Latest architecture
- [Service Layer Migration](development/SERVICE_LAYER_MIGRATION_GUIDE.md) - Migration guide

## Codebase Organization

- [Codebase Organization](ORGANIZATION.md) - Directory structure
- [Codebase Reorganization](development/CODEBASE_REORGANIZATION.md) - Reorganization guide

## ML Pipeline

- [ML Pipeline Analysis](analysis/ML_PIPELINE_ANALYSIS.md) - Comprehensive ML pipeline analysis
- [Model Tuning Evaluation](analysis/MODEL_TUNING_EVALUATION.md) - Tuning results

## Configuration & Migration

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

## Security

- [Security Guide](SECURITY.md) - Security practices
- [Security Audit](SECURITY_AUDIT_REPORT.md) - Security audit results

## Git History

See `GIT_HISTORY_CLEANUP_GUIDE.md` for instructions on cleaning git history (e.g., removing API keys).

## GitHub Preparation

See `GITHUB_PREP_CHECKLIST.md` for pre-push checklist.
