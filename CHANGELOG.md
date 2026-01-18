# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- CI/CD pipeline with GitHub Actions
- Configuration management system (`bondtrader/config.py`)
- Comprehensive type hints to core modules
- Enhanced error handling with specific exceptions
- Test coverage expansion (test_arbitrage_detector.py, test_config.py)
- Pre-commit hooks configuration
- Documentation consolidation and GitHub best practices

### Changed
- Improved type hints in `bond_valuation.py`
- Enhanced error handling in core modules
- Organized documentation structure

### Fixed
- Import paths after package reorganization
- Circular import issues in analytics module

## [1.0.0] - 2024-01-18

### Added
- Initial release
- Core bond valuation engine
- Machine learning price adjustments
- Arbitrage detection system
- Interactive Streamlit dashboard
- Risk management framework
- Portfolio optimization
- Factor models
- Backtesting engine
- Multi-curve framework
- Option-adjusted spread (OAS) pricing
- Key rate duration analysis
- Floating rate bond pricing
- Drift detection and model tuning
- AutoML integration
- Explainable AI features

### Features
- Support for multiple bond types (Zero Coupon, Fixed Rate, Floating Rate, Treasury, Corporate, Municipal, High Yield)
- Comprehensive risk metrics (VaR, credit risk, liquidity risk, tail risk)
- Advanced analytics (portfolio optimization, factor models, correlation analysis)
- Machine learning models (Random Forest, Gradient Boosting, Ensemble methods)
- Real-time dashboard with interactive visualizations

---

## Version History

- **1.0.0** - Initial release with full feature set

---

## Types of Changes

- **Added** for new features
- **Changed** for changes in existing functionality
- **Deprecated** for soon-to-be removed features
- **Removed** for now removed features
- **Fixed** for any bug fixes
- **Security** in case of vulnerabilities
