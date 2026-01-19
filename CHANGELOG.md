# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Security
- **CORS Configuration**: Fixed security vulnerability - removed wildcard CORS origins, now configurable via environment variables
- **Default Passwords**: Removed hardcoded default passwords in authentication and secrets management
- **API Authentication**: Implemented optional Bearer token authentication for API endpoints
- **Rate Limiting**: Added per-IP rate limiting middleware to prevent abuse
- **Input Validation**: Enhanced date validation and logical checks in API endpoints
- **Error Handling**: Improved error handling to prevent information leakage

### Changed
- **Codebase Truncation**: Reduced codebase size by ~25.6% (7,933 lines) while maintaining full functionality
  - Removed deprecated files: `data_persistence.py`, `bond_models_pydantic.py`, `config_pydantic.py`
  - Consolidated documentation: Reduced from 45 markdown files to 18 essential files (~7,326 lines saved)
  - Consolidated risk modules: Merged enhanced credit risk methods into `RiskManager` (80 lines saved)
  - Removed unused `base_ml_adjuster.py` (231 lines saved)
  - Updated all documentation references to reflect current structure
- **Risk Module Consolidation**: Enhanced credit risk methods (Merton model, migration analysis, CVaR) now in `RiskManager`
  - `CreditRiskEnhanced` is now a thin backward-compatible wrapper delegating to `RiskManager`
  - All functionality preserved with improved code organization

### Added
- **Input Validation**: Comprehensive validation module (`bondtrader/utils/validation.py`) with 9+ validators
  - Numeric validation (positive, range, percentage, probability)
  - List and weight validation
  - Secure file path validation with security checks
  - Credit rating and bond input validation
- **Integration Tests**: End-to-end test coverage for training and evaluation pipelines
  - `tests/integration/test_training_pipeline.py` - Training workflow tests
  - `tests/integration/test_evaluation_pipeline.py` - Evaluation workflow tests
- **Performance Benchmarks**: Performance test infrastructure (`tests/benchmarks/test_performance.py`)
  - Benchmarks for critical operations
  - Performance regression tests
  - Scalability tests
- **Base ML Class**: Abstract base class (`bondtrader/ml/base_ml_adjuster.py`) to reduce code duplication
  - Common save/load functionality
  - Shared feature creation
  - Path validation integration
- **Error Handling**: Enhanced error handling with specific exception types
  - Replaced generic `Exception` with specific types (ValueError, TypeError, FileNotFoundError, etc.)
  - Improved error messages and logging
  - Better error recovery strategies
- **Security**: File path validation and sanitization
  - Path traversal prevention
  - File extension validation
  - Dangerous character filtering
- CI/CD pipeline with GitHub Actions
- Configuration management system (`bondtrader/config.py`)
- Pre-commit hooks configuration
- Table of contents in README.md for better navigation
- Organized test structure by module type

### Changed
- **Test Coverage**: Increased from ~10% to ~65-70%
  - Added validation utilities tests (100% coverage)
  - Added integration tests
  - Added performance benchmarks
- **Type Hints**: Increased from ~40% to ~90% coverage
  - Added type hints to scripts
  - Added type hints to data modules
  - Added type hints to analytics modules
- **Error Handling**: Specific exception handling throughout codebase
  - Improved error messages in ML model save/load
  - Better error recovery in training pipelines
- Improved type hints in `bond_valuation.py`, `bond_models.py`
- Enhanced error handling in core modules
- Organized documentation structure (moved review files to `docs/development/reviews/`)
- Reorganized tests into module-based subdirectories
- Consolidated data_persistence.py to use enhanced module
- Consolidated duplicate test files (merged test_arbitrage.py into test_arbitrage_detector.py)
- Updated CONTRIBUTING.md with Code of Conduct references
- Improved README.md structure with comprehensive documentation links
- **CI/CD Quality Gates**: Enabled coverage threshold enforcement (50%)

### Fixed
- Import paths after package reorganization
- Circular import issues in analytics module
- Security vulnerabilities (path traversal in file I/O, CORS wildcard, default passwords)
- Generic exception handling patterns (replaced 23 bare except clauses with specific exceptions)
- Code formatting issues (black and isort compliance)
- Duplicate pytest markers in pytest.ini
- Hardcoded database paths (now configurable via environment variables)

### Security
- Added file path validation to prevent directory traversal attacks
- Enhanced input sanitization for file paths
- Secure file I/O operations in ML model save/load
- Fixed CORS configuration (removed wildcard, added environment-based configuration)
- Removed default passwords (now requires environment variables)
- Implemented API key authentication (optional Bearer token)
- Added rate limiting middleware (per-IP, configurable limits)
- Improved error handling to prevent information leakage

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
