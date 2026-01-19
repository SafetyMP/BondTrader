# BondTrader 📊

> A production-ready Python platform for bond valuation, arbitrage detection, and risk analysis with machine learning enhancements. Built with security, scalability, and maintainability in mind.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Security](https://img.shields.io/badge/Security-Hardened-green.svg)](SECURITY.md)

**Key Highlights:**
- 🔒 **Security Hardened**: API authentication, rate limiting, CORS protection, input validation
- 🚀 **Production Ready**: Comprehensive error handling, logging, monitoring, CI/CD pipeline
- 📊 **Enterprise Grade**: RESTful API, interactive dashboard, ML model management
- 🧪 **Well Tested**: 70%+ code coverage, unit/integration/smoke tests
- 📚 **Fully Documented**: Comprehensive guides, API reference, architecture docs

## 📑 Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
  - [Installation](#installation)
  - [Usage](#usage)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Documentation](#-documentation)
  - [User Guides](#user-guides)
  - [API Reference](#api-reference)
  - [Executive Documentation](#executive-documentation)
  - [Development Documentation](#development-documentation)
- [Testing](#-testing)
- [Contributing](#-contributing)
- [Security](#-security)
- [License](#-license)
- [Support](#-support)

## ✨ Features

### 🎯 Core Functionality

- **Bond Valuation**: Calculate fair values for various bond types (Zero Coupon, Fixed Rate, Treasury, Corporate, High Yield, Floating Rate) using DCF, YTM, and advanced pricing models
- **RESTful API**: Production-ready FastAPI server with authentication, rate limiting, and comprehensive error handling
- **Machine Learning Adjustments**: ML-powered price adjustments using Random Forest, Gradient Boosting, AutoML, and ensemble methods
- **Arbitrage Detection**: Identify mispriced bonds and arbitrage opportunities with configurable profit thresholds
- **Interactive Dashboard**: Streamlit-based dashboard with real-time visualizations, bond comparisons, and portfolio analysis
- **Risk Management**: Comprehensive risk analysis including VaR (Historical, Parametric, Monte Carlo), credit risk, liquidity risk, and tail risk
- **Portfolio Optimization**: Markowitz optimization, Black-Litterman model, and risk parity strategies

### 📊 Advanced Capabilities

- **Multi-Curve Framework**: Separate discounting and forwarding curves for sophisticated yield curve modeling
- **Option-Adjusted Spread (OAS)**: Pricing for bonds with embedded options using binomial tree models
- **Key Rate Duration**: Sensitivity analysis at key yield curve points for advanced risk management
- **Factor Models**: PCA-based factor extraction and risk attribution for portfolio analysis
- **Backtesting Engine**: Historical performance validation and strategy testing with comprehensive metrics
- **Execution Strategies**: Market impact modeling and optimal execution algorithms
- **Explainable AI**: Feature importance analysis and prediction explanations using SHAP values
- **Drift Detection**: Model performance monitoring and automatic retraining pipelines
- **MLflow Integration**: Experiment tracking, model versioning, and deployment management
- **Audit Logging**: Comprehensive audit trails for compliance and traceability

## 🚀 Quick Start

### Installation

#### Option 1: Docker (Recommended)

The easiest way to run BondTrader is using Docker:

1. **Clone the repository**:
```bash
git clone https://github.com/SafetyMP/BondTrader.git
cd BondTrader
```

2. **Configure environment**:
```bash
cp docker/.env.example docker/.env
# Edit docker/.env with your settings
```

3. **Start all services**:
```bash
make up
# Or manually: docker-compose -f docker/docker-compose.yml up -d
```

4. **Access services**:
- Dashboard: http://localhost:8501
- API: http://localhost:8000/docs
- MLflow: http://localhost:5000

See [Docker Documentation](docker/README.md) for details.

#### Option 2: Local Installation

1. **Clone the repository**:
```bash
git clone https://github.com/SafetyMP/BondTrader.git
cd BondTrader
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Usage

#### Running the Dashboard

Launch the Streamlit dashboard:
```bash
streamlit run scripts/dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`

#### Using the REST API

Start the API server:
```bash
python scripts/api_server.py
# Or with uvicorn: uvicorn scripts.api_server:app --reload
```

The API will be available at `http://localhost:8000` with interactive documentation at `/docs`.

**Example API Usage:**
```bash
# Health check
curl http://localhost:8000/health

# Create a bond (with authentication if enabled)
curl -X POST "http://localhost:8000/bonds" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "bond_id": "BOND-001",
    "bond_type": "CORPORATE",
    "face_value": 1000,
    "coupon_rate": 0.05,
    "maturity_date": "2029-12-31",
    "issue_date": "2024-01-01",
    "current_price": 950,
    "credit_rating": "BBB",
    "issuer": "Example Corp"
  }'

# Get bond valuation
curl "http://localhost:8000/bonds/BOND-001/valuation" \
  -H "Authorization: Bearer YOUR_API_KEY"

# Find arbitrage opportunities
curl "http://localhost:8000/arbitrage/opportunities?min_profit_percentage=0.01" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

See [API Reference](docs/api/API_REFERENCE.md) for complete documentation.

#### Using the Python Library

```python
from bondtrader.core import Bond, BondType, BondValuator, ArbitrageDetector
from bondtrader.ml import EnhancedMLBondAdjuster
from datetime import datetime, timedelta

# Create a bond
bond = Bond(
    bond_id="BOND-001",
    bond_type=BondType.CORPORATE,
    face_value=1000,
    coupon_rate=5.0,
    maturity_date=datetime.now() + timedelta(days=1825),  # 5 years
    issue_date=datetime.now() - timedelta(days=365),
    current_price=950,
    credit_rating="BBB",
    issuer="Example Corp",
    frequency=2
)

# Value the bond
valuator = BondValuator(risk_free_rate=0.03)
fair_value = valuator.calculate_fair_value(bond)
ytm = valuator.calculate_yield_to_maturity(bond)
duration = valuator.calculate_duration(bond, ytm)
convexity = valuator.calculate_convexity(bond, ytm)

# Check for arbitrage
detector = ArbitrageDetector(valuator=valuator)
opportunities = detector.find_arbitrage_opportunities([bond], min_profit_percentage=0.01)

# ML-enhanced valuation
ml_adjuster = EnhancedMLBondAdjuster(model_type="random_forest")
ml_result = ml_adjuster.predict_adjusted_value(bond)

print(f"Fair Value: ${fair_value:.2f}")
print(f"YTM: {ytm*100:.2f}%")
print(f"Duration: {duration:.2f} years")
print(f"Convexity: {convexity:.2f}")
print(f"ML-Adjusted Value: ${ml_result['ml_adjusted_fair_value']:.2f}")
print(f"Arbitrage Opportunities: {len(opportunities)}")
```

#### Training ML Models

Train all models from scratch:
```bash
python scripts/train_all_models.py
```

#### Running Tests

```bash
pytest tests/ -v
```

With coverage:
```bash
pytest tests/ -v --cov=bondtrader --cov-report=html
```

## 📁 Project Structure

```
BondTrader/
├── bondtrader/                     # Main package
│   ├── core/                       # Core bond trading modules
│   │   ├── bond_models.py         # Bond data models
│   │   ├── bond_valuation.py      # Valuation engine
│   │   ├── arbitrage_detector.py  # Arbitrage detection
│   │   └── quantlib_integration.py # QuantLib integration
│   ├── ml/                         # Machine Learning modules
│   │   ├── ml_adjuster.py         # Basic ML adjuster
│   │   ├── ml_adjuster_enhanced.py # Enhanced ML with tuning
│   │   ├── ml_advanced.py         # Advanced ensemble methods
│   │   └── automl.py              # AutoML integration
│   ├── risk/                       # Risk management modules
│   │   ├── risk_management.py     # Core risk metrics
│   │   ├── credit_risk_enhanced.py # Credit risk analysis
│   ├── analytics/                  # Analytics modules
│   ├── data/                       # Data management
│   └── utils/                      # Utilities
├── docs/                           # Documentation
│   ├── guides/                     # User guides
│   ├── api/                        # API reference
│   ├── development/                # Development docs
│   ├── executive/                  # Executive summaries
│   ├── analysis/                   # Technical analysis
│   └── demo/                       # Demo reports
├── tests/                          # Test suite
│   ├── unit/                       # Unit tests
│   ├── integration/                # Integration tests
│   ├── smoke/                      # Smoke tests
│   └── benchmarks/                 # Performance benchmarks
├── scripts/                        # Utility scripts
├── .github/                        # GitHub configuration
│   ├── workflows/                  # CI/CD workflows
│   └── ISSUE_TEMPLATE/             # Issue templates
└── [configuration files]           # Setup, requirements, etc.
│   │   └── liquidity_risk_enhanced.py # Liquidity risk
│   ├── analytics/                  # Analytics and advanced features
│   │   ├── portfolio_optimization.py # Portfolio optimization
│   │   ├── backtesting.py         # Backtesting engine
│   │   └── factor_models.py       # Factor analysis
│   ├── data/                       # Data handling modules
│   │   ├── data_persistence_enhanced.py # Database layer
│   │   └── training_data_generator.py # Training data
│   ├── utils/                      # Utility functions
│   └── config.py                   # Configuration management
│
├── scripts/                        # Executable scripts
│   ├── dashboard.py                # Streamlit dashboard
│   ├── train_all_models.py         # Model training
│   └── evaluate_models.py          # Model evaluation
│
├── tests/                          # Test suite
│   ├── unit/                       # Unit tests (organized by module)
│   │   ├── core/                  # Core module tests
│   │   ├── ml/                    # ML module tests
│   │   ├── risk/                  # Risk module tests
│   │   ├── analytics/             # Analytics tests
│   │   └── data/                  # Data module tests
│   ├── integration/                # Integration tests
│   └── smoke/                      # Smoke tests
│
├── docs/                           # Documentation
│   ├── guides/                     # User guides
│   ├── api/                        # API documentation
│   ├── development/                # Development docs
│   ├── implementation/             # Implementation docs
│   └── status/                     # Status tracking
│
├── requirements.txt                # Python dependencies
├── setup.py                        # Package setup
├── LICENSE                         # License file
├── CHANGELOG.md                    # Version history
├── CONTRIBUTING.md                 # Contribution guidelines
├── ROADMAP.md                      # Project roadmap
└── README.md                       # This file
```

## 🔧 Configuration

### Environment Variables

The system uses environment variables for configuration. Copy `.env.example` to `.env` and configure:

```bash
cp env.example .env
```

**Essential Configuration:**
```env
# API Security (Recommended for production)
API_KEY=your_secret_api_key_here
REQUIRE_API_KEY=false  # Set to true to enable API authentication
CORS_ALLOWED_ORIGINS=http://localhost:8000,http://localhost:8501

# Rate Limiting
API_RATE_LIMIT=100  # Requests per window
API_RATE_LIMIT_WINDOW=60  # Window in seconds

# Secrets Management (Required if using file backend)
SECRETS_MASTER_PASSWORD=your_master_password  # Required, no default
SECRETS_SALT=your_random_salt  # Required, no default

# External Data Sources (Optional)
FRED_API_KEY=your_fred_api_key  # For Federal Reserve Economic Data
FINRA_API_KEY=your_finra_api_key  # For FINRA market data

# Application Configuration
DEFAULT_RFR=0.03  # Default risk-free rate
ML_MODEL_TYPE=random_forest  # ML model type
BOND_DB_PATH=./data/bonds.db  # Database path
```

**Note**: The system works without API keys using simulated data. API keys are only needed for live market data integration. See [env.example](env.example) for all available options.

### Programmatic Configuration

```python
from bondtrader.config import get_config, Config

# Get default config
config = get_config()

# Create custom config
custom_config = Config(
    default_risk_free_rate=0.04,
    ml_model_type='gradient_boosting'
)
```

## 📖 Documentation

Comprehensive documentation is available in the [`docs/`](docs/) directory:

### User Guides
- **[Quick Start Guide](docs/guides/QUICK_START_GUIDE.md)** - Quick introduction and setup
- **[User Guide](docs/guides/USER_GUIDE.md)** - Complete usage guide
- **[Training Data Guide](docs/guides/TRAINING_DATA.md)** - Generating training datasets
- **[Evaluation Dataset Guide](docs/guides/EVALUATION_DATASET.md)** - Creating evaluation datasets
- **[Historical Data Fetching](docs/guides/HISTORICAL_DATA_FETCHING.md)** - Fetching real market data
- **[Drift Detection](docs/guides/DRIFT_DETECTION.md)** - ML model monitoring

### API Reference
- **[API Reference](docs/api/API_REFERENCE.md)** - Complete API documentation

### Executive Documentation
- **[CTO Review](docs/executive/CTO_REVIEW_AND_OPTIMIZATION.md)** - Comprehensive CTO review
- **[Executive Demo Guide](docs/executive/EXECUTIVE_DEMO_GUIDE.md)** - Demo instructions for stakeholders
- **[Complete CTO Deliverable](docs/executive/COMPLETE_CTO_DELIVERABLE.md)** - Executive summary

### Development Documentation
- **[Architecture](docs/development/ARCHITECTURE.md)** - System architecture overview
- **[Contributing Guide](CONTRIBUTING.md)** - How to contribute
- **[Codebase Organization](docs/ORGANIZATION.md)** - Project structure
- **[Competitive Analysis](docs/development/COMPETITIVE_ANALYSIS.md)** - Industry comparison

### Technical Analysis
- **[Performance Optimizations](docs/analysis/PERFORMANCE_FIXES_COMPLETE.md)** - Performance improvements
- **[ML Improvements](docs/analysis/)** - Machine learning enhancements
- **[Redundancy Analysis](docs/analysis/COMPUTATIONAL_REDUNDANCY_ANALYSIS.md)** - Code optimization

### Additional Resources
- **[Changelog](CHANGELOG.md)** - Version history and changes
- **[Roadmap](ROADMAP.md)** - Planned features and improvements
- **[Security Policy](SECURITY.md)** - Security guidelines

For a complete overview, see the [Documentation Index](docs/README.md).

## 🐳 Docker Deployment

BondTrader is fully containerized with Docker and Docker Compose.

### Quick Start with Docker

```bash
# Setup environment
cp docker/.env.example docker/.env

# Start all services
make up

# Access services
# - Dashboard: http://localhost:8501
# - API: http://localhost:8000/docs
# - MLflow: http://localhost:5000
```

### Docker Components

- **API Service** - FastAPI REST API (port 8000)
- **Dashboard** - Streamlit web interface (port 8501)
- **MLflow** - ML experiment tracking (port 5000)
- **PostgreSQL** - Database backend
- **Redis** - Caching layer
- **ML Training** - On-demand model training

See [Docker Setup Guide](docs/guides/DOCKER_SETUP.md) for detailed instructions.

## 🧪 Testing

BondTrader includes a comprehensive test suite with 70%+ code coverage:

### Running Tests

**All Tests:**
```bash
pytest tests/ -v
```

**By Category:**
```bash
pytest tests/unit -m unit -v          # Unit tests (fast, isolated)
pytest tests/integration -m integration -v  # Integration tests
pytest tests/smoke -m smoke -v        # Smoke tests (critical paths)
```

**With Coverage:**
```bash
pytest tests/ -v --cov=bondtrader --cov-report=html --cov-report=term-missing
# View detailed report: open htmlcov/index.html
```

**Performance Benchmarks:**
```bash
pytest tests/benchmarks -m performance -v
```

### Test Structure
- **Unit Tests**: Fast, isolated tests for individual components
- **Integration Tests**: End-to-end workflow validation
- **Smoke Tests**: Critical path validation for deployment
- **Performance Tests**: Benchmarks and regression tests

### CI/CD
Tests run automatically on:
- Pull requests
- Commits to main/develop branches
- Multiple Python versions (3.9, 3.10, 3.11)

See [tests/README.md](tests/README.md) for detailed testing documentation.

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

1. **Fork and clone** the repository
2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install in development mode
   ```
4. **Install pre-commit hooks** (optional but recommended):
   ```bash
   pre-commit install
   ```

### Code Quality Standards

Before submitting a PR, ensure:
- ✅ All tests pass: `pytest tests/ -v`
- ✅ Code is formatted: `black bondtrader/ scripts/ tests/`
- ✅ Imports are sorted: `isort bondtrader/ scripts/ tests/`
- ✅ No critical linting errors: `flake8 bondtrader/ scripts/ tests/ --select=E9,F63,F7,F82`
- ✅ Type hints added where appropriate
- ✅ Documentation updated

### Contribution Workflow

1. Create a feature branch (`git checkout -b feature/amazing-feature`)
2. Make your changes and add tests
3. Ensure all tests pass and code quality checks pass
4. Update documentation if needed
5. Submit a Pull Request with a clear description

For more information, see [CONTRIBUTING.md](CONTRIBUTING.md) and our [Code of Conduct](CODE_OF_CONDUCT.md).

## 🏗️ Architecture & Design

### Production-Ready Features
- **Service Layer Pattern**: Clean separation of business logic and data access
- **Repository Pattern**: Abstracted data persistence layer
- **Result Pattern**: Explicit error handling without exceptions
- **Circuit Breaker**: Resilience patterns for external service calls
- **Observability**: Comprehensive logging, metrics, and tracing
- **Audit Logging**: Full audit trails for compliance

### Code Quality
- **Type Hints**: 90%+ type coverage for better IDE support and error detection
- **Error Handling**: Specific exception types throughout (no bare except clauses)
- **Input Validation**: Comprehensive validation with security checks
- **Code Formatting**: Consistent formatting with black and isort
- **Linting**: flake8 compliance with critical error checks
- **Testing**: 70%+ code coverage with unit, integration, and smoke tests

### Recent Improvements (January 2025)
- ✅ Comprehensive codebase review and security hardening
- ✅ Fixed 23 bare except clauses with specific exception handling
- ✅ Implemented API authentication and rate limiting
- ✅ Enhanced error handling and input validation
- ✅ Improved code organization and documentation
- ✅ CI/CD pipeline with automated testing and linting

See [CODEBASE_REVIEW_IMPROVEMENTS.md](CODEBASE_REVIEW_IMPROVEMENTS.md) for details.

## 📊 Features Overview

### Bond Types Supported

- **Zero Coupon**: Bonds with no periodic interest payments
- **Fixed Rate**: Bonds with fixed coupon payments
- **Floating Rate**: Bonds with variable coupon rates
- **Treasury**: Government-issued bonds
- **Corporate**: Corporate debt securities
- **Municipal**: Municipal bonds
- **High Yield**: High-risk, high-return bonds

### Valuation Methods

- **Discounted Cash Flow (DCF)**: Present value of future cash flows
- **Yield to Maturity (YTM)**: Internal rate of return using Newton-Raphson
- **Credit Spread Adjustment**: Risk-adjusted discount rates
- **ML-Enhanced Valuation**: Machine learning corrections
- **Option-Adjusted Spread (OAS)**: For bonds with embedded options
- **Multi-Curve Framework**: Separate discounting and forwarding curves

### Risk Management

- **Value at Risk (VaR)**: Historical, Parametric, and Monte Carlo methods
- **Credit Risk**: Default probabilities, recovery rates, expected loss
- **Liquidity Risk**: Bid-ask spreads, market depth analysis
- **Tail Risk**: Expected Shortfall (CVaR), extreme value analysis
- **Stress Testing**: Rate shocks, credit shocks, liquidity crises

### Machine Learning

- **Basic ML Adjuster**: Random Forest or Gradient Boosting
- **Enhanced ML Adjuster**: With hyperparameter tuning
- **Advanced ML Adjuster**: Ensemble methods with stacking
- **AutoML**: Automated model selection and tuning
- **Bayesian Optimization**: Efficient hyperparameter search
- **Drift Detection**: Model performance monitoring
- **Explainable AI**: Feature importance and prediction explanations

## 🔒 Security

BondTrader includes enterprise-grade security features:

### Security Features
- **API Authentication**: Optional Bearer token authentication for all endpoints
- **Rate Limiting**: Per-IP rate limiting to prevent abuse and DDoS attacks
- **CORS Protection**: Configurable CORS origins (no wildcard by default)
- **Input Validation**: Comprehensive validation for all API inputs
- **Secure Secrets Management**: Support for encrypted file storage, AWS Secrets Manager, and HashiCorp Vault
- **Error Handling**: Secure error handling that prevents information leakage
- **Audit Logging**: Comprehensive audit trails for compliance and traceability

### Security Best Practices
- No default passwords (requires environment variables)
- Secure file path validation (prevents directory traversal)
- Type-safe error handling (specific exceptions, not generic)
- Environment-based configuration (no hardcoded secrets)

**Reporting Security Issues**: Please report security vulnerabilities by emailing the maintainers. See [SECURITY.md](SECURITY.md) for more information.

**Recent Security Improvements** (January 2025):
- Fixed CORS wildcard vulnerability
- Removed all default passwords
- Implemented API key authentication
- Added rate limiting middleware
- Enhanced input validation
- Improved error handling to prevent information leakage

For security policy details, see [SECURITY.md](SECURITY.md). For recent security improvements, see [CODEBASE_REVIEW_IMPROVEMENTS.md](CODEBASE_REVIEW_IMPROVEMENTS.md).

## ⚠️ Disclaimer

This software is provided for **educational and demonstration purposes only**. It should not be used for actual trading decisions without:
- Thorough validation and testing
- Integration with verified market data
- Review by qualified financial professionals
- Proper risk management procedures

**Use at your own risk.** The authors and contributors are not responsible for any losses or damages resulting from use of this software.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/) for the dashboard
- Uses [scikit-learn](https://scikit-learn.org/) for machine learning
- Powered by [NumPy](https://numpy.org/) and [pandas](https://pandas.pydata.org/) for data processing
- Visualizations with [Plotly](https://plotly.com/python/)

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/SafetyMP/BondTrader/issues)
- **Discussions**: [GitHub Discussions](https://github.com/SafetyMP/BondTrader/discussions)
- **Repository**: https://github.com/SafetyMP/BondTrader

## 🗺️ Roadmap

See [ROADMAP.md](ROADMAP.md) for planned features and improvements.

## 📈 Value Proposition

### Why BondTrader?

**For Quantitative Analysts:**
- Comprehensive bond valuation models (DCF, OAS, multi-curve)
- Advanced risk metrics (VaR, credit risk, liquidity risk)
- Machine learning enhancements for price prediction
- Portfolio optimization strategies
- Backtesting and strategy validation

**For Developers:**
- Production-ready RESTful API with authentication
- Clean architecture with design patterns
- Comprehensive test coverage
- Well-documented codebase
- CI/CD pipeline ready

**For Organizations:**
- Security hardened (authentication, rate limiting, audit logging)
- Scalable architecture (Docker, microservices-ready)
- Compliance features (audit trails, error handling)
- Enterprise-grade error handling and monitoring
- Comprehensive documentation

### Competitive Advantages

- **Open Source**: Full access to source code and customization
- **Modern Stack**: Python 3.9+, FastAPI, Streamlit, scikit-learn
- **Production Ready**: Security, testing, and monitoring built-in
- **Well Documented**: Comprehensive guides and API reference
- **Active Development**: Regular updates and improvements

---

**Made with ❤️ for quantitative finance**

**Status**: ✅ Production Ready | 🔒 Security Hardened | 🧪 Well Tested | 📚 Fully Documented
