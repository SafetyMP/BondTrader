# BondTrader 📊

> A Python-based fixed income analytics platform for bond valuation, risk management, and arbitrage detection. Supports multiple bond types, ML-enhanced pricing, and quantitative risk metrics.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 📋 Overview

BondTrader is an open-source bond analytics platform that provides:

- **Core Bond Valuation**: DCF, YTM, duration, and convexity calculations for multiple bond types
- **Machine Learning Integration**: ML models to enhance pricing accuracy (Random Forest, XGBoost, LightGBM, AutoML)
- **Risk Management**: VaR calculations (Historical, Parametric, Monte Carlo), credit risk, liquidity risk, and tail risk metrics
- **Arbitrage Detection**: Automated identification of mispriced securities
- **REST API**: FastAPI-based API for programmatic access
- **Interactive Dashboard**: Streamlit-based dashboard for visualization and analysis

**Project Status**: Active development. Core features are implemented and functional. See [Current Limitations](#-current-limitations) below for areas still in development.

## 📑 Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
- [Current Limitations](#-current-limitations)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Testing](#-testing)
- [Documentation](#-documentation)
- [Contributing](#-contributing)
- [Security](#-security)
- [License](#-license)

## ✨ Features

### Core Capabilities

- **Bond Valuation**: DCF, YTM (Newton-Raphson), duration, convexity, and credit spread adjustments
- **Bond Types**: Zero Coupon, Fixed Rate, Floating Rate, Treasury, Corporate, Municipal, High Yield
- **Arbitrage Detection**: Configurable profit threshold detection for mispriced bonds
- **Risk Analytics**: VaR (3 methodologies), credit risk, liquidity risk, tail risk (CVaR)
- **ML-Enhanced Pricing**: Multiple model types with hyperparameter tuning and ensemble methods
- **Portfolio Analytics**: Portfolio optimization (Markowitz, Black-Litterman, risk parity)

### Advanced Features (Implementation Status Varies)

- **Option-Adjusted Spread (OAS)**: Basic implementation for bonds with embedded options
- **Multi-Curve Framework**: Yield curve modeling with discounting and forwarding curves
- **Factor Models**: PCA-based risk attribution
- **Backtesting**: Historical strategy validation framework
- **ML Operations**: MLflow integration for experiment tracking
- **Explainable AI**: SHAP values and feature importance

See [ROADMAP.md](ROADMAP.md) for detailed feature status and planned improvements.

## 🚀 Quick Start

### Installation

#### Option 1: Docker (Recommended)

```bash
git clone https://github.com/SafetyMP/BondTrader.git
cd BondTrader

# Configure environment
cp docker/.env.example docker/.env
# Edit docker/.env with your settings

# Start all services
make up
# Or: docker-compose -f docker/docker-compose.yml up -d
```

Access services:
- Dashboard: http://localhost:8501
- API: http://localhost:8000/docs
- MLflow: http://localhost:5000

#### Option 2: Local Installation

```bash
git clone https://github.com/SafetyMP/BondTrader.git
cd BondTrader

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Note**: The project has many dependencies. Installation may take several minutes. See [Current Limitations](#-current-limitations) for dependency management considerations.

### Usage Examples

#### Python Library

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
    maturity_date=datetime.now() + timedelta(days=1825),
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

print(f"Fair Value: ${fair_value:.2f}")
print(f"YTM: {ytm*100:.2f}%")
```

#### REST API

Start the API server:
```bash
python scripts/api_server.py
# Or: uvicorn scripts.api_server:app --reload
```

API available at `http://localhost:8000` with interactive docs at `/docs`.

**Example API Usage:**
```bash
# Health check
curl http://localhost:8000/health

# Create a bond (authentication required if enabled)
curl -X POST "http://localhost:8000/bonds" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "bond_id": "BOND-001",
    "bond_type": "CORPORATE",
    "face_value": 1000,
    "coupon_rate": 0.05,
    "maturity_date": "2029-12-31",
    "current_price": 950
  }'
```

#### Dashboard

```bash
streamlit run scripts/dashboard.py
```

Dashboard opens at `http://localhost:8501`.

See [API Reference](docs/api/API_REFERENCE.md) and [User Guide](docs/guides/USER_GUIDE.md) for detailed documentation.

## ⚠️ Current Limitations

This section provides an honest assessment of what's implemented, what's in progress, and what's planned.

### Implementation Status

**Fully Implemented:**
- Core bond valuation (DCF, YTM, duration, convexity)
- Multiple bond types (Fixed Rate, Zero Coupon, Treasury, Corporate, etc.)
- Basic ML models (Random Forest, XGBoost, LightGBM)
- VaR calculations (3 methodologies)
- REST API with authentication
- Streamlit dashboard
- Docker containerization

**Partially Implemented:**
- Floating rate bonds (enum exists, pricing logic incomplete)
- Option-Adjusted Spread (OAS) - basic implementation, needs enhancement
- Portfolio optimization - Markowitz implemented, Black-Litterman incomplete
- Factor models - PCA exists but needs full integration
- Backtesting - framework exists, needs historical data integration
- Alternative data pipeline - framework only, not production-ready

**Not Yet Implemented:**
- Real-time market data integration (Bloomberg, Reuters)
- Full regulatory compliance framework (audit trails exist but need enhancement)
- Comprehensive day count conventions (currently assumes 30/360 or ACT/365.25)
- Advanced credit risk models (Merton structural model, reduced-form models)
- Production-scale performance benchmarks
- Horizontal scaling architecture

### Known Issues

- **Dependencies**: Large dependency list (60+ packages) may cause conflicts. Some optional dependencies require system-level libraries (e.g., QuantLib requires C++).
- **Test Coverage**: Current test coverage is ~46% (target: 60%). Some modules have lower coverage than others.
- **Performance**: No comprehensive performance benchmarks. Performance characteristics not measured for large portfolios (>1000 bonds).
- **Data Validation**: Market data validation is basic. Users should validate data quality before use.
- **Model Risk**: ML models are trained on synthetic/historical data. Real-world performance may vary.
- **Documentation**: Some advanced features lack detailed usage examples.

### Production Readiness Considerations

Before deploying to production:

1. **Security Audit**: Conduct a security review. Authentication and rate limiting exist but haven't undergone professional security audit.
2. **Performance Testing**: Benchmark with your expected workload and data volumes.
3. **Data Integration**: Integrate with your market data providers. Current implementations are examples.
4. **Regulatory Compliance**: Review compliance requirements. Audit trails exist but may need enhancement for your jurisdiction.
5. **Monitoring**: Set up monitoring and alerting. Prometheus/Grafana configs exist but need customization.
6. **Backup & Recovery**: Implement backup strategies for PostgreSQL data and trained ML models.

### Scaling Limitations

- **Single-Node Design**: Current architecture assumes single-node deployment. Horizontal scaling requires architectural changes.
- **Database**: PostgreSQL setup is basic. Production deployments should consider connection pooling, read replicas, and tuning.
- **ML Model Serving**: Models are loaded in-memory. Large model serving requires additional infrastructure.
- **API Rate Limits**: Rate limiting is per-IP in-memory. Distributed rate limiting (Redis-based) is not yet implemented.

See [ROADMAP.md](ROADMAP.md) for planned improvements and [docs/development/COMPETITIVE_ANALYSIS.md](docs/development/COMPETITIVE_ANALYSIS.md) for detailed feature gaps.

## 📁 Project Structure

```
BondTrader/
├── bondtrader/              # Main package
│   ├── core/               # Core bond trading logic
│   ├── ml/                 # Machine Learning pipeline
│   ├── risk/               # Risk management
│   ├── analytics/          # Advanced analytics
│   ├── data/               # Data management
│   └── utils/              # Utilities (auth, caching, etc.)
├── scripts/                # Executable scripts (dashboard, API server)
├── tests/                  # Test suite
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── smoke/             # Smoke tests
├── docs/                   # Documentation
└── docker/                 # Docker configuration
```

See [docs/ORGANIZATION.md](docs/ORGANIZATION.md) for detailed structure.

## 🔧 Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
cp env.example .env
```

**Essential Configuration:**
```env
# API Security
API_KEY=your_secret_api_key_here
REQUIRE_API_KEY=false  # Set to true to enable authentication
CORS_ALLOWED_ORIGINS=http://localhost:8000,http://localhost:8501

# Rate Limiting
API_RATE_LIMIT=100  # Requests per window
API_RATE_LIMIT_WINDOW=60  # Window in seconds

# External Data Sources (Optional)
FRED_API_KEY=your_fred_api_key  # For Federal Reserve data
FINRA_API_KEY=your_finra_api_key  # For FINRA market data

# Application
DEFAULT_RFR=0.03  # Default risk-free rate
ML_MODEL_TYPE=random_forest
BOND_DB_PATH=./data/bonds.db
```

**Note**: The system works with simulated data without API keys. External API keys are only needed for live market data.

## 🧪 Testing

### Running Tests

```bash
# All tests
pytest tests/ -v

# Unit tests only
pytest tests/unit -m unit -v

# Integration tests
pytest tests/integration -m integration -v

# With coverage report
pytest tests/ -v --cov=bondtrader --cov-report=html
```

### Test Coverage

- **Current**: ~46% (varies by module)
- **Target**: 60%
- **CI Threshold**: 55% (enforced in CI/CD)

Coverage report available after running tests: `open htmlcov/index.html`

See [tests/README.md](tests/README.md) for detailed testing documentation.

### CI/CD

Automated CI runs on pushes and pull requests:
- Tests across Python 3.9, 3.10, 3.11
- Code formatting (black, isort)
- Linting (flake8)
- Type checking (mypy, gradual)
- Security scanning (safety, pip-audit)
- Coverage enforcement (55% threshold)

See [.github/workflows/ci.yml](.github/workflows/ci.yml) for configuration.

## 📖 Documentation

Documentation is available in the [`docs/`](docs/) directory:

### Quick Links
- **[Quick Start Guide](docs/guides/QUICK_START_GUIDE.md)** - Getting started
- **[User Guide](docs/guides/USER_GUIDE.md)** - Complete usage guide
- **[API Reference](docs/api/API_REFERENCE.md)** - API documentation
- **[Architecture](docs/development/ARCHITECTURE.md)** - System architecture
- **[Contributing Guide](CONTRIBUTING.md)** - How to contribute

### Additional Documentation
- User guides (training data, historical data, drift detection)
- Development docs (architecture, competitive analysis)
- Technical analysis reports

See [docs/README.md](docs/README.md) for complete index.

## 🤝 Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

1. Fork and clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```
4. Install pre-commit hooks (optional):
   ```bash
   pre-commit install
   ```

### Code Quality

Before submitting a PR:
- All tests pass: `pytest tests/ -v`
- Code formatted: `black bondtrader/ scripts/ tests/`
- Imports sorted: `isort bondtrader/ scripts/ tests/`
- No critical linting errors: `flake8 bondtrader/ scripts/ tests/ --select=E9,F63,F7,F82`
- Type hints added where appropriate
- Documentation updated

## 🔒 Security

### Security Features

- **API Authentication**: Bearer token authentication (optional, configurable)
- **Rate Limiting**: Per-IP rate limiting (in-memory, configurable)
- **CORS Protection**: Configurable CORS policies
- **Input Validation**: Basic input validation and path traversal prevention
- **Secrets Management**: Support for encrypted file storage, AWS Secrets Manager, HashiCorp Vault
- **Audit Logging**: Basic audit trails for operations

### Security Considerations

- Authentication is optional by default. Enable `REQUIRE_API_KEY=true` for production.
- Rate limiting is per-IP in-memory. For distributed deployments, implement Redis-based rate limiting.
- Secrets management requires configuration. No default credentials.
- Security features exist but haven't undergone professional security audit.

**Security Disclosure**: Report vulnerabilities through [SECURITY.md](SECURITY.md).

## 🏗️ Architecture & Design Patterns

BondTrader uses common software design patterns:

- **Service Layer**: Business logic separation
- **Repository Pattern**: Data access abstraction
- **Result Pattern**: Type-safe error handling
- **Circuit Breaker**: Resilience for external dependencies
- **Factory Pattern**: Object creation abstraction

See [docs/development/ARCHITECTURE.md](docs/development/ARCHITECTURE.md) for details.

## 📊 Technology Stack

- **Python**: 3.9+
- **Web Framework**: FastAPI (REST API), Streamlit (Dashboard)
- **ML Libraries**: scikit-learn, XGBoost, LightGBM, CatBoost, Optuna, MLflow
- **Data**: pandas, NumPy, SciPy
- **Database**: PostgreSQL, SQLAlchemy
- **Caching**: Redis
- **Visualization**: Plotly
- **Testing**: pytest, pytest-cov

**Note**: Large dependency list (60+ packages). See [Current Limitations](#-current-limitations) for dependency considerations.

## 📄 License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/SafetyMP/BondTrader/issues)
- **Discussions**: [GitHub Discussions](https://github.com/SafetyMP/BondTrader/discussions)
- **Repository**: https://github.com/SafetyMP/BondTrader

## 🗺️ Roadmap

See [ROADMAP.md](ROADMAP.md) for planned features and improvements.

## ⚠️ Disclaimer

BondTrader is provided as-is for educational and research purposes. Before using in production:

- Thoroughly test with your data and use cases
- Validate against your market data sources
- Review by qualified financial and technology professionals
- Ensure compliance with applicable regulations
- Implement appropriate risk management procedures

The authors and contributors are not liable for any losses or damages resulting from use of this software.

---

**BondTrader** - Fixed income analytics platform for bond valuation, risk management, and arbitrage detection.
