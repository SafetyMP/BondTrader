# BondTrader 📊

> A comprehensive Python application for valuing bonds, detecting arbitrage opportunities, and analyzing bond market data using machine learning and financial modeling.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 📑 Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
  - [Installation](#installation)
  - [Usage](#usage)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Documentation](#-documentation)
- [Testing](#-testing)
- [Contributing](#-contributing)
- [Security](#-security)
- [License](#-license)
- [Support](#-support)

## ✨ Features

### 🎯 Core Functionality

- **Bond Valuation**: Calculate fair values for various bond types (Zero Coupon, Fixed Rate, Treasury, Corporate, High Yield, Floating Rate)
- **Bond Classification**: Automatic classification of bonds based on characteristics
- **Machine Learning Adjustments**: ML-powered price adjustments using Random Forest, Gradient Boosting, or AutoML
- **Arbitrage Detection**: Identify mispriced bonds and arbitrage opportunities
- **Interactive Dashboard**: Streamlit-based dashboard with visualizations and comparisons
- **Risk Management**: Comprehensive risk analysis including VaR, credit risk, liquidity risk, and tail risk
- **Portfolio Optimization**: Markowitz optimization, Black-Litterman model, and risk parity strategies

### 📊 Advanced Capabilities

- **Multi-Curve Framework**: Separate discounting and forwarding curves
- **Option-Adjusted Spread (OAS)**: Pricing for bonds with embedded options
- **Key Rate Duration**: Sensitivity analysis at key yield curve points
- **Factor Models**: PCA-based factor extraction and risk attribution
- **Backtesting Engine**: Historical performance validation and strategy testing
- **Execution Strategies**: Market impact modeling and optimal execution
- **Explainable AI**: Feature importance analysis and prediction explanations
- **Drift Detection**: Model performance monitoring and automatic retraining

## 🚀 Quick Start

### Installation

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

#### Using the Python API

```python
from bondtrader.core import Bond, BondType, BondValuator, ArbitrageDetector
from bondtrader.ml import MLBondAdjuster
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

# Check for arbitrage
detector = ArbitrageDetector(valuator=valuator)
opportunities = detector.find_arbitrage_opportunities([bond])

print(f"Fair Value: ${fair_value:.2f}")
print(f"YTM: {ytm*100:.2f}%")
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

The system supports optional API keys for external data sources. Copy `.env.example` to `.env` and add your keys:

```bash
cp .env.example .env
```

Example `.env`:
```env
# FRED API Key (Optional - for real market data)
FRED_API_KEY=your_api_key_here

# Configuration
DEFAULT_RFR=0.03
ML_MODEL_TYPE=random_forest
```

**Note**: The system works without API keys using simulated data. API keys are only needed for live market data integration.

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

### User Documentation
- **[Quick Start Guide](docs/guides/QUICK_START_GUIDE.md)** - Quick introduction and setup
- **[User Guide](docs/guides/USER_GUIDE.md)** - Complete usage guide
- **[Training Data Guide](docs/guides/TRAINING_DATA.md)** - Generating training datasets
- **[Evaluation Dataset Guide](docs/guides/EVALUATION_DATASET.md)** - Creating evaluation datasets

### Developer Documentation
- **[API Reference](docs/api/API_REFERENCE.md)** - Complete API documentation
- **[Architecture](docs/development/ARCHITECTURE.md)** - System architecture overview
- **[Contributing Guide](CONTRIBUTING.md)** - How to contribute
- **[Codebase Organization](docs/ORGANIZATION.md)** - Project structure

### Additional Resources
- **[Changelog](CHANGELOG.md)** - Version history and changes
- **[Roadmap](ROADMAP.md)** - Planned features and improvements

For a complete overview, see the [Documentation Index](docs/README.md).

## 🧪 Testing

Run the test suite:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ -v --cov=bondtrader --cov-report=html
```

Run specific test file:
```bash
pytest tests/test_bond_valuation.py -v
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

**Quick start:**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Ensure all tests pass (`pytest tests/`)
5. Run code formatters (`black bondtrader/` and `isort bondtrader/`)
6. Submit a Pull Request

For more information, see [CONTRIBUTING.md](CONTRIBUTING.md) and our [Code of Conduct](CODE_OF_CONDUCT.md).

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

**Reporting Security Issues**: Please report security vulnerabilities by emailing the maintainers. See [SECURITY.md](SECURITY.md) for more information.

**Security Considerations**: This is a demonstration/training system using synthetic data. For production use:
- Integrate with real market data feeds
- Implement proper authentication and authorization
- Add audit trails and compliance features
- Review and secure all API endpoints

For security policy details, see [SECURITY.md](SECURITY.md).

## ⚠️ Disclaimer

This software is provided for **educational and demonstration purposes only**. It should not be used for actual trading decisions without:
- Thorough validation and testing
- Integration with verified market data
- Review by qualified financial professionals
- Proper risk management procedures

**Use at your own risk.** The authors and contributors are not responsible for any losses or damages resulting from use of this software.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

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

---

**Made with ❤️ for quantitative finance**
