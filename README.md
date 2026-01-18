# BondTrader 📊

> A comprehensive Python application for valuing bonds, detecting arbitrage opportunities, and analyzing bond market data using machine learning and financial modeling.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

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
│   ├── ml/                         # Machine Learning modules
│   ├── risk/                       # Risk management modules
│   ├── analytics/                  # Analytics and advanced features
│   ├── data/                       # Data handling modules
│   ├── utils/                      # Utility functions
│   └── config.py                   # Configuration management
│
├── scripts/                        # Executable scripts
│   ├── dashboard.py                # Streamlit dashboard
│   ├── train_all_models.py         # Model training
│   └── evaluate_models.py          # Model evaluation
│
├── tests/                          # Unit tests
│   ├── conftest.py                 # Shared fixtures
│   ├── test_bond_valuation.py
│   ├── test_arbitrage.py
│   ├── test_arbitrage_detector.py
│   └── test_config.py
│
├── docs/                           # Documentation
│   ├── guides/                     # User guides
│   ├── api/                        # API documentation
│   └── development/                # Development docs
│
├── requirements.txt                # Python dependencies
├── setup.py                        # Package setup
├── LICENSE                         # License file
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

- **[User Guide](docs/guides/USER_GUIDE.md)** - Getting started and usage
- **[API Reference](docs/api/API_REFERENCE.md)** - Complete API documentation
- **[Development Guide](docs/development/DEVELOPMENT.md)** - Contributing and development
- **[Architecture](docs/development/ARCHITECTURE.md)** - System architecture overview

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

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Quick Contribution Checklist

- [ ] Fork the repository
- [ ] Create a feature branch (`git checkout -b feature/amazing-feature`)
- [ ] Make your changes
- [ ] Add tests for new functionality
- [ ] Ensure all tests pass (`pytest tests/`)
- [ ] Run code formatters (`black bondtrader/` and `isort bondtrader/`)
- [ ] Commit your changes (`git commit -m 'Add amazing feature'`)
- [ ] Push to the branch (`git push origin feature/amazing-feature`)
- [ ] Open a Pull Request

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

This is a demonstration/training system using synthetic data. For production use:
- Integrate with real market data feeds
- Implement proper authentication and authorization
- Add audit trails and compliance features
- Review and secure all API endpoints

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
