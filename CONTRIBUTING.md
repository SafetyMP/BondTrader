# Contributing to BondTrader

First off, thank you for considering contributing to BondTrader! 🎉

This document provides guidelines and instructions for contributing to the project.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inspiring community for all. Please be respectful and considerate of others.

### Expected Behavior

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Give constructive feedback
- Focus on what is best for the community

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the [issue tracker](https://github.com/yourusername/BondTrader/issues) to ensure the bug hasn't already been reported.

**Good bug reports include:**
- Clear, descriptive title
- Steps to reproduce the issue
- Expected vs. actual behavior
- Environment details (OS, Python version)
- Relevant code snippets or error messages

### Suggesting Enhancements

Enhancement suggestions are tracked as [GitHub issues](https://github.com/yourusername/BondTrader/issues). When creating an enhancement suggestion:

- Use a clear and descriptive title
- Provide detailed explanation of the proposed enhancement
- Explain why this enhancement would be useful
- Include examples if applicable

### Pull Requests

- Fill in the required template
- Do not include issue numbers in the PR title
- Include screenshots and animated GIFs if applicable
- Follow the Python style guide (see below)
- Include tests for new functionality
- Update documentation as needed

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment (recommended)

### Setting Up Development Environment

1. **Fork and clone the repository**:
```bash
git clone https://github.com/yourusername/BondTrader.git
cd BondTrader
```

2. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
pip install pytest pytest-cov black isort flake8 mypy pre-commit
```

4. **Install pre-commit hooks**:
```bash
pre-commit install
```

5. **Run tests to verify setup**:
```bash
pytest tests/ -v
```

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line length**: 127 characters (allows for better readability)
- **Indentation**: 4 spaces
- **Imports**: Use absolute imports, sorted with `isort`

### Code Formatting

We use automated tools for code formatting:

1. **Black** for code formatting:
```bash
black bondtrader/ scripts/ tests/
```

2. **isort** for import sorting:
```bash
isort bondtrader/ scripts/ tests/
```

3. **Pre-commit hooks** (automatically run before commits):
```bash
pre-commit run --all-files
```

### Type Hints

- Add type hints to all function signatures
- Use `typing` module for complex types
- Use `Optional` for nullable values
- Use `List`, `Dict`, `Tuple` for collections

Example:
```python
from typing import List, Dict, Optional

def calculate_fair_value(
    self,
    bond: Bond,
    required_yield: Optional[float] = None
) -> float:
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def calculate_yield_to_maturity(
    self,
    bond: Bond,
    market_price: Optional[float] = None
) -> float:
    """
    Calculate Yield to Maturity using Newton-Raphson method.
    
    Args:
        bond: Bond object to calculate YTM for
        market_price: Current market price (uses bond.current_price if None)
        
    Returns:
        YTM as decimal (e.g., 0.05 for 5%)
        
    Raises:
        ValueError: If bond has invalid maturity date or negative values
        TypeError: If bond is not a Bond instance
        
    Example:
        >>> bond = Bond(...)
        >>> valuator = BondValuator()
        >>> ytm = valuator.calculate_yield_to_maturity(bond)
        >>> print(f"YTM: {ytm*100:.2f}%")
    """
    ...
```

## Testing

### Writing Tests

- Write tests for all new functionality
- Use descriptive test names
- Follow the pattern: `test_<functionality>_<scenario>`
- Use pytest fixtures for common setup

Example:
```python
def test_calculate_fair_value_positive_ytm(sample_bond, valuator):
    """Test fair value calculation with positive yield"""
    fair_value = valuator.calculate_fair_value(sample_bond)
    assert fair_value > 0
    assert isinstance(fair_value, float)
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_bond_valuation.py -v

# Run with coverage
pytest tests/ -v --cov=bondtrader --cov-report=html

# Run specific test
pytest tests/test_bond_valuation.py::test_fair_value_calculation -v
```

### Test Coverage

We aim for 70-80% test coverage. Check coverage:
```bash
pytest tests/ --cov=bondtrader --cov-report=term-missing
```

## Submitting Changes

### Commit Messages

Follow these guidelines for commit messages:

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests after the first line

Example:
```
Add configuration management system

Implements centralized configuration with environment variable support.
Includes validation and singleton pattern for global access.

Closes #123
```

### Branch Naming

Use descriptive branch names:
- `feature/add-new-valuator`
- `bugfix/fix-ytm-calculation`
- `docs/update-readme`
- `refactor/improve-error-handling`

## Pull Request Process

1. **Update documentation** if you've changed functionality
2. **Add tests** for new functionality
3. **Ensure tests pass** (`pytest tests/`)
4. **Run code formatters** (`black` and `isort`)
5. **Update CHANGELOG.md** with your changes
6. **Request review** from maintainers

### Pull Request Template

Your PR should include:
- Clear description of changes
- Reference to related issues
- Tests added/updated
- Documentation updated
- Screenshots (if UI changes)

### Review Process

- All PRs require at least one approval
- Address review comments promptly
- Maintainers may request changes
- PRs are merged via squash commit

## 📚 Additional Resources

- [Python Documentation](https://docs.python.org/3/)
- [Pytest Documentation](https://docs.pytest.org/)
- [Black Code Style](https://black.readthedocs.io/)
- [Type Hints Documentation](https://docs.python.org/3/library/typing.html)

## 🙏 Thank You!

Your contributions make BondTrader better for everyone. Thank you for taking the time to contribute!
