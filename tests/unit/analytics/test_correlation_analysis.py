"""
Unit tests for correlation analysis module
"""

import os
import sys

import pytest

pytestmark = pytest.mark.unit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bondtrader.analytics.correlation_analysis import CorrelationAnalyzer
from fixtures.bond_factory import create_multiple_bonds


@pytest.fixture
def correlation_analyzer():
    """Create correlation analyzer instance"""
    return CorrelationAnalyzer()


@pytest.fixture
def portfolio_bonds():
    """Create bonds for correlation analysis"""
    return create_multiple_bonds(count=5)


def test_correlation_analyzer_initialization():
    """Test correlation analyzer initialization"""
    analyzer = CorrelationAnalyzer()
    assert analyzer is not None


def test_calculate_correlation_matrix(correlation_analyzer, portfolio_bonds):
    """Test correlation matrix calculation"""
    result = correlation_analyzer.calculate_correlation_matrix(portfolio_bonds)
    
    assert "correlation_matrix" in result
    assert "bond_ids" in result
    matrix = result["correlation_matrix"]
    assert isinstance(matrix, list)
    assert len(matrix) == len(portfolio_bonds)


def test_calculate_covariance_matrix(correlation_analyzer, portfolio_bonds):
    """Test covariance matrix calculation"""
    result = correlation_analyzer.calculate_covariance_matrix(portfolio_bonds)
    
    assert "covariance_matrix" in result
    assert len(result["covariance_matrix"]) == len(portfolio_bonds)
