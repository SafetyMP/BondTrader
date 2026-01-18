"""
Unit tests for constants module
"""

import os
import sys

import pytest

pytestmark = pytest.mark.unit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bondtrader.utils.constants import (
    DEFAULT_PROBABILITIES,
    RECOVERY_RATES_ENHANCED,
    RECOVERY_RATES_STANDARD,
    get_default_probability,
    get_recovery_rate_enhanced,
    get_recovery_rate_standard,
)


def test_get_default_probability_known_ratings():
    """Test default probability lookup for known ratings"""
    prob_aaa = get_default_probability("AAA")
    prob_bbb = get_default_probability("BBB")
    prob_b = get_default_probability("B")
    prob_d = get_default_probability("D")
    
    assert prob_aaa < prob_bbb < prob_b < prob_d
    assert prob_aaa > 0
    assert prob_d == 1.0  # Default should be 100%


def test_get_default_probability_unknown_rating():
    """Test default probability for unknown rating"""
    prob = get_default_probability("UNKNOWN")
    
    assert prob == 0.020  # Default fallback


def test_get_default_probability_case_insensitive():
    """Test default probability is case insensitive"""
    prob_upper = get_default_probability("AAA")
    prob_lower = get_default_probability("aaa")
    
    assert prob_upper == prob_lower


def test_get_recovery_rate_standard():
    """Test standard recovery rate lookup"""
    rate_aaa = get_recovery_rate_standard("AAA")
    rate_bbb = get_recovery_rate_standard("BBB")
    rate_b = get_recovery_rate_standard("B")
    
    assert rate_aaa > rate_bbb > rate_b
    assert 0 < rate_aaa <= 1
    assert 0 < rate_bbb <= 1


def test_get_recovery_rate_enhanced():
    """Test enhanced recovery rate lookup"""
    rate_aaa = get_recovery_rate_enhanced("AAA")
    rate_bbb = get_recovery_rate_enhanced("BBB")
    
    assert rate_aaa > rate_bbb
    assert 0 < rate_aaa <= 1


def test_recovery_rate_standard_vs_enhanced():
    """Test that standard and enhanced rates may differ"""
    rate_std = get_recovery_rate_standard("BBB")
    rate_enh = get_recovery_rate_enhanced("BBB")
    
    # Both should be valid rates
    assert 0 < rate_std <= 1
    assert 0 < rate_enh <= 1


def test_default_probability_consistency():
    """Test default probability values are consistent"""
    # Higher rated bonds should have lower default probability
    prob_aaa = get_default_probability("AAA")
    prob_aa = get_default_probability("AA")
    prob_a = get_default_probability("A")
    
    assert prob_aaa < prob_aa < prob_a


def test_recovery_rate_consistency():
    """Test recovery rate values are consistent"""
    # Higher rated bonds should have higher recovery rates
    rate_aaa = get_recovery_rate_standard("AAA")
    rate_aa = get_recovery_rate_standard("AA")
    rate_a = get_recovery_rate_standard("A")
    
    assert rate_aaa >= rate_aa >= rate_a


def test_constants_dictionaries_exist():
    """Test that constant dictionaries exist"""
    assert DEFAULT_PROBABILITIES is not None
    assert RECOVERY_RATES_STANDARD is not None
    assert RECOVERY_RATES_ENHANCED is not None
    assert len(DEFAULT_PROBABILITIES) > 0
