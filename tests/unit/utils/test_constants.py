"""
Tests for constants module
"""

import pytest

from bondtrader.utils.constants import (
    DEFAULT_PROBABILITIES,
    RECOVERY_RATES_ENHANCED,
    RECOVERY_RATES_STANDARD,
    get_default_probability,
    get_recovery_rate_enhanced,
    get_recovery_rate_standard,
)


@pytest.mark.unit
class TestConstants:
    """Test constants values"""

    def test_default_probabilities(self):
        """Test default probabilities dictionary"""
        assert isinstance(DEFAULT_PROBABILITIES, dict)
        assert "AAA" in DEFAULT_PROBABILITIES
        assert DEFAULT_PROBABILITIES["AAA"] < DEFAULT_PROBABILITIES["BBB"]

    def test_recovery_rates_standard(self):
        """Test standard recovery rates"""
        assert isinstance(RECOVERY_RATES_STANDARD, dict)
        assert "AAA" in RECOVERY_RATES_STANDARD
        assert 0 < RECOVERY_RATES_STANDARD["AAA"] < 1

    def test_recovery_rates_enhanced(self):
        """Test enhanced recovery rates"""
        assert isinstance(RECOVERY_RATES_ENHANCED, dict)
        assert "AAA" in RECOVERY_RATES_ENHANCED
        assert 0 < RECOVERY_RATES_ENHANCED["AAA"] < 1

    def test_get_default_probability(self):
        """Test get_default_probability function"""
        prob = get_default_probability("AAA")
        assert isinstance(prob, float)
        assert 0 <= prob <= 1
        assert prob < get_default_probability("BBB")

    def test_get_default_probability_unknown(self):
        """Test get_default_probability with unknown rating"""
        prob = get_default_probability("UNKNOWN")
        assert isinstance(prob, float)
        assert prob == 0.020  # Default value

    def test_get_recovery_rate_standard(self):
        """Test get_recovery_rate_standard function"""
        rate = get_recovery_rate_standard("AAA")
        assert isinstance(rate, float)
        assert 0 < rate < 1

    def test_get_recovery_rate_enhanced(self):
        """Test get_recovery_rate_enhanced function"""
        rate = get_recovery_rate_enhanced("AAA")
        assert isinstance(rate, float)
        assert 0 < rate < 1

    def test_get_default_probability_all_ratings(self):
        """Test getting default probabilities for all ratings"""
        for rating in DEFAULT_PROBABILITIES.keys():
            prob = get_default_probability(rating)
            assert 0 <= prob <= 1

    def test_get_recovery_rate_all_ratings_standard(self):
        """Test getting recovery rates for all ratings (standard)"""
        for rating in RECOVERY_RATES_STANDARD.keys():
            rate = get_recovery_rate_standard(rating)
            assert 0 <= rate <= 1

    def test_get_recovery_rate_all_ratings_enhanced(self):
        """Test getting recovery rates for all ratings (enhanced)"""
        for rating in RECOVERY_RATES_ENHANCED.keys():
            rate = get_recovery_rate_enhanced(rating)
            assert 0 <= rate <= 1
