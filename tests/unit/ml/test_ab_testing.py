"""
Tests for A/B testing module
"""

import pytest

from bondtrader.ml.ab_testing import ABTestFramework, ABTestConfig, Variant


@pytest.mark.unit
class TestABTestFramework:
    """Test ABTestFramework functionality"""

    @pytest.fixture
    def config(self):
        """Create A/B test config"""
        return ABTestConfig(
            test_name="test_ab",
            control_model_name="control",
            treatment_model_name="treatment",
        )

    @pytest.fixture
    def framework(self, config):
        """Create A/B test framework"""
        # Use mock models
        mock_control = None
        mock_treatment = None
        return ABTestFramework(
            config=config,
            control_model=mock_control,
            treatment_model=mock_treatment,
        )

    def test_framework_init(self, framework):
        """Test framework initialization"""
        assert framework is not None
        assert framework.config is not None
        assert framework.is_running is False

    def test_assign_variant(self, framework):
        """Test variant assignment"""
        variant = framework.assign_variant("test_bond_1")
        assert variant in [Variant.CONTROL, Variant.TREATMENT]

    def test_get_current_stats(self, framework):
        """Test getting current stats"""
        stats = framework.get_current_stats()
        assert isinstance(stats, dict)
