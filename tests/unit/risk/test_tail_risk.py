"""
Unit tests for tail risk analysis
"""

from datetime import datetime, timedelta

import pytest

from bondtrader.core.bond_models import Bond, BondType
from bondtrader.core.bond_valuation import BondValuator
from bondtrader.risk.tail_risk import TailRiskAnalyzer


@pytest.mark.unit
class TestTailRiskAnalyzer:
    """Test TailRiskAnalyzer class"""

    @pytest.fixture
    def valuator(self):
        """Create valuator"""
        return BondValuator(risk_free_rate=0.03)

    @pytest.fixture
    def tail_risk(self, valuator):
        """Create tail risk analyzer"""
        return TailRiskAnalyzer(valuator=valuator)

    @pytest.fixture
    def sample_bonds(self):
        """Create sample bonds"""
        now = datetime.now()
        return [
            Bond(
                bond_id=f"BOND-{i:03d}",
                bond_type=BondType.CORPORATE,
                face_value=1000,
                coupon_rate=5.0 + i * 0.1,
                maturity_date=now + timedelta(days=1825 + i * 30),
                issue_date=now - timedelta(days=365),
                current_price=950 + i * 5,
                credit_rating="BBB",
                issuer=f"Corp {i}",
                frequency=2,
            )
            for i in range(10)
        ]

    def test_tail_risk_creation(self, tail_risk):
        """Test creating tail risk analyzer"""
        assert tail_risk.valuator is not None
        assert tail_risk.risk_manager is not None

    def test_calculate_cvar(self, tail_risk, sample_bonds):
        """Test calculating Conditional Value at Risk"""
        weights = [1.0 / len(sample_bonds)] * len(sample_bonds)

        result = tail_risk.calculate_cvar(sample_bonds, weights, confidence_level=0.95)
        assert "cvar_value" in result
        assert "cvar_pct" in result
        assert "var_value" in result
        assert result["cvar_value"] >= 0

    def test_calculate_cvar_different_confidence(self, tail_risk, sample_bonds):
        """Test CVaR with different confidence level"""
        weights = [1.0 / len(sample_bonds)] * len(sample_bonds)

        result = tail_risk.calculate_cvar(sample_bonds, weights, confidence_level=0.99)
        assert "cvar_value" in result
        assert result["cvar_pct"] >= 0

    def test_calculate_expected_shortfall(self, tail_risk, sample_bonds):
        """Test calculating Expected Shortfall"""
        weights = [1.0 / len(sample_bonds)] * len(sample_bonds)

        result = tail_risk.calculate_expected_shortfall(sample_bonds, weights)
        assert "expected_shortfall" in result
        assert "es_pct" in result
        assert result["expected_shortfall"] >= 0

    def test_calculate_tail_expectation(self, tail_risk, sample_bonds):
        """Test calculating tail expectation"""
        weights = [1.0 / len(sample_bonds)] * len(sample_bonds)

        result = tail_risk.calculate_tail_expectation(sample_bonds, weights, tail_threshold=0.05)
        assert "tail_expectation" in result
        assert "tail_pct" in result

    def test_calculate_cvar_single_bond(self, tail_risk, sample_bonds):
        """Test CVaR with single bond"""
        bond = sample_bonds[0]
        result = tail_risk.calculate_cvar([bond], [1.0])
        assert "cvar_value" in result
        assert result["cvar_value"] >= 0

    def test_cvar_greater_than_var(self, tail_risk, sample_bonds):
        """Test that CVaR is typically greater than VaR"""
        weights = [1.0 / len(sample_bonds)] * len(sample_bonds)

        result = tail_risk.calculate_cvar(sample_bonds, weights, confidence_level=0.95)
        # CVaR should be >= VaR (average of tail losses)
        assert result["cvar_value"] >= result["var_value"]
