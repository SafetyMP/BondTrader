"""
Unit tests for enhanced liquidity risk
"""

import pytest
from datetime import datetime, timedelta

from bondtrader.core.bond_models import Bond, BondType
from bondtrader.core.bond_valuation import BondValuator
from bondtrader.risk.liquidity_risk_enhanced import LiquidityRiskEnhanced


@pytest.mark.unit
class TestLiquidityRiskEnhanced:
    """Test LiquidityRiskEnhanced class"""

    @pytest.fixture
    def valuator(self):
        """Create valuator"""
        return BondValuator(risk_free_rate=0.03)

    @pytest.fixture
    def liquidity_risk(self, valuator):
        """Create liquidity risk analyzer"""
        return LiquidityRiskEnhanced(valuator=valuator)

    @pytest.fixture
    def sample_bond(self):
        """Create sample bond"""
        now = datetime.now()
        return Bond(
            bond_id="TEST-001",
            bond_type=BondType.CORPORATE,
            face_value=1000,
            coupon_rate=5.0,
            maturity_date=now + timedelta(days=1825),
            issue_date=now - timedelta(days=365),
            current_price=950,
            credit_rating="BBB",
            issuer="Test Corp",
            frequency=2,
        )

    def test_liquidity_risk_creation(self, liquidity_risk):
        """Test creating liquidity risk analyzer"""
        assert liquidity_risk.valuator is not None

    def test_calculate_bid_ask_spread(self, liquidity_risk, sample_bond):
        """Test calculating bid-ask spread"""
        result = liquidity_risk.calculate_bid_ask_spread(sample_bond)
        assert "bid_price" in result
        assert "ask_price" in result
        assert "spread_bps" in result
        assert "spread_pct" in result
        assert result["bid_price"] < result["ask_price"]

    def test_calculate_bid_ask_spread_with_base_spread(self, liquidity_risk, sample_bond):
        """Test calculating bid-ask spread with custom base spread"""
        result = liquidity_risk.calculate_bid_ask_spread(sample_bond, base_spread=50)  # 50 bps
        assert result["spread_bps"] == 50
        assert result["bid_price"] < sample_bond.current_price
        assert result["ask_price"] > sample_bond.current_price

    def test_estimate_spread_from_characteristics(self, liquidity_risk, sample_bond):
        """Test estimating spread from bond characteristics"""
        spread = liquidity_risk._estimate_spread_from_characteristics(sample_bond)
        assert spread > 0
        assert isinstance(spread, float)

    def test_estimate_spread_high_quality(self, liquidity_risk):
        """Test spread estimation for high quality bond"""
        now = datetime.now()
        aaa_bond = Bond(
            bond_id="AAA-001",
            bond_type=BondType.TREASURY,
            face_value=1000,
            coupon_rate=3.0,
            maturity_date=now + timedelta(days=1825),
            issue_date=now - timedelta(days=365),
            current_price=980,
            credit_rating="AAA",
            issuer="US Treasury",
            frequency=2,
        )
        spread = liquidity_risk._estimate_spread_from_characteristics(aaa_bond)
        assert spread > 0  # Should have some spread

    def test_calculate_lvar(self, liquidity_risk, sample_bond):
        """Test calculating Liquidity-adjusted VaR"""
        bonds = [sample_bond]
        weights = [1.0]
        
        result = liquidity_risk.calculate_lvar(bonds, weights, confidence_level=0.95)
        assert "lvar_value" in result
        assert "lvar_pct" in result
        assert "var_value" in result
        assert "liquidity_adjustment" in result

    def test_calculate_lvar_multiple_bonds(self, liquidity_risk, sample_bond):
        """Test LVaR with multiple bonds"""
        now = datetime.now()
        bond2 = Bond(
            bond_id="TEST-002",
            bond_type=BondType.CORPORATE,
            face_value=1000,
            coupon_rate=6.0,
            maturity_date=now + timedelta(days=1825),
            issue_date=now - timedelta(days=365),
            current_price=960,
            credit_rating="BB",
            issuer="Test Corp 2",
            frequency=2,
        )
        
        bonds = [sample_bond, bond2]
        weights = [0.6, 0.4]
        
        result = liquidity_risk.calculate_lvar(bonds, weights)
        assert "lvar_value" in result
        assert result["lvar_value"] >= 0

    def test_estimate_market_depth(self, liquidity_risk, sample_bond):
        """Test estimating market depth"""
        result = liquidity_risk.estimate_market_depth(sample_bond)
        assert "depth_notional" in result
        assert "depth_bonds" in result
        assert "liquidity_score" in result
        assert 0 <= result["liquidity_score"] <= 1

    def test_assess_liquidity_risk(self, liquidity_risk, sample_bond):
        """Test comprehensive liquidity risk assessment"""
        bonds = [sample_bond]
        weights = [1.0]
        
        result = liquidity_risk.assess_liquidity_risk(bonds, weights)
        assert "lvar_value" in result
        assert "avg_spread_bps" in result
        assert "liquidity_score" in result
        assert "risk_level" in result