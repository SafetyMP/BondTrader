"""
Tests for execution strategies
"""

import pytest

from bondtrader.analytics.execution_strategies import ExecutionStrategy
from bondtrader.core.bond_models import Bond, BondType
from datetime import datetime, timedelta


@pytest.mark.unit
class TestExecutionStrategy:
    """Test ExecutionStrategy functionality"""

    @pytest.fixture
    def strategy(self):
        """Create execution strategy"""
        return ExecutionStrategy()

    @pytest.fixture
    def sample_bond(self):
        """Create sample bond"""
        return Bond(
            bond_id="TEST-001",
            bond_type=BondType.CORPORATE,
            face_value=1000,
            coupon_rate=5.0,
            maturity_date=datetime.now() + timedelta(days=1825),
            issue_date=datetime.now() - timedelta(days=365),
            current_price=950,
            credit_rating="BBB",
            issuer="Test Corp",
            frequency=2,
        )

    def test_twap_execution(self, strategy):
        """Test TWAP execution"""
        from datetime import datetime, timedelta

        start = datetime.now()
        end = start + timedelta(hours=1)
        result = strategy.twap_execution(total_quantity=100.0, start_time=start, end_time=end)
        assert isinstance(result, dict)
        assert "schedule" in result or "execution" in result

    def test_vwap_execution(self, strategy, sample_bond):
        """Test VWAP execution"""
        from datetime import datetime, timedelta

        start = datetime.now()
        end = start + timedelta(hours=1)
        volume_profile = [{"time": start + timedelta(minutes=i * 6), "expected_volume": 10.0} for i in range(10)]
        result = strategy.vwap_execution(total_quantity=100.0, volume_profile=volume_profile, start_time=start, end_time=end)
        assert isinstance(result, dict)

    def test_implementation_shortfall(self, strategy, sample_bond):
        """Test implementation shortfall"""
        execution_prices = [950.0, 951.0, 952.0, 951.5, 950.5]
        result = strategy.implementation_shortfall(bond=sample_bond, target_quantity=100.0, execution_prices=execution_prices)
        assert isinstance(result, dict)

    def test_optimal_execution(self, strategy, sample_bond):
        """Test optimal execution"""
        result = strategy.optimal_execution(sample_bond, total_quantity=100.0, urgency=0.5, volatility=0.01)
        assert isinstance(result, dict)
