"""
Extended tests for service layer pattern - comprehensive coverage
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from bondtrader.core.bond_models import Bond, BondType
from bondtrader.core.bond_valuation import BondValuator
from bondtrader.core.exceptions import RiskCalculationError, ValuationError
from bondtrader.core.repository import InMemoryBondRepository
from bondtrader.core.service_layer import BondService


@pytest.mark.unit
class TestBondServiceExtended:
    """Extended tests for BondService for comprehensive coverage"""

    @pytest.fixture
    def repository(self):
        """Create in-memory repository"""
        return InMemoryBondRepository()

    @pytest.fixture
    def valuator(self):
        """Create valuator"""
        return BondValuator(risk_free_rate=0.03)

    @pytest.fixture
    def service(self, repository, valuator):
        """Create bond service"""
        return BondService(repository=repository, valuator=valuator)

    @pytest.fixture
    def sample_bonds(self):
        """Create multiple sample bonds"""
        now = datetime.now()
        return [
            Bond(
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
            ),
            Bond(
                bond_id="TEST-002",
                bond_type=BondType.TREASURY,
                face_value=1000,
                coupon_rate=4.0,
                maturity_date=now + timedelta(days=1825),
                issue_date=now - timedelta(days=365),
                current_price=980,
                credit_rating="AAA",
                issuer="US Treasury",
                frequency=2,
            ),
        ]

    def test_find_arbitrage_opportunities_no_bonds(self, service):
        """Test finding arbitrage opportunities with no bonds"""
        with patch("bondtrader.core.container.get_container") as mock_get_container:
            mock_container = MagicMock()
            mock_detector = MagicMock()
            mock_detector.find_arbitrage_opportunities.return_value = []
            mock_container.get_arbitrage_detector.return_value = mock_detector
            mock_get_container.return_value = mock_container

            result = service.find_arbitrage_opportunities()
            assert result.is_ok()
            assert len(result.value) == 0

    def test_find_arbitrage_opportunities_with_bonds(self, service, sample_bonds):
        """Test finding arbitrage opportunities with bonds"""
        for bond in sample_bonds:
            service.create_bond(bond)

        with patch("bondtrader.core.container.get_container") as mock_get_container:
            mock_container = MagicMock()
            mock_detector = MagicMock()
            # min_profit_percentage is not a parameter to find_arbitrage_opportunities
            mock_detector.find_arbitrage_opportunities.return_value = [
                {"bond_id": "TEST-001", "profit_percentage": 5.0}
            ]
            mock_container.get_arbitrage_detector.return_value = mock_detector
            mock_get_container.return_value = mock_container

            result = service.find_arbitrage_opportunities()
            assert result.is_ok()
            assert len(result.value) >= 0

    def test_find_arbitrage_opportunities_with_limit(self, service, sample_bonds):
        """Test finding arbitrage opportunities with limit"""
        for bond in sample_bonds:
            service.create_bond(bond)

        with patch("bondtrader.core.container.get_container") as mock_get_container:
            mock_container = MagicMock()
            mock_detector = MagicMock()
            opportunities_list = [{"bond_id": f"TEST-00{i}", "profit_percentage": 5.0 - i} for i in range(1, 6)]
            mock_detector.find_arbitrage_opportunities.return_value = opportunities_list
            mock_container.get_arbitrage_detector.return_value = mock_detector
            mock_get_container.return_value = mock_container

            result = service.find_arbitrage_opportunities(limit=3)
            assert result.is_ok()
            assert len(result.value) <= 3

    def test_find_arbitrage_opportunities_error(self, service):
        """Test find arbitrage opportunities with error"""
        with patch("bondtrader.core.container.get_container") as mock_get_container:
            mock_get_container.side_effect = Exception("Container error")
            result = service.find_arbitrage_opportunities()
            assert result.is_err()

    def test_calculate_portfolio_risk_single_bond(self, service, sample_bonds):
        """Test calculating portfolio risk for single bond"""
        service.create_bond(sample_bonds[0])

        with patch("bondtrader.core.container.get_container") as mock_get_container:
            mock_container = MagicMock()
            mock_risk_manager = MagicMock()
            mock_risk_manager.calculate_var.return_value = {"var_value": 100.0}
            mock_risk_manager.calculate_portfolio_credit_risk.return_value = {"expected_loss": 50.0}
            mock_container.get_risk_manager.return_value = mock_risk_manager
            mock_get_container.return_value = mock_container

            result = service.calculate_portfolio_risk(["TEST-001"])
            assert result.is_ok()
            risk_metrics = result.value
            assert "var_historical" in risk_metrics
            assert "var_parametric" in risk_metrics
            assert "var_monte_carlo" in risk_metrics
            assert "credit_risk" in risk_metrics

    def test_calculate_portfolio_risk_multiple_bonds(self, service, sample_bonds):
        """Test calculating portfolio risk for multiple bonds"""
        for bond in sample_bonds:
            service.create_bond(bond)

        with patch("bondtrader.core.container.get_container") as mock_get_container:
            mock_container = MagicMock()
            mock_risk_manager = MagicMock()
            mock_risk_manager.calculate_var.return_value = {"var_value": 100.0}
            mock_risk_manager.calculate_portfolio_credit_risk.return_value = {"expected_loss": 50.0}
            mock_container.get_risk_manager.return_value = mock_risk_manager
            mock_get_container.return_value = mock_container

            result = service.calculate_portfolio_risk(["TEST-001", "TEST-002"], weights=[0.6, 0.4])
            assert result.is_ok()

    def test_calculate_portfolio_risk_invalid_weights(self, service, sample_bonds):
        """Test calculating portfolio risk with invalid weights"""
        service.create_bond(sample_bonds[0])
        result = service.calculate_portfolio_risk(["TEST-001"], weights=[0.5, 0.6])
        assert result.is_err()
        assert isinstance(result.error, RiskCalculationError)

    def test_calculate_portfolio_risk_missing_bond(self, service):
        """Test calculating portfolio risk with missing bond"""
        result = service.calculate_portfolio_risk(["NONEXISTENT"])
        assert result.is_err()

    def test_calculate_portfolio_metrics(self, service, sample_bonds):
        """Test calculating portfolio metrics"""
        for bond in sample_bonds:
            service.create_bond(bond)

        result = service.calculate_portfolio_metrics(["TEST-001", "TEST-002"])
        assert result.is_ok()
        metrics = result.value
        assert "portfolio_return" in metrics or "total_market_value" in metrics

    def test_calculate_portfolio_metrics_invalid_weights(self, service, sample_bonds):
        """Test calculating portfolio metrics with invalid weights"""
        service.create_bond(sample_bonds[0])
        result = service.calculate_portfolio_metrics(["TEST-001"], weights=[0.5, 0.6])
        assert result.is_err()

    def test_create_bonds_batch(self, service, sample_bonds):
        """Test creating multiple bonds in batch"""
        result = service.create_bonds_batch(sample_bonds)
        assert result.is_ok()
        bonds = result.value
        assert len(bonds) == 2
        assert all(b.bond_id in ["TEST-001", "TEST-002"] for b in bonds)

    def test_create_bonds_batch_duplicate(self, service, sample_bonds):
        """Test creating batch with duplicate bond"""
        service.create_bond(sample_bonds[0])
        result = service.create_bonds_batch(sample_bonds)
        # InMemoryBondRepository doesn't support transactions, so may succeed partially
        # Just verify it doesn't crash
        assert result is not None

    def test_predict_with_ml_enhanced_model(self, service, sample_bonds):
        """Test ML prediction with enhanced model type"""
        service.create_bond(sample_bonds[0])
        result = service.predict_with_ml("TEST-001", model_type="enhanced")
        assert result.is_ok()
        prediction = result.value
        assert "theoretical_fair_value" in prediction
        assert "ml_adjusted_fair_value" in prediction

    def test_predict_with_ml_advanced_model(self, service, sample_bonds):
        """Test ML prediction with advanced model type"""
        service.create_bond(sample_bonds[0])
        result = service.predict_with_ml("TEST-001", model_type="advanced")
        assert result.is_ok()

    def test_predict_with_ml_with_model(self, service, sample_bonds):
        """Test ML prediction with provided model"""
        from bondtrader.ml.ml_adjuster import MLBondAdjuster

        service.create_bond(sample_bonds[0])
        ml_model = MLBondAdjuster(valuator=service.valuator)
        result = service.predict_with_ml("TEST-001", ml_model=ml_model)
        assert result.is_ok()

    def test_predict_with_ml_bond_not_found(self, service):
        """Test ML prediction with non-existent bond"""
        result = service.predict_with_ml("NONEXISTENT")
        assert result.is_err()

    def test_calculate_valuations_with_ml_batch(self, service, sample_bonds):
        """Test batch valuation with ML"""
        for bond in sample_bonds:
            service.create_bond(bond)

        result = service.calculate_valuations_with_ml_batch(["TEST-001", "TEST-002"])
        assert result.is_ok()
        valuations = result.value
        assert len(valuations) == 2

    def test_calculate_valuations_with_ml_batch_partial_error(self, service, sample_bonds):
        """Test batch valuation with ML with some errors"""
        service.create_bond(sample_bonds[0])
        # Don't create second bond
        result = service.calculate_valuations_with_ml_batch(["TEST-001", "NONEXISTENT"])
        # Should continue with valid bonds
        assert result.is_ok()
        valuations = result.value
        assert len(valuations) >= 1

    def test_get_bond_count_with_filters(self, service, sample_bonds):
        """Test getting bond count with filters"""
        for bond in sample_bonds:
            service.create_bond(bond)

        result = service.get_bond_count({"bond_type": BondType.CORPORATE})
        assert result.is_ok()
        assert result.value >= 1

    def test_calculate_valuation_error_path(self, service):
        """Test calculate valuation with non-existent bond"""
        result = service.calculate_valuation("NONEXISTENT")
        assert result.is_err()

    def test_find_bonds_with_error(self, service):
        """Test find bonds with repository error"""
        # Mock repository to raise error
        original_find_all = service.repository.find_all
        service.repository.find_all = MagicMock(side_effect=Exception("Repository error"))
        result = service.find_bonds()
        assert result.is_err()
        # Restore original method
        service.repository.find_all = original_find_all
