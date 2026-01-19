"""
Unit tests for core helpers module
"""

from datetime import datetime, timedelta

import pytest

from bondtrader.core.bond_models import Bond, BondType
from bondtrader.core.bond_valuation import BondValuator
from bondtrader.core.helpers import (
    calculate_portfolio_value,
    format_valuation_result,
    get_bond_or_error,
    get_bonds_or_error,
    validate_bond_data,
)
from bondtrader.core.repository import InMemoryBondRepository
from bondtrader.core.service_layer import BondService


@pytest.mark.unit
class TestCoreHelpers:
    """Test core helper functions"""

    @pytest.fixture
    def sample_bond_data(self):
        """Create sample bond data dictionary"""
        now = datetime.now()
        return {
            "bond_id": "TEST-001",
            "bond_type": BondType.CORPORATE,
            "face_value": 1000,
            "coupon_rate": 0.05,
            "maturity_date": now + timedelta(days=1825),
            "issue_date": now - timedelta(days=365),
            "current_price": 950,
            "credit_rating": "BBB",
            "issuer": "Test Corp",
            "frequency": 2,
        }

    @pytest.fixture
    def sample_bond(self, sample_bond_data):
        """Create sample bond"""
        return Bond(**sample_bond_data)

    @pytest.fixture
    def repository(self):
        """Create repository"""
        return InMemoryBondRepository()

    @pytest.fixture
    def service(self, repository):
        """Create bond service"""
        return BondService(repository=repository, valuator=BondValuator())

    def test_validate_bond_data_valid(self, sample_bond_data):
        """Test validating valid bond data"""
        result = validate_bond_data(sample_bond_data)
        assert result.is_ok()
        assert result.value == sample_bond_data

    def test_validate_bond_data_missing_field(self, sample_bond_data):
        """Test validating bond data with missing field"""
        del sample_bond_data["bond_id"]
        result = validate_bond_data(sample_bond_data)
        assert result.is_err()
        assert "bond_id" in str(result.error)

    def test_validate_bond_data_invalid_face_value(self, sample_bond_data):
        """Test validating bond data with invalid face value"""
        sample_bond_data["face_value"] = -100
        result = validate_bond_data(sample_bond_data)
        assert result.is_err()

    def test_validate_bond_data_invalid_price(self, sample_bond_data):
        """Test validating bond data with invalid price"""
        sample_bond_data["current_price"] = 0
        result = validate_bond_data(sample_bond_data)
        assert result.is_err()

    def test_validate_bond_data_invalid_coupon_rate(self, sample_bond_data):
        """Test validating bond data with invalid coupon rate"""
        sample_bond_data["coupon_rate"] = 1.5  # > 1
        result = validate_bond_data(sample_bond_data)
        assert result.is_err()

    def test_validate_bond_data_invalid_dates(self, sample_bond_data):
        """Test validating bond data with invalid dates"""
        sample_bond_data["issue_date"] = sample_bond_data["maturity_date"]
        result = validate_bond_data(sample_bond_data)
        assert result.is_err()

    def test_get_bond_or_error_success(self, repository, service, sample_bond):
        """Test getting bond successfully"""
        service.create_bond(sample_bond)
        # The helper uses container, so we need to ensure the container has the service
        from bondtrader.core.container import get_container

        container = get_container()
        container._bond_service = service
        result = get_bond_or_error("TEST-001")
        assert result.is_ok()
        assert result.value.bond_id == "TEST-001"

    def test_get_bond_or_error_not_found(self, repository, service):
        """Test getting non-existent bond"""
        from bondtrader.core.container import get_container

        container = get_container()
        container._bond_service = service
        result = get_bond_or_error("NONEXISTENT")
        assert result.is_err()

    def test_get_bonds_or_error_success(self, repository, service, sample_bond):
        """Test getting multiple bonds successfully"""
        service.create_bond(sample_bond)
        bond2 = Bond(
            bond_id="TEST-002",
            bond_type=BondType.TREASURY,
            face_value=1000,
            coupon_rate=4.0,
            maturity_date=datetime.now() + timedelta(days=1825),
            issue_date=datetime.now() - timedelta(days=365),
            current_price=980,
            credit_rating="AAA",
            issuer="US Treasury",
            frequency=2,
        )
        service.create_bond(bond2)
        from bondtrader.core.container import get_container

        container = get_container()
        container._bond_service = service
        result = get_bonds_or_error(["TEST-001", "TEST-002"])
        assert result.is_ok()
        assert len(result.value) == 2

    def test_get_bonds_or_error_partial_failure(self, repository, service, sample_bond):
        """Test getting bonds with some failures"""
        service.create_bond(sample_bond)
        from bondtrader.core.container import get_container

        container = get_container()
        container._bond_service = service
        result = get_bonds_or_error(["TEST-001", "NONEXISTENT"])
        assert result.is_err()

    def test_calculate_portfolio_value_empty(self):
        """Test calculating portfolio value for empty list"""
        result = calculate_portfolio_value([])
        assert result["total_market_value"] == 0
        assert result["total_fair_value"] == 0
        assert result["num_bonds"] == 0

    def test_calculate_portfolio_value_single_bond(self, repository, service, sample_bond):
        """Test calculating portfolio value for single bond"""
        service.create_bond(sample_bond)
        from bondtrader.core.container import get_container

        container = get_container()
        container._bond_service = service
        result = calculate_portfolio_value([sample_bond])
        assert result["total_market_value"] > 0
        assert "total_fair_value" in result
        assert result["num_bonds"] == 1

    def test_calculate_portfolio_value_with_weights(self, repository, service, sample_bond):
        """Test calculating portfolio value with weights"""
        bond2 = Bond(
            bond_id="TEST-002",
            bond_type=BondType.TREASURY,
            face_value=1000,
            coupon_rate=4.0,
            maturity_date=datetime.now() + timedelta(days=1825),
            issue_date=datetime.now() - timedelta(days=365),
            current_price=980,
            credit_rating="AAA",
            issuer="US Treasury",
            frequency=2,
        )
        service.create_bond(sample_bond)
        service.create_bond(bond2)
        from bondtrader.core.container import get_container

        container = get_container()
        container._bond_service = service
        result = calculate_portfolio_value([sample_bond, bond2], weights=[0.6, 0.4])
        assert result["total_market_value"] > 0
        assert result["num_bonds"] == 2

    def test_format_valuation_result(self):
        """Test formatting valuation result"""
        valuation = {
            "bond_id": "TEST-001",
            "fair_value": 1000.50,
            "market_price": 950.25,
            "ytm": 0.05,
            "duration": 4.5,
            "mismatch_percentage": -5.0,
        }
        formatted = format_valuation_result(valuation)
        assert "TEST-001" in formatted
        assert "1000.50" in formatted
        assert "5.00%" in formatted
