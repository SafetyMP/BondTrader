"""
Unit tests for factory patterns
"""

from datetime import datetime, timedelta

import pytest

from bondtrader.core.bond_models import BondType
from bondtrader.core.exceptions import InvalidBondError
from bondtrader.core.factories import BondFactory


@pytest.mark.unit
class TestBondFactory:
    """Test BondFactory functionality"""

    def test_create_bond_success(self):
        """Test creating bond successfully"""
        now = datetime.now()
        bond = BondFactory.create(
            bond_id="TEST-001",
            bond_type=BondType.CORPORATE,
            face_value=1000,
            coupon_rate=0.05,
            maturity_date=now + timedelta(days=1825),
            issue_date=now - timedelta(days=365),
            current_price=950,
        )
        assert bond.bond_id == "TEST-001"
        assert bond.bond_type == BondType.CORPORATE
        assert bond.face_value == 1000

    def test_create_bond_invalid_face_value(self):
        """Test creating bond with invalid face value"""
        now = datetime.now()
        with pytest.raises(InvalidBondError):
            BondFactory.create(
                bond_id="TEST-001",
                bond_type=BondType.CORPORATE,
                face_value=-1000,
                coupon_rate=0.05,
                maturity_date=now + timedelta(days=1825),
                issue_date=now - timedelta(days=365),
                current_price=950,
            )

    def test_create_bond_invalid_price(self):
        """Test creating bond with invalid price"""
        now = datetime.now()
        with pytest.raises(InvalidBondError):
            BondFactory.create(
                bond_id="TEST-001",
                bond_type=BondType.CORPORATE,
                face_value=1000,
                coupon_rate=0.05,
                maturity_date=now + timedelta(days=1825),
                issue_date=now - timedelta(days=365),
                current_price=0,
            )

    def test_create_bond_invalid_dates(self):
        """Test creating bond with invalid dates"""
        now = datetime.now()
        with pytest.raises(InvalidBondError):
            BondFactory.create(
                bond_id="TEST-001",
                bond_type=BondType.CORPORATE,
                face_value=1000,
                coupon_rate=0.05,
                maturity_date=now - timedelta(days=365),
                issue_date=now + timedelta(days=1825),
                current_price=950,
            )

    def test_create_bond_invalid_coupon_rate(self):
        """Test creating bond with invalid coupon rate"""
        now = datetime.now()
        with pytest.raises(InvalidBondError):
            BondFactory.create(
                bond_id="TEST-001",
                bond_type=BondType.CORPORATE,
                face_value=1000,
                coupon_rate=1.5,
                maturity_date=now + timedelta(days=1825),
                issue_date=now - timedelta(days=365),
                current_price=950,
            )

    def test_create_bond_invalid_frequency(self):
        """Test creating bond with invalid frequency"""
        now = datetime.now()
        with pytest.raises(InvalidBondError):
            BondFactory.create(
                bond_id="TEST-001",
                bond_type=BondType.CORPORATE,
                face_value=1000,
                coupon_rate=0.05,
                maturity_date=now + timedelta(days=1825),
                issue_date=now - timedelta(days=365),
                current_price=950,
                frequency=0,
            )

    def test_create_from_dict(self):
        """Test creating bond from dictionary"""
        now = datetime.now()
        data = {
            "bond_id": "TEST-001",
            "bond_type": "CORPORATE",
            "face_value": 1000,
            "coupon_rate": 0.05,
            "maturity_date": now + timedelta(days=1825),
            "issue_date": now - timedelta(days=365),
            "current_price": 950,
            "credit_rating": "BBB",
            "issuer": "Test Corp",
            "frequency": 2,
        }
        bond = BondFactory.create_from_dict(data)
        assert bond.bond_id == "TEST-001"
        assert bond.bond_type == BondType.CORPORATE

    def test_create_from_dict_with_string_bond_type(self):
        """Test creating bond from dict with string bond type"""
        now = datetime.now()
        data = {
            "bond_id": "TEST-001",
            "bond_type": "CORPORATE",
            "face_value": 1000,
            "coupon_rate": 0.05,
            "maturity_date": now + timedelta(days=1825),
            "issue_date": now - timedelta(days=365),
            "current_price": 950,
        }
        bond = BondFactory.create_from_dict(data)
        assert bond.bond_id == "TEST-001"
        assert bond.bond_type == BondType.CORPORATE

    def test_create_from_dict_with_datetime_dates(self):
        """Test creating bond from dict with datetime objects"""
        now = datetime.now()
        data = {
            "bond_id": "TEST-001",
            "bond_type": BondType.CORPORATE,
            "face_value": 1000,
            "coupon_rate": 0.05,
            "maturity_date": now + timedelta(days=1825),
            "issue_date": now - timedelta(days=365),
            "current_price": 950,
        }
        bond = BondFactory.create_from_dict(data)
        assert bond.bond_id == "TEST-001"


@pytest.mark.unit
class TestMLModelFactory:
    """Test MLModelFactory functionality"""

    def test_create_basic(self):
        """Test creating basic ML adjuster"""
        from bondtrader.core.container import get_container
        from bondtrader.core.factories import MLModelFactory
        from unittest.mock import MagicMock, patch

        # Mock container to avoid actual setup
        with patch("bondtrader.core.factories.get_container") as mock_get_container:
            mock_container = MagicMock()
            mock_valuator = MagicMock()
            mock_container.get_valuator.return_value = mock_valuator
            mock_get_container.return_value = mock_container

            # Should not raise - factory creates the adjuster
            result = MLModelFactory.create_basic()
            assert result is not None

    def test_create_enhanced(self):
        """Test creating enhanced ML adjuster"""
        from bondtrader.core.factories import MLModelFactory
        from unittest.mock import MagicMock, patch

        with patch("bondtrader.core.factories.get_container") as mock_get_container:
            mock_container = MagicMock()
            mock_valuator = MagicMock()
            mock_container.get_valuator.return_value = mock_valuator
            mock_get_container.return_value = mock_container

            result = MLModelFactory.create_enhanced()
            assert result is not None

    def test_create_advanced(self):
        """Test creating advanced ML adjuster"""
        from bondtrader.core.factories import MLModelFactory
        from unittest.mock import MagicMock, patch

        with patch("bondtrader.core.factories.get_container") as mock_get_container:
            mock_container = MagicMock()
            mock_valuator = MagicMock()
            mock_container.get_valuator.return_value = mock_valuator
            mock_get_container.return_value = mock_container

            result = MLModelFactory.create_advanced()
            assert result is not None

    def test_create_automl(self):
        """Test creating AutoML adjuster"""
        from bondtrader.core.factories import MLModelFactory
        from unittest.mock import MagicMock, patch

        with patch("bondtrader.core.factories.get_container") as mock_get_container:
            mock_container = MagicMock()
            mock_valuator = MagicMock()
            mock_container.get_valuator.return_value = mock_valuator
            mock_get_container.return_value = mock_container

            result = MLModelFactory.create_automl()
            assert result is not None


@pytest.mark.unit
class TestAnalyticsFactory:
    """Test AnalyticsFactory functionality"""

    def test_create_portfolio_optimizer(self):
        """Test creating portfolio optimizer"""
        from bondtrader.core.factories import AnalyticsFactory
        from unittest.mock import MagicMock, patch

        with patch("bondtrader.core.factories.get_container") as mock_get_container:
            mock_container = MagicMock()
            mock_valuator = MagicMock()
            mock_container.get_valuator.return_value = mock_valuator
            mock_get_container.return_value = mock_container

            result = AnalyticsFactory.create_portfolio_optimizer()
            assert result is not None
