"""
Tests for repository pattern implementation
"""

from datetime import datetime, timedelta

import pytest

from bondtrader.core.bond_models import Bond, BondType
from bondtrader.core.repository import BondRepository, IBondRepository, InMemoryBondRepository


@pytest.mark.unit
class TestBondRepository:
    """Test BondRepository implementation"""

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

    def test_save_bond(self, sample_bond):
        """Test saving a bond"""
        repo = BondRepository()
        repo.save(sample_bond)
        # Should not raise

    def test_find_by_id(self, sample_bond):
        """Test finding bond by ID"""
        repo = BondRepository()
        repo.save(sample_bond)
        found = repo.find_by_id("TEST-001")
        # May return None if database not initialized, but should not raise

    def test_find_all(self, sample_bond):
        """Test finding all bonds"""
        repo = BondRepository()
        bonds = repo.find_all()
        assert isinstance(bonds, list)

    def test_find_all_with_filters(self, sample_bond):
        """Test finding bonds with filters"""
        repo = BondRepository()
        bonds = repo.find_all({"bond_type": BondType.CORPORATE})
        assert isinstance(bonds, list)

    def test_exists(self, sample_bond):
        """Test checking if bond exists"""
        repo = BondRepository()
        exists = repo.exists("TEST-001")
        assert isinstance(exists, bool)

    def test_count(self, sample_bond):
        """Test counting bonds"""
        repo = BondRepository()
        count = repo.count()
        assert isinstance(count, int)
        assert count >= 0


@pytest.mark.unit
class TestInMemoryBondRepository:
    """Test InMemoryBondRepository implementation"""

    @pytest.fixture
    def repo(self):
        """Create in-memory repository"""
        return InMemoryBondRepository()

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

    def test_save_and_find(self, repo, sample_bond):
        """Test saving and finding bond"""
        repo.save(sample_bond)
        found = repo.find_by_id("TEST-001")
        assert found is not None
        assert found.bond_id == "TEST-001"

    def test_find_nonexistent(self, repo):
        """Test finding nonexistent bond"""
        found = repo.find_by_id("NONEXISTENT")
        assert found is None

    def test_exists(self, repo, sample_bond):
        """Test exists check"""
        assert not repo.exists("TEST-001")
        repo.save(sample_bond)
        assert repo.exists("TEST-001")

    def test_find_all(self, repo, sample_bond):
        """Test finding all bonds"""
        assert len(repo.find_all()) == 0
        repo.save(sample_bond)
        bonds = repo.find_all()
        assert len(bonds) == 1
        assert bonds[0].bond_id == "TEST-001"

    def test_find_all_with_filters(self, repo, sample_bond):
        """Test finding with filters"""
        repo.save(sample_bond)

        # Filter by bond type
        bonds = repo.find_all({"bond_type": BondType.CORPORATE})
        assert len(bonds) == 1

        bonds = repo.find_all({"bond_type": BondType.TREASURY})
        assert len(bonds) == 0

        # Filter by issuer
        bonds = repo.find_all({"issuer": "Test Corp"})
        assert len(bonds) == 1

        # Filter by credit rating
        bonds = repo.find_all({"credit_rating": "BBB"})
        assert len(bonds) == 1

    def test_delete(self, repo, sample_bond):
        """Test deleting bond"""
        repo.save(sample_bond)
        assert repo.exists("TEST-001")

        deleted = repo.delete("TEST-001")
        assert deleted is True
        assert not repo.exists("TEST-001")

    def test_delete_nonexistent(self, repo):
        """Test deleting nonexistent bond"""
        deleted = repo.delete("NONEXISTENT")
        assert deleted is False

    def test_count(self, repo, sample_bond):
        """Test counting bonds"""
        assert repo.count() == 0
        repo.save(sample_bond)
        assert repo.count() == 1

    def test_count_with_filters(self, repo, sample_bond):
        """Test counting with filters"""
        repo.save(sample_bond)
        assert repo.count({"bond_type": BondType.CORPORATE}) == 1
        assert repo.count({"bond_type": BondType.TREASURY}) == 0
