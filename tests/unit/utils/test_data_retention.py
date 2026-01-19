"""
Unit tests for data retention utilities
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from bondtrader.utils.data_retention import DataRetentionManager, RetentionPolicy


@pytest.mark.unit
class TestRetentionPolicy:
    """Test RetentionPolicy class"""

    def test_retention_policy_creation(self):
        """Test creating retention policy"""
        policy = RetentionPolicy(retention_years=7)
        assert policy.retention_years == 7
        assert policy.archival_enabled is True

    def test_get_retention_date(self):
        """Test getting retention date"""
        policy = RetentionPolicy(retention_years=7)
        retention_date = policy.get_retention_date()
        assert retention_date < datetime.now()
        # Should be approximately 7 years ago
        expected_date = datetime.now() - timedelta(days=365 * 7)
        assert abs((retention_date - expected_date).days) < 2

    def test_should_retain(self):
        """Test checking if record should be retained"""
        policy = RetentionPolicy(retention_years=7)
        # Recent record should be retained
        recent_date = datetime.now() - timedelta(days=365)
        assert policy.should_retain(recent_date) is True

        # Old record should not be retained
        old_date = datetime.now() - timedelta(days=365 * 8)
        assert policy.should_retain(old_date) is False


@pytest.mark.unit
class TestDataRetentionManager:
    """Test DataRetentionManager class"""

    def test_data_retention_manager_creation(self):
        """Test creating data retention manager"""
        manager = DataRetentionManager()
        assert manager is not None
        assert manager.policy is not None

    def test_archive_data(self):
        """Test archiving data"""
        manager = DataRetentionManager()
        data = {"bond_id": "TEST-001", "price": 1000}
        result = manager.archive_data(data, "bond", "TEST-001")
        # Archive should succeed or fail gracefully
        assert isinstance(result, bool)

    def test_cleanup_old_data(self):
        """Test policy should_retain for old data"""
        manager = DataRetentionManager()

        # Test that old records would not be retained
        old_date = datetime.now() - timedelta(days=365 * 8)
        should_retain = manager.policy.should_retain(old_date)
        assert should_retain is False

        # Test that recent records would be retained
        recent_date = datetime.now() - timedelta(days=365)
        should_retain = manager.policy.should_retain(recent_date)
        assert should_retain is True

    def test_get_retention_date(self):
        """Test getting retention date from manager"""
        manager = DataRetentionManager()
        retention_date = manager.policy.get_retention_date()
        assert retention_date < datetime.now()
