"""
Unit tests for pool monitoring utilities
"""

from unittest.mock import MagicMock, patch

import pytest

from bondtrader.utils.pool_monitoring import PoolMonitor, get_pool_monitor


@pytest.mark.unit
class TestPoolMonitor:
    """Test PoolMonitor class"""

    def test_pool_monitor_creation(self):
        """Test creating pool monitor"""
        monitor = PoolMonitor()
        assert monitor is not None

    def test_get_pool_stats(self):
        """Test getting pool statistics"""
        monitor = PoolMonitor()

        # Mock database with engine
        mock_database = MagicMock()
        mock_pool = MagicMock()
        mock_pool.size.return_value = 10
        mock_pool.checkedin.return_value = 8
        mock_pool.checkedout.return_value = 2
        mock_pool.overflow.return_value = 0
        mock_pool.invalid.return_value = 0
        mock_database.engine.pool = mock_pool

        stats = monitor.get_pool_stats(mock_database)
        assert stats["pool_size"] == 10
        assert stats["checked_in"] == 8
        assert stats["checked_out"] == 2
        assert stats["utilization_pct"] == 20.0

    def test_get_pool_stats_no_engine(self):
        """Test getting stats when engine not available"""
        monitor = PoolMonitor()
        mock_database = MagicMock()
        del mock_database.engine

        stats = monitor.get_pool_stats(mock_database)
        assert "error" in stats

    def test_get_pool_stats_high_utilization(self):
        """Test pool stats with high utilization"""
        monitor = PoolMonitor()

        mock_database = MagicMock()
        mock_pool = MagicMock()
        mock_pool.size.return_value = 10
        mock_pool.checkedin.return_value = 1
        mock_pool.checkedout.return_value = 9
        mock_pool.overflow.return_value = 0
        mock_pool.invalid.return_value = 0
        mock_database.engine.pool = mock_pool

        stats = monitor.get_pool_stats(mock_database)
        assert stats["utilization_pct"] == 90.0

    def test_check_pool_health(self):
        """Test checking pool health"""
        monitor = PoolMonitor()

        mock_database = MagicMock()
        mock_pool = MagicMock()
        mock_pool.size.return_value = 10
        mock_pool.checkedin.return_value = 8
        mock_pool.checkedout.return_value = 2
        mock_pool.overflow.return_value = 0
        mock_pool.invalid.return_value = 0
        mock_database.engine.pool = mock_pool

        health = monitor.check_pool_health(mock_database)
        assert health["status"] == "healthy"
        assert "stats" in health
        assert "issues" in health

    def test_check_pool_health_critical(self):
        """Test pool health with critical utilization"""
        monitor = PoolMonitor()

        mock_database = MagicMock()
        mock_pool = MagicMock()
        mock_pool.size.return_value = 10
        mock_pool.checkedin.return_value = 0
        mock_pool.checkedout.return_value = 10
        mock_pool.overflow.return_value = 0
        mock_pool.invalid.return_value = 0
        mock_database.engine.pool = mock_pool

        health = monitor.check_pool_health(mock_database)
        assert health["status"] == "critical"
        assert len(health["issues"]) > 0

    def test_check_pool_health_invalid_connections(self):
        """Test pool health with invalid connections"""
        monitor = PoolMonitor()

        mock_database = MagicMock()
        mock_pool = MagicMock()
        mock_pool.size.return_value = 10
        mock_pool.checkedin.return_value = 7
        mock_pool.checkedout.return_value = 2
        mock_pool.overflow.return_value = 0
        mock_pool.invalid.return_value = 1
        mock_database.engine.pool = mock_pool

        health = monitor.check_pool_health(mock_database)
        assert health["status"] == "degraded"
        assert "invalid" in str(health["issues"]).lower()


@pytest.mark.unit
class TestPoolMonitorFunctions:
    """Test pool monitor helper functions"""

    def test_get_pool_monitor(self):
        """Test getting global pool monitor"""
        monitor1 = get_pool_monitor()
        monitor2 = get_pool_monitor()
        # Should return same instance (singleton)
        assert monitor1 is monitor2
