"""
Unit tests for alerting utilities
"""

import pytest
from unittest.mock import MagicMock, patch

from bondtrader.utils.alerting import Alert, AlertChannel, AlertManager, AlertSeverity


@pytest.mark.unit
class TestAlert:
    """Test Alert class"""

    def test_alert_creation(self):
        """Test creating an alert"""
        alert = Alert("Test Alert", "Test message", AlertSeverity.WARNING)
        assert alert.title == "Test Alert"
        assert alert.message == "Test message"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.source == "bondtrader"
        assert alert.acknowledged is False
        assert alert.resolved is False

    def test_alert_to_dict(self):
        """Test converting alert to dictionary"""
        alert = Alert("Test Alert", "Test message", AlertSeverity.ERROR, metadata={"key": "value"})
        alert_dict = alert.to_dict()
        assert alert_dict["title"] == "Test Alert"
        assert alert_dict["message"] == "Test message"
        assert alert_dict["severity"] == "error"
        assert alert_dict["metadata"] == {"key": "value"}
        assert "timestamp" in alert_dict

    def test_alert_with_metadata(self):
        """Test alert with metadata"""
        alert = Alert("Test", "Message", metadata={"component": "database", "error_code": "500"})
        assert alert.metadata["component"] == "database"
        assert alert.metadata["error_code"] == "500"


@pytest.mark.unit
class TestAlertManager:
    """Test AlertManager class"""

    def test_alert_manager_creation(self):
        """Test creating alert manager"""
        manager = AlertManager()
        assert manager is not None

    def test_send_alert(self):
        """Test sending an alert"""
        manager = AlertManager()
        alert = Alert("Test Alert", "Test message", AlertSeverity.INFO)

        with patch.object(manager, "_send_to_channel") as mock_send:
            manager.send_alert(alert)
            mock_send.assert_called()

    def test_send_alert_with_channels(self):
        """Test sending alert to specific channels"""
        manager = AlertManager()
        alert = Alert("Test Alert", "Test message", AlertSeverity.WARNING)

        with patch.object(manager, "_send_to_channel") as mock_send:
            manager.send_alert(alert, channels=[AlertChannel.EMAIL])
            assert mock_send.called

    def test_get_active_alerts(self):
        """Test getting active alerts"""
        manager = AlertManager()
        alert = Alert("Test Alert", "Test message", AlertSeverity.ERROR)
        manager.send_alert(alert)

        active = manager.get_active_alerts()
        assert len(active) >= 1

    def test_acknowledge_alert(self):
        """Test acknowledging an alert"""
        manager = AlertManager()
        alert = Alert("Test Alert", "Test message", AlertSeverity.WARNING)
        alert_id = manager.send_alert(alert)

        manager.acknowledge_alert(alert_id)
        assert alert.acknowledged is True

    def test_resolve_alert(self):
        """Test resolving an alert"""
        manager = AlertManager()
        alert = Alert("Test Alert", "Test message", AlertSeverity.ERROR)
        alert_id = manager.send_alert(alert)

        manager.resolve_alert(alert_id)
        assert alert.resolved is True