"""
Alerting System Integration
Integrates with PagerDuty, Opsgenie, and other alerting systems

CRITICAL: Required for production operations in Fortune 10 financial institutions
"""

import json
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import requests

from bondtrader.core.audit import AuditEventType, get_audit_logger
from bondtrader.utils.utils import logger


class AlertSeverity(str, Enum):
    """Alert severity levels"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertChannel(str, Enum):
    """Alert channel types"""

    PAGERDUTY = "pagerduty"
    OPSGENIE = "opsgenie"
    SLACK = "slack"
    EMAIL = "email"
    WEBHOOK = "webhook"


class Alert:
    """Alert object"""

    def __init__(
        self,
        title: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.INFO,
        source: str = "bondtrader",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.title = title
        self.message = message
        self.severity = severity
        self.source = source
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
        self.acknowledged = False
        self.resolved = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "title": self.title,
            "message": self.message,
            "severity": self.severity.value,
            "source": self.source,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged,
            "resolved": self.resolved,
        }


class AlertManager:
    """
    Alert Manager

    Manages alerts and integrates with external alerting systems.
    """

    def __init__(self):
        """Initialize alert manager"""
        self.alerts: List[Alert] = []
        self.channels: Dict[AlertChannel, Dict[str, Any]] = {}
        self.audit_logger = get_audit_logger()
        self._load_config()

    def _load_config(self):
        """Load alerting configuration from environment"""
        import os

        # PagerDuty
        pagerduty_key = os.getenv("PAGERDUTY_INTEGRATION_KEY")
        if pagerduty_key:
            self.channels[AlertChannel.PAGERDUTY] = {"integration_key": pagerduty_key}

        # Opsgenie
        opsgenie_key = os.getenv("OPSGENIE_API_KEY")
        if opsgenie_key:
            self.channels[AlertChannel.OPSGENIE] = {"api_key": opsgenie_key}

        # Slack
        slack_webhook = os.getenv("SLACK_WEBHOOK_URL")
        if slack_webhook:
            self.channels[AlertChannel.SLACK] = {"webhook_url": slack_webhook}

        # Generic webhook
        webhook_url = os.getenv("ALERT_WEBHOOK_URL")
        if webhook_url:
            self.channels[AlertChannel.WEBHOOK] = {"url": webhook_url}

    def send_alert(
        self,
        title: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.INFO,
        channels: Optional[List[AlertChannel]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Send alert to configured channels.

        Args:
            title: Alert title
            message: Alert message
            severity: Alert severity
            channels: Optional list of channels to send to (defaults to all configured)
            metadata: Optional metadata

        Returns:
            True if at least one channel succeeded
        """
        alert = Alert(title, message, severity, metadata=metadata)
        self.alerts.append(alert)

        # Keep only last 1000 alerts
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-1000:]

        # Determine channels to use
        if channels is None:
            channels = list(self.channels.keys())

        success = False

        # Send to each channel
        for channel in channels:
            if channel not in self.channels:
                continue

            try:
                if channel == AlertChannel.PAGERDUTY:
                    success = self._send_pagerduty(alert) or success
                elif channel == AlertChannel.OPSGENIE:
                    success = self._send_opsgenie(alert) or success
                elif channel == AlertChannel.SLACK:
                    success = self._send_slack(alert) or success
                elif channel == AlertChannel.WEBHOOK:
                    success = self._send_webhook(alert) or success
            except Exception as e:
                logger.error(f"Failed to send alert to {channel.value}: {e}")

        # Audit log
        self.audit_logger.log(
            AuditEventType.USER_ACTION,
            "system",
            "alert_sent",
            details={
                "title": title,
                "severity": severity.value,
                "channels": [c.value for c in channels],
                "success": success,
            },
        )

        return success

    def _send_pagerduty(self, alert: Alert) -> bool:
        """Send alert to PagerDuty"""
        config = self.channels[AlertChannel.PAGERDUTY]
        integration_key = config.get("integration_key")

        if not integration_key:
            return False

        # Map severity to PagerDuty severity
        severity_map = {
            AlertSeverity.INFO: "info",
            AlertSeverity.WARNING: "warning",
            AlertSeverity.ERROR: "error",
            AlertSeverity.CRITICAL: "critical",
        }

        payload = {
            "routing_key": integration_key,
            "event_action": "trigger",
            "payload": {
                "summary": alert.title,
                "source": alert.source,
                "severity": severity_map.get(alert.severity, "info"),
                "custom_details": {
                    "message": alert.message,
                    "metadata": alert.metadata,
                },
            },
        }

        try:
            response = requests.post(
                "https://events.pagerduty.com/v2/enqueue",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=5,
            )
            response.raise_for_status()
            logger.info(f"Alert sent to PagerDuty: {alert.title}")
            return True
        except Exception as e:
            logger.error(f"Failed to send alert to PagerDuty: {e}")
            return False

    def _send_opsgenie(self, alert: Alert) -> bool:
        """Send alert to Opsgenie"""
        config = self.channels[AlertChannel.OPSGENIE]
        api_key = config.get("api_key")

        if not api_key:
            return False

        # Map severity to Opsgenie priority
        priority_map = {
            AlertSeverity.INFO: "P5",
            AlertSeverity.WARNING: "P4",
            AlertSeverity.ERROR: "P3",
            AlertSeverity.CRITICAL: "P1",
        }

        payload = {
            "message": alert.title,
            "description": alert.message,
            "priority": priority_map.get(alert.severity, "P5"),
            "source": alert.source,
            "details": alert.metadata,
        }

        try:
            response = requests.post(
                "https://api.opsgenie.com/v2/alerts",
                json=payload,
                headers={
                    "Authorization": f"GenieKey {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=5,
            )
            response.raise_for_status()
            logger.info(f"Alert sent to Opsgenie: {alert.title}")
            return True
        except Exception as e:
            logger.error(f"Failed to send alert to Opsgenie: {e}")
            return False

    def _send_slack(self, alert: Alert) -> bool:
        """Send alert to Slack"""
        config = self.channels[AlertChannel.SLACK]
        webhook_url = config.get("webhook_url")

        if not webhook_url:
            return False

        # Map severity to Slack color
        color_map = {
            AlertSeverity.INFO: "#36a64f",
            AlertSeverity.WARNING: "#ff9900",
            AlertSeverity.ERROR: "#ff0000",
            AlertSeverity.CRITICAL: "#8b0000",
        }

        payload = {
            "attachments": [
                {
                    "color": color_map.get(alert.severity, "#36a64f"),
                    "title": alert.title,
                    "text": alert.message,
                    "fields": [
                        {"title": "Severity", "value": alert.severity.value, "short": True},
                        {"title": "Source", "value": alert.source, "short": True},
                    ],
                    "ts": int(alert.timestamp.timestamp()),
                }
            ]
        }

        try:
            response = requests.post(webhook_url, json=payload, timeout=5)
            response.raise_for_status()
            logger.info(f"Alert sent to Slack: {alert.title}")
            return True
        except Exception as e:
            logger.error(f"Failed to send alert to Slack: {e}")
            return False

    def _send_webhook(self, alert: Alert) -> bool:
        """Send alert to generic webhook"""
        config = self.channels[AlertChannel.WEBHOOK]
        url = config.get("url")

        if not url:
            return False

        payload = alert.to_dict()

        try:
            response = requests.post(url, json=payload, timeout=5)
            response.raise_for_status()
            logger.info(f"Alert sent to webhook: {alert.title}")
            return True
        except Exception as e:
            logger.error(f"Failed to send alert to webhook: {e}")
            return False

    def get_recent_alerts(self, limit: int = 100) -> List[Alert]:
        """Get recent alerts"""
        return self.alerts[-limit:]

    def acknowledge_alert(self, alert_index: int) -> bool:
        """Acknowledge an alert"""
        if 0 <= alert_index < len(self.alerts):
            self.alerts[alert_index].acknowledged = True
            return True
        return False


# Global alert manager instance
_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """Get or create global alert manager instance"""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager
