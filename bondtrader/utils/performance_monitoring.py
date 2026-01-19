"""
Performance Monitoring and Alerting
Tracks performance metrics and triggers alerts

CRITICAL: Required for production operations in Fortune 10 financial institutions
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from bondtrader.core.audit import AuditEventType, get_audit_logger
from bondtrader.core.observability import get_metrics
from bondtrader.utils.utils import logger


class PerformanceThreshold:
    """Performance threshold configuration"""

    def __init__(
        self,
        metric_name: str,
        warning_threshold: float,
        critical_threshold: float,
        comparison: str = "greater",  # "greater" or "less"
    ):
        self.metric_name = metric_name
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.comparison = comparison


class Alert:
    """Alert object"""

    def __init__(
        self,
        severity: str,
        metric_name: str,
        value: float,
        threshold: float,
        message: str,
        timestamp: datetime,
    ):
        self.severity = severity  # "warning" or "critical"
        self.metric_name = metric_name
        self.value = value
        self.threshold = threshold
        self.message = message
        self.timestamp = timestamp

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "severity": self.severity,
            "metric_name": self.metric_name,
            "value": self.value,
            "threshold": self.threshold,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
        }


class PerformanceMonitor:
    """
    Performance Monitor

    Monitors performance metrics and triggers alerts.
    """

    def __init__(self):
        """Initialize performance monitor"""
        self.thresholds: Dict[str, PerformanceThreshold] = {}
        self.alerts: List[Alert] = []
        self.audit_logger = get_audit_logger()
        self.metrics = get_metrics()

        # Default thresholds
        self._setup_default_thresholds()

    def _setup_default_thresholds(self):
        """Setup default performance thresholds"""
        # Response time thresholds (milliseconds)
        self.add_threshold("api.response_time", 1000, 5000, "greater")  # 1s warning, 5s critical
        self.add_threshold(
            "valuation.calculation_time", 500, 2000, "greater"
        )  # 500ms warning, 2s critical
        self.add_threshold("ml.prediction_time", 1000, 5000, "greater")  # 1s warning, 5s critical

        # Error rate thresholds (percentage)
        self.add_threshold("api.error_rate", 1.0, 5.0, "greater")  # 1% warning, 5% critical
        self.add_threshold("valuation.error_rate", 0.5, 2.0, "greater")  # 0.5% warning, 2% critical

        # Throughput thresholds
        self.add_threshold("api.requests_per_second", 10, 5, "less")  # Below 10 RPS warning

    def add_threshold(
        self,
        metric_name: str,
        warning_threshold: float,
        critical_threshold: float,
        comparison: str = "greater",
    ):
        """
        Add performance threshold.

        Args:
            metric_name: Name of metric to monitor
            warning_threshold: Warning threshold value
            critical_threshold: Critical threshold value
            comparison: "greater" for upper bound, "less" for lower bound
        """
        threshold = PerformanceThreshold(
            metric_name, warning_threshold, critical_threshold, comparison
        )
        self.thresholds[metric_name] = threshold

    def check_threshold(self, metric_name: str, value: float) -> Optional[Alert]:
        """
        Check if metric value exceeds threshold.

        Args:
            metric_name: Metric name
            value: Current metric value

        Returns:
            Alert if threshold exceeded, None otherwise
        """
        threshold = self.thresholds.get(metric_name)
        if not threshold:
            return None

        is_exceeded = False
        severity = None

        if threshold.comparison == "greater":
            if value >= threshold.critical_threshold:
                is_exceeded = True
                severity = "critical"
            elif value >= threshold.warning_threshold:
                is_exceeded = True
                severity = "warning"
        else:  # less
            if value <= threshold.critical_threshold:
                is_exceeded = True
                severity = "critical"
            elif value <= threshold.warning_threshold:
                is_exceeded = True
                severity = "warning"

        if is_exceeded:
            message = (
                f"{metric_name} = {value} exceeds {severity} threshold "
                f"({threshold.critical_threshold if severity == 'critical' else threshold.warning_threshold})"
            )

            alert = Alert(
                severity, metric_name, value, threshold.critical_threshold, message, datetime.now()
            )
            self.alerts.append(alert)

            # Keep only last 1000 alerts
            if len(self.alerts) > 1000:
                self.alerts = self.alerts[-1000:]

            # Audit log
            self.audit_logger.log(
                AuditEventType.USER_ACTION,
                "system",
                "performance_alert",
                details=alert.to_dict(),
            )

            # Log alert
            if severity == "critical":
                logger.critical(f"PERFORMANCE ALERT: {message}")
            else:
                logger.warning(f"PERFORMANCE WARNING: {message}")

            return alert

        return None

    def get_recent_alerts(self, minutes: int = 60) -> List[Alert]:
        """
        Get recent alerts.

        Args:
            minutes: Number of minutes to look back

        Returns:
            List of recent alerts
        """
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return [alert for alert in self.alerts if alert.timestamp >= cutoff]

    def get_critical_alerts(self) -> List[Alert]:
        """Get all critical alerts"""
        return [alert for alert in self.alerts if alert.severity == "critical"]

    def get_performance_summary(self) -> Dict:
        """
        Get performance summary.

        Returns:
            Dictionary with performance metrics and alerts
        """
        all_metrics = self.metrics.get_metrics()
        recent_alerts = self.get_recent_alerts(60)

        return {
            "metrics": all_metrics,
            "recent_alerts": [alert.to_dict() for alert in recent_alerts],
            "critical_alerts": [alert.to_dict() for alert in self.get_critical_alerts()],
            "thresholds": {
                name: {
                    "warning": t.warning_threshold,
                    "critical": t.critical_threshold,
                    "comparison": t.comparison,
                }
                for name, t in self.thresholds.items()
            },
        }


# Global performance monitor instance
_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get or create global performance monitor instance"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor
