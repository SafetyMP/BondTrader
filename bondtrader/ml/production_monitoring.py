"""
Production Model Monitoring and Alerting
Real-time monitoring with automated alerting

Industry Best Practices:
- Real-time prediction monitoring
- Performance metrics tracking
- Automated alerting
- Monitoring dashboards
"""

import json
import os
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from bondtrader.config import get_config
from bondtrader.utils.utils import logger


@dataclass
class PredictionRecord:
    """Record of a single prediction"""

    timestamp: datetime
    bond_id: str
    predicted_value: float
    actual_value: Optional[float] = None
    error: Optional[float] = None
    model_version: str = "unknown"
    features: Optional[Dict[str, float]] = None
    latency_ms: float = 0.0


@dataclass
class MonitoringMetrics:
    """Aggregated monitoring metrics"""

    timestamp: datetime
    window_start: datetime
    window_end: datetime
    n_predictions: int
    mean_error: float
    rmse: float
    mae: float
    max_error: float
    error_rate: float  # Percentage of predictions with error > threshold
    prediction_latency_ms: float
    model_version: str


class ModelMonitor:
    """
    Production model monitoring with alerting

    Industry Best Practices:
    - Real-time metrics tracking
    - Automated alerting
    - Performance degradation detection
    - Drift detection
    """

    def __init__(
        self,
        model_name: str,
        alert_thresholds: Dict[str, float] = None,
        window_size: int = 1000,
        monitoring_dir: str = None,
    ):
        """
        Initialize model monitor

        Args:
            model_name: Name of the model being monitored
            alert_thresholds: Dictionary of metric thresholds for alerting
            window_size: Size of sliding window for metrics
            monitoring_dir: Directory to store monitoring data
        """
        self.model_name = model_name
        self.config = get_config()
        self.monitoring_dir = monitoring_dir or os.path.join(
            self.config.model_dir, "monitoring", model_name
        )
        os.makedirs(self.monitoring_dir, exist_ok=True)

        # Default alert thresholds
        self.alert_thresholds = alert_thresholds or {
            "rmse_threshold": 100.0,  # RMSE threshold
            "mae_threshold": 50.0,  # MAE threshold
            "error_rate_threshold": 0.10,  # 10% error rate
            "latency_threshold_ms": 1000.0,  # 1 second latency
        }

        # Sliding window for predictions
        self.window_size = window_size
        self.prediction_window = deque(maxlen=window_size)

        # Alert callbacks
        self.alert_callbacks: List[Callable] = []

        # Historical metrics
        self.metrics_history: List[MonitoringMetrics] = []

    def record_prediction(
        self,
        bond_id: str,
        predicted_value: float,
        actual_value: float = None,
        model_version: str = "unknown",
        features: Dict[str, float] = None,
        latency_ms: float = 0.0,
    ):
        """
        Record a prediction for monitoring

        Args:
            bond_id: Bond identifier
            predicted_value: Model prediction
            actual_value: Actual value (if available)
            model_version: Model version
            features: Feature values used
            latency_ms: Prediction latency in milliseconds
        """
        error = None
        if actual_value is not None:
            error = abs(predicted_value - actual_value)

        record = PredictionRecord(
            timestamp=datetime.now(),
            bond_id=bond_id,
            predicted_value=predicted_value,
            actual_value=actual_value,
            error=error,
            model_version=model_version,
            features=features,
            latency_ms=latency_ms,
        )

        self.prediction_window.append(record)

        # Check for immediate alerts
        if error is not None and error > self.alert_thresholds.get("mae_threshold", 50.0) * 2:
            self._trigger_alert(
                "high_error",
                {
                    "bond_id": bond_id,
                    "error": error,
                    "predicted": predicted_value,
                    "actual": actual_value,
                },
            )

        # Compute metrics periodically
        if len(self.prediction_window) >= min(100, self.window_size):
            metrics = self._compute_metrics()
            self.metrics_history.append(metrics)
            self._check_alert_thresholds(metrics)

    def _compute_metrics(self) -> MonitoringMetrics:
        """Compute aggregated metrics from prediction window"""
        if len(self.prediction_window) == 0:
            return MonitoringMetrics(
                timestamp=datetime.now(),
                window_start=datetime.now(),
                window_end=datetime.now(),
                n_predictions=0,
                mean_error=0.0,
                rmse=0.0,
                mae=0.0,
                max_error=0.0,
                error_rate=0.0,
                prediction_latency_ms=0.0,
                model_version="unknown",
            )

        # Filter records with actual values
        records_with_actuals = [r for r in self.prediction_window if r.actual_value is not None]

        if len(records_with_actuals) == 0:
            return MonitoringMetrics(
                timestamp=datetime.now(),
                window_start=self.prediction_window[0].timestamp,
                window_end=self.prediction_window[-1].timestamp,
                n_predictions=len(self.prediction_window),
                mean_error=0.0,
                rmse=0.0,
                mae=0.0,
                max_error=0.0,
                error_rate=0.0,
                prediction_latency_ms=0.0,
                model_version=self.prediction_window[0].model_version,
            )

        errors = [r.error for r in records_with_actuals if r.error is not None]

        mean_error = np.mean(errors) if errors else 0.0
        rmse = np.sqrt(np.mean([e**2 for e in errors])) if errors else 0.0
        mae = np.mean(errors) if errors else 0.0
        max_error = np.max(errors) if errors else 0.0

        # Error rate (percentage of predictions with error > threshold)
        error_threshold = self.alert_thresholds.get("mae_threshold", 50.0)
        error_rate = sum(1 for e in errors if e > error_threshold) / len(errors) if errors else 0.0

        return MonitoringMetrics(
            timestamp=datetime.now(),
            window_start=self.prediction_window[0].timestamp,
            window_end=self.prediction_window[-1].timestamp,
            n_predictions=len(self.prediction_window),
            mean_error=mean_error,
            rmse=rmse,
            mae=mae,
            max_error=max_error,
            error_rate=error_rate,
            prediction_latency_ms=self._calculate_average_latency(),
            model_version=records_with_actuals[0].model_version,
        )

    def _check_alert_thresholds(self, metrics: MonitoringMetrics):
        """Check if metrics exceed alert thresholds"""
        alerts = []

        if metrics.rmse > self.alert_thresholds.get("rmse_threshold", 100.0):
            alerts.append(
                (
                    "high_rmse",
                    {"rmse": metrics.rmse, "threshold": self.alert_thresholds["rmse_threshold"]},
                )
            )

        if metrics.mae > self.alert_thresholds.get("mae_threshold", 50.0):
            alerts.append(
                (
                    "high_mae",
                    {"mae": metrics.mae, "threshold": self.alert_thresholds["mae_threshold"]},
                )
            )

        if metrics.error_rate > self.alert_thresholds.get("error_rate_threshold", 0.10):
            alerts.append(
                (
                    "high_error_rate",
                    {
                        "error_rate": metrics.error_rate,
                        "threshold": self.alert_thresholds["error_rate_threshold"],
                    },
                )
            )

        # Trigger alerts
        for alert_type, alert_data in alerts:
            self._trigger_alert(alert_type, alert_data)

    def _trigger_alert(self, alert_type: str, alert_data: Dict[str, Any]):
        """Trigger an alert"""
        alert = {
            "timestamp": datetime.now().isoformat(),
            "model_name": self.model_name,
            "alert_type": alert_type,
            "alert_data": alert_data,
            "severity": "high" if alert_type in ["high_rmse", "high_mae"] else "medium",
        }

        # Log alert
        logger.warning(f"ALERT [{self.model_name}]: {alert_type} - {alert_data}")

        # Save alert to file
        alert_file = os.path.join(
            self.monitoring_dir, f"alerts_{datetime.now().strftime('%Y%m%d')}.jsonl"
        )
        with open(alert_file, "a") as f:
            f.write(json.dumps(alert) + "\n")

        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}", exc_info=True)

    def register_alert_callback(self, callback: Callable[[Dict], None]):
        """Register a callback function for alerts"""
        self.alert_callbacks.append(callback)

    def get_current_metrics(self) -> Optional[MonitoringMetrics]:
        """Get current monitoring metrics"""
        if len(self.prediction_window) == 0:
            return None
        return self._compute_metrics()

    def get_metrics_history(
        self, start_time: datetime = None, end_time: datetime = None
    ) -> List[MonitoringMetrics]:
        """Get metrics history within time range"""
        if start_time is None:
            start_time = datetime.now() - timedelta(days=1)
        if end_time is None:
            end_time = datetime.now()

        return [m for m in self.metrics_history if start_time <= m.timestamp <= end_time]

    def save_metrics(self, filepath: str = None):
        """Save metrics to file"""
        if filepath is None:
            filepath = os.path.join(
                self.monitoring_dir, f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

        metrics_data = {
            "model_name": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "current_metrics": (
                asdict(self.get_current_metrics()) if self.get_current_metrics() else None
            ),
            "metrics_history": [asdict(m) for m in self.metrics_history[-100:]],  # Last 100
        }

        with open(filepath, "w") as f:
            json.dump(metrics_data, f, indent=2, default=str)

        logger.info(f"Saved monitoring metrics to {filepath}")

    def detect_performance_degradation(
        self,
        baseline_metrics: MonitoringMetrics,
        current_metrics: MonitoringMetrics,
        threshold: float = 0.2,
    ) -> bool:
        """
        Detect performance degradation compared to baseline

        Args:
            baseline_metrics: Baseline metrics
            current_metrics: Current metrics
            threshold: Relative degradation threshold (20% by default)

        Returns:
            True if degradation detected
        """
        if baseline_metrics.rmse == 0:
            return False

        rmse_increase = (current_metrics.rmse - baseline_metrics.rmse) / baseline_metrics.rmse

        if rmse_increase > threshold:
            self._trigger_alert(
                "performance_degradation",
                {
                    "baseline_rmse": baseline_metrics.rmse,
                    "current_rmse": current_metrics.rmse,
                    "degradation_pct": rmse_increase * 100,
                },
            )
            return True

        return False

    def _calculate_average_latency(self) -> float:
        """Calculate average prediction latency from recent predictions"""
        if len(self.prediction_window) == 0:
            return 0.0

        latencies = [r.latency_ms for r in self.prediction_window if r.latency_ms > 0]
        if not latencies:
            return 0.0

        return float(np.mean(latencies))


def create_slack_alert_callback(webhook_url: str) -> Callable[[Dict], None]:
    """Create Slack alert callback"""
    import requests

    def callback(alert: Dict):
        message = {
            "text": f"🚨 Model Alert: {alert['model_name']}",
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Alert Type:* {alert['alert_type']}\n*Model:* {alert['model_name']}\n*Time:* {alert['timestamp']}",
                    },
                },
            ],
        }

        try:
            requests.post(webhook_url, json=message, timeout=5)
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")

    return callback


def create_email_alert_callback(
    email_address: str, smtp_server: str = None, smtp_port: int = 587
) -> Callable[[Dict], None]:
    """
    Create email alert callback

    Args:
        email_address: Email address to send alerts to
        smtp_server: SMTP server (defaults to environment variable SMTP_SERVER)
        smtp_port: SMTP port (defaults to 587)

    Returns:
        Callback function for email alerts
    """
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    def callback(alert: Dict):
        """Send email alert"""
        try:
            # Get SMTP configuration from environment or use defaults
            smtp_host = smtp_server or os.getenv("SMTP_SERVER", "localhost")
            smtp_port_num = smtp_port or int(os.getenv("SMTP_PORT", "587"))
            smtp_user = os.getenv("SMTP_USER", "")
            smtp_password = os.getenv("SMTP_PASSWORD", "")

            # Create email message
            msg = MIMEMultipart()
            msg["From"] = os.getenv("SMTP_FROM", smtp_user or "noreply@bondtrader.local")
            msg["To"] = email_address
            msg["Subject"] = f"🚨 BondTrader Alert: {alert['alert_type']}"

            # Create email body
            body = f"""
            Model Monitoring Alert
            
            Model: {alert['model_name']}
            Alert Type: {alert['alert_type']}
            Severity: {alert.get('severity', 'medium')}
            Timestamp: {alert['timestamp']}
            
            Alert Data:
            {json.dumps(alert.get('alert_data', {}), indent=2)}
            
            Please review the model performance and take appropriate action.
            """

            msg.attach(MIMEText(body, "plain"))

            # Send email
            if smtp_host == "localhost" or not smtp_user:
                # For local development, just log
                logger.info(f"Email alert would be sent to {email_address}: {alert['alert_type']}")
                logger.debug(f"Email body: {body}")
            else:
                # Production: actually send email
                with smtplib.SMTP(smtp_host, smtp_port_num) as server:
                    if smtp_user and smtp_password:
                        server.starttls()
                        server.login(smtp_user, smtp_password)
                    server.send_message(msg)
                    logger.info(f"Email alert sent to {email_address}: {alert['alert_type']}")

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}", exc_info=True)

    return callback
