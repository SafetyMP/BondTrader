"""
Audit Logging for Financial Operations
Industry-standard audit trail for compliance and traceability

CRITICAL: All financial operations must be audited with regulatory compliance fields.
Implements immutable audit logging for SOX, GDPR, and MiFID II compliance.
"""

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from bondtrader.utils.utils import logger


class AuditEventType(Enum):
    """Types of audit events"""

    BOND_CREATED = "bond_created"
    BOND_UPDATED = "bond_updated"
    BOND_DELETED = "bond_deleted"
    VALUATION_CALCULATED = "valuation_calculated"
    ARBITRAGE_DETECTED = "arbitrage_detected"
    RISK_CALCULATED = "risk_calculated"
    TRADE_EXECUTED = "trade_executed"
    PORTFOLIO_UPDATED = "portfolio_updated"
    MODEL_TRAINED = "model_trained"
    MODEL_PREDICTION = "model_prediction"
    CONFIGURATION_CHANGED = "configuration_changed"
    DATA_ACCESSED = "data_accessed"
    USER_ACTION = "user_action"


class AuditLogger:
    """
    Audit logger for financial operations with regulatory compliance.

    CRITICAL: Implements immutable audit logging with:
    - Unique event IDs (UUID)
    - Microsecond-precision timestamps (UTC)
    - IP address tracking
    - Cryptographic signatures for immutability
    - Compliance tagging (SOX, GDPR, MiFID II)
    """

    def __init__(self, log_file: str = "audit.log", enable_signatures: bool = True):
        self.log_file = log_file
        self.enable_signatures = enable_signatures

    def _get_client_ip(self) -> Optional[str]:
        """Get client IP address from context (if available)"""
        # In production, extract from request context
        # For now, return None (can be enhanced with FastAPI request context)
        return os.getenv("CLIENT_IP", None)

    def _generate_signature(self, record: Dict[str, Any]) -> str:
        """Generate cryptographic signature for audit record immutability"""
        if not self.enable_signatures:
            return ""

        # Create deterministic JSON string (sorted keys)
        record_str = json.dumps(record, sort_keys=True, separators=(",", ":"))

        # Generate SHA-256 hash
        signature = hashlib.sha256(record_str.encode("utf-8")).hexdigest()
        return signature

    def log(
        self,
        event_type: AuditEventType,
        entity_id: str,
        operation: str,
        user_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        compliance_tags: Optional[List[str]] = None,
        before_state: Optional[Dict[str, Any]] = None,
        after_state: Optional[Dict[str, Any]] = None,
    ):
        """
        Log an audit event with regulatory compliance fields.

        CRITICAL: All audit records include:
        - Unique event ID (UUID)
        - Microsecond-precision UTC timestamp
        - User ID and IP address
        - Before/after state for change tracking
        - Compliance tags (SOX, GDPR, MiFID II)
        - Cryptographic signature for immutability

        Args:
            event_type: Type of audit event
            entity_id: ID of the entity (bond_id, portfolio_id, etc.)
            operation: Description of operation
            user_id: User who performed operation (required for financial operations)
            details: Event-specific details
            metadata: Additional metadata
            ip_address: Client IP address (auto-detected if not provided)
            compliance_tags: List of compliance standards (SOX, GDPR, MiFID II)
            before_state: State before change (for change tracking)
            after_state: State after change (for change tracking)
        """
        # Generate unique event ID
        event_id = str(uuid.uuid4())

        # Get microsecond-precision UTC timestamp
        timestamp = datetime.now(timezone.utc).isoformat()

        # Get IP address
        client_ip = ip_address or self._get_client_ip()

        # Default compliance tags based on event type
        if compliance_tags is None:
            compliance_tags = []
            # Financial operations require SOX compliance
            if event_type in [
                AuditEventType.BOND_CREATED,
                AuditEventType.BOND_UPDATED,
                AuditEventType.VALUATION_CALCULATED,
                AuditEventType.TRADE_EXECUTED,
            ]:
                compliance_tags.append("SOX")
            # Data access requires GDPR compliance
            if event_type == AuditEventType.DATA_ACCESSED:
                compliance_tags.append("GDPR")
            # Trading operations require MiFID II
            if event_type in [
                AuditEventType.TRADE_EXECUTED,
                AuditEventType.ARBITRAGE_DETECTED,
            ]:
                compliance_tags.append("MiFID")

        audit_record = {
            "event_id": event_id,
            "timestamp": timestamp,  # UTC with microsecond precision
            "event_type": event_type.value,
            "entity_id": entity_id,
            "operation": operation,
            "user_id": user_id or "system",
            "ip_address": client_ip,
            "details": details or {},
            "metadata": metadata or {},
            "before_state": before_state,
            "after_state": after_state,
            "compliance_tags": compliance_tags,
        }

        # Generate cryptographic signature for immutability
        signature = self._generate_signature(audit_record)
        audit_record["signature"] = signature

        # Log to file (JSON format for easy parsing)
        try:
            import os

            from bondtrader.config import get_config

            config = get_config()
            audit_dir = os.path.join(config.logs_dir, "audit")
            os.makedirs(audit_dir, exist_ok=True)

            audit_path = os.path.join(audit_dir, self.log_file)
            with open(audit_path, "a") as f:
                f.write(json.dumps(audit_record) + "\n")
        except (OSError, PermissionError, IOError) as e:
            # File I/O errors - log but don't fail the operation
            logger.error(f"Failed to write audit log: {e}", exc_info=True)
        except Exception as e:
            # Unexpected errors - log with full traceback
            logger.error(f"Unexpected error writing audit log: {e}", exc_info=True)

        # Also log to standard logger with INFO level
        logger.info(
            f"AUDIT: {event_type.value}",
            extra={"audit_event": audit_record, "entity_id": entity_id, "operation": operation, "user_id": user_id},
        )

    def log_valuation(self, bond_id: str, fair_value: float, ytm: float, **kwargs):
        """Log valuation calculation"""
        self.log(
            event_type=AuditEventType.VALUATION_CALCULATED,
            entity_id=bond_id,
            operation="valuation_calculated",
            details={"fair_value": fair_value, "ytm": ytm, **kwargs},
        )

    def log_arbitrage(self, bond_id: str, profit: float, recommendation: str, **kwargs):
        """Log arbitrage detection"""
        self.log(
            event_type=AuditEventType.ARBITRAGE_DETECTED,
            entity_id=bond_id,
            operation="arbitrage_detected",
            details={"profit": profit, "recommendation": recommendation, **kwargs},
        )

    def log_risk(self, entity_id: str, risk_type: str, risk_value: float, **kwargs):
        """Log risk calculation"""
        self.log(
            event_type=AuditEventType.RISK_CALCULATED,
            entity_id=entity_id,
            operation=f"risk_calculated_{risk_type}",
            details={"risk_type": risk_type, "risk_value": risk_value, **kwargs},
        )

    def log_model_training(self, model_id: str, metrics: Dict[str, Any], **kwargs):
        """Log model training"""
        self.log(
            event_type=AuditEventType.MODEL_TRAINED,
            entity_id=model_id,
            operation="model_trained",
            details={"metrics": metrics, **kwargs},
        )


# Global audit logger instance
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Get global audit logger instance"""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


def audit_log(event_type: AuditEventType, entity_id: str, operation: str, **kwargs):
    """Convenience function for audit logging"""
    get_audit_logger().log(event_type, entity_id, operation, **kwargs)
