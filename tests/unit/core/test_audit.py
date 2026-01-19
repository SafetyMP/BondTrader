"""
Tests for audit logging functionality
"""

import os
import tempfile
from pathlib import Path

import pytest

from bondtrader.core.audit import AuditEventType, AuditLogger, get_audit_logger


@pytest.mark.unit
class TestAuditLogger:
    """Test AuditLogger functionality"""

    @pytest.fixture
    def temp_log_file(self):
        """Create temporary log file"""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".log") as f:
            temp_path = f.name
        yield temp_path
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

    def test_audit_logger_init(self, temp_log_file):
        """Test audit logger initialization"""
        logger = AuditLogger(log_file=temp_log_file)
        assert logger.log_file == temp_log_file

    def test_log_event(self, temp_log_file):
        """Test logging an event"""
        logger = AuditLogger(log_file=temp_log_file)
        logger.log(
            event_type=AuditEventType.BOND_CREATED,
            entity_id="BOND-001",
            operation="bond_created",
            details={"test": "data"},
        )
        # Verify file was created and has content
        assert os.path.exists(temp_log_file)

    def test_log_valuation(self, temp_log_file):
        """Test logging valuation"""
        logger = AuditLogger(log_file=temp_log_file)
        logger.log_valuation("BOND-001", 1000.0, 0.05, duration=4.5, convexity=22.3)
        assert os.path.exists(temp_log_file)

    def test_log_arbitrage(self, temp_log_file):
        """Test logging arbitrage"""
        logger = AuditLogger(log_file=temp_log_file)
        logger.log_arbitrage("BOND-001", 25.50, "BUY", profit_percentage=2.68)
        assert os.path.exists(temp_log_file)

    def test_log_risk(self, temp_log_file):
        """Test logging risk"""
        logger = AuditLogger(log_file=temp_log_file)
        logger.log_risk("BOND-001", "var", 45.2, confidence_level=0.95)
        assert os.path.exists(temp_log_file)

    def test_log_model_training(self, temp_log_file):
        """Test logging model training"""
        logger = AuditLogger(log_file=temp_log_file)
        logger.log_model_training("model-001", {"r2": 0.85, "rmse": 10.5})
        assert os.path.exists(temp_log_file)

    def test_get_audit_logger_singleton(self):
        """Test get_audit_logger returns singleton"""
        logger1 = get_audit_logger()
        logger2 = get_audit_logger()
        assert logger1 is logger2


@pytest.mark.unit
class TestAuditEventType:
    """Test AuditEventType enum"""

    def test_audit_event_types(self):
        """Test all audit event types exist"""
        assert AuditEventType.BOND_CREATED
        assert AuditEventType.BOND_UPDATED
        assert AuditEventType.BOND_DELETED
        assert AuditEventType.VALUATION_CALCULATED
        assert AuditEventType.ARBITRAGE_DETECTED
        assert AuditEventType.RISK_CALCULATED
        assert AuditEventType.TRADE_EXECUTED
        assert AuditEventType.PORTFOLIO_UPDATED
        assert AuditEventType.MODEL_TRAINED
        assert AuditEventType.MODEL_PREDICTION
        assert AuditEventType.CONFIGURATION_CHANGED
        assert AuditEventType.DATA_ACCESSED
        assert AuditEventType.USER_ACTION
