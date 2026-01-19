"""
Data Retention Policy Management
Implements 7-year retention policy for financial data (SOX compliance)

CRITICAL: Required for regulatory compliance in Fortune 10 financial institutions
"""

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from bondtrader.core.audit import AuditEventType, get_audit_logger
from bondtrader.utils.utils import logger


class RetentionPolicy:
    """Data retention policy configuration"""

    def __init__(
        self,
        retention_years: int = 7,
        archival_enabled: bool = True,
        archival_path: Optional[str] = None,
    ):
        """
        Initialize retention policy.

        Args:
            retention_years: Years to retain data (default: 7 for SOX compliance)
            archival_enabled: Whether to archive before deletion
            archival_path: Path for archived data
        """
        self.retention_years = retention_years
        self.archival_enabled = archival_enabled
        self.archival_path = archival_path or os.path.join("archived_data", datetime.now().strftime("%Y"))

        if archival_enabled:
            os.makedirs(self.archival_path, exist_ok=True)

    def get_retention_date(self) -> datetime:
        """Get cutoff date for data retention"""
        return datetime.now() - timedelta(days=365 * self.retention_years)

    def should_retain(self, record_date: datetime) -> bool:
        """Check if record should be retained"""
        cutoff = self.get_retention_date()
        return record_date >= cutoff


class DataRetentionManager:
    """
    Data Retention Manager

    Manages data retention policies and archival for regulatory compliance.
    """

    def __init__(self, policy: Optional[RetentionPolicy] = None):
        """
        Initialize data retention manager.

        Args:
            policy: Retention policy (default: 7-year SOX compliance policy)
        """
        self.policy = policy or RetentionPolicy()
        self.audit_logger = get_audit_logger()

    def archive_data(self, data: Dict, record_type: str, record_id: str) -> bool:
        """
        Archive data before deletion.

        Args:
            data: Data to archive
            record_type: Type of record (e.g., "bond", "valuation")
            record_id: Record identifier

        Returns:
            True if archived successfully
        """
        if not self.policy.archival_enabled:
            return False

        try:
            import json

            # Create archive directory structure
            archive_dir = os.path.join(
                self.policy.archival_path,
                record_type,
                datetime.now().strftime("%Y-%m"),
            )
            os.makedirs(archive_dir, exist_ok=True)

            # Save archived data
            archive_file = os.path.join(archive_dir, f"{record_id}_{datetime.now().isoformat()}.json")
            with open(archive_file, "w") as f:
                json.dump(data, f, indent=2, default=str)

            # Audit log
            self.audit_logger.log(
                AuditEventType.DATA_ACCESSED,
                record_id,
                "data_archived",
                details={"record_type": record_type, "archive_path": archive_file},
                compliance_tags=["SOX"],
            )

            return True
        except Exception as e:
            logger.error(f"Failed to archive data: {e}")
            return False

    def get_expired_records(self, records: List[Dict], date_field: str = "created_at") -> List[Dict]:
        """
        Get records that have exceeded retention period.

        Args:
            records: List of records with date fields
            date_field: Field name containing date

        Returns:
            List of expired records
        """
        cutoff = self.policy.get_retention_date()
        expired = []

        for record in records:
            try:
                record_date_str = record.get(date_field)
                if not record_date_str:
                    continue

                # Parse date (handle ISO format)
                if isinstance(record_date_str, str):
                    record_date = datetime.fromisoformat(record_date_str.replace("Z", "+00:00"))
                else:
                    record_date = record_date_str

                if record_date < cutoff:
                    expired.append(record)
            except Exception as e:
                logger.warning(f"Error parsing date for record: {e}")
                continue

        return expired

    def apply_retention_policy(
        self, records: List[Dict], record_type: str, delete_callback: Optional[callable] = None
    ) -> Dict:
        """
        Apply retention policy to records.

        Args:
            records: List of records to process
            record_type: Type of records
            delete_callback: Optional callback to delete records

        Returns:
            Dictionary with retention statistics
        """
        expired = self.get_expired_records(records)

        archived_count = 0
        deleted_count = 0

        for record in expired:
            # Archive if enabled
            if self.policy.archival_enabled:
                record_id = record.get("id") or record.get("bond_id") or "unknown"
                if self.archive_data(record, record_type, str(record_id)):
                    archived_count += 1

            # Delete if callback provided
            if delete_callback:
                try:
                    delete_callback(record)
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Failed to delete record: {e}")

        # Audit log
        self.audit_logger.log(
            AuditEventType.DATA_ACCESSED,
            "system",
            "retention_policy_applied",
            details={
                "record_type": record_type,
                "total_records": len(records),
                "expired_records": len(expired),
                "archived": archived_count,
                "deleted": deleted_count,
            },
            compliance_tags=["SOX"],
        )

        return {
            "total_records": len(records),
            "expired_records": len(expired),
            "archived": archived_count,
            "deleted": deleted_count,
        }


# Global retention manager instance
_retention_manager: Optional[DataRetentionManager] = None


def get_retention_manager() -> DataRetentionManager:
    """Get or create global retention manager instance"""
    global _retention_manager
    if _retention_manager is None:
        _retention_manager = DataRetentionManager()
    return _retention_manager
