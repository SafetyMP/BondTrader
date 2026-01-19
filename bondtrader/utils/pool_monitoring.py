"""
Connection Pool Monitoring
Monitors database connection pool health and usage

CRITICAL: Prevents resource exhaustion in production
"""

from typing import Dict, Optional

from bondtrader.core.audit import AuditEventType, get_audit_logger
from bondtrader.utils.utils import logger


class PoolMonitor:
    """
    Connection Pool Monitor

    Monitors connection pool health and provides statistics.
    """

    def __init__(self):
        """Initialize pool monitor"""
        self.audit_logger = get_audit_logger()

    def get_pool_stats(self, database) -> Dict:
        """
        Get connection pool statistics.

        Args:
            database: EnhancedBondDatabase instance

        Returns:
            Dictionary with pool statistics
        """
        if not hasattr(database, "engine"):
            return {"error": "Database engine not available"}

        pool = database.engine.pool

        stats = {
            "pool_size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "invalid": pool.invalid(),
        }

        # Calculate utilization
        total_connections = stats["pool_size"] + stats["overflow"]
        if total_connections > 0:
            stats["utilization_pct"] = (stats["checked_out"] / total_connections) * 100
        else:
            stats["utilization_pct"] = 0

        # Check for warnings
        if stats["utilization_pct"] > 80:
            logger.warning(f"High pool utilization: {stats['utilization_pct']:.1f}%")
            self.audit_logger.log(
                AuditEventType.USER_ACTION,
                "system",
                "pool_high_utilization",
                details=stats,
            )

        if stats["invalid"] > 0:
            logger.warning(f"Invalid connections in pool: {stats['invalid']}")
            self.audit_logger.log(
                AuditEventType.USER_ACTION,
                "system",
                "pool_invalid_connections",
                details=stats,
            )

        return stats

    def check_pool_health(self, database) -> Dict:
        """
        Check connection pool health.

        Args:
            database: EnhancedBondDatabase instance

        Returns:
            Health status dictionary
        """
        stats = self.get_pool_stats(database)

        health_status = "healthy"
        issues = []

        # Check utilization
        if stats.get("utilization_pct", 0) > 90:
            health_status = "critical"
            issues.append("Pool utilization > 90%")
        elif stats.get("utilization_pct", 0) > 80:
            health_status = "degraded"
            issues.append("Pool utilization > 80%")

        # Check for invalid connections
        if stats.get("invalid", 0) > 0:
            health_status = "degraded"
            issues.append(f"Invalid connections: {stats['invalid']}")

        return {
            "status": health_status,
            "stats": stats,
            "issues": issues,
        }


# Global pool monitor instance
_pool_monitor: Optional[PoolMonitor] = None


def get_pool_monitor() -> PoolMonitor:
    """Get or create global pool monitor instance"""
    global _pool_monitor
    if _pool_monitor is None:
        _pool_monitor = PoolMonitor()
    return _pool_monitor
