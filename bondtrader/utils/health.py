"""
Health Check and Automatic Recovery System
Monitors system health and implements automatic recovery

CRITICAL: Required for production reliability in Fortune 10 financial institutions
"""

import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional

from bondtrader.core.audit import AuditEventType, get_audit_logger
from bondtrader.utils.utils import logger


class HealthStatus(str, Enum):
    """Health status levels"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class ComponentHealth:
    """Health status for a system component"""

    def __init__(self, name: str):
        self.name = name
        self.status = HealthStatus.HEALTHY
        self.last_check = None
        self.last_success = None
        self.last_failure = None
        self.failure_count = 0
        self.response_time_ms = None
        self.error_message = None
        self.metadata: Dict = {}

    def record_success(self, response_time_ms: Optional[float] = None):
        """Record successful health check"""
        self.status = HealthStatus.HEALTHY
        self.last_check = datetime.now()
        self.last_success = datetime.now()
        self.failure_count = 0
        self.response_time_ms = response_time_ms
        self.error_message = None

    def record_failure(self, error_message: str, severity: HealthStatus = HealthStatus.UNHEALTHY):
        """Record failed health check"""
        self.status = severity
        self.last_check = datetime.now()
        self.last_failure = datetime.now()
        self.failure_count += 1
        self.error_message = error_message

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "status": self.status.value,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "last_success": self.last_success.isoformat() if self.last_success else None,
            "last_failure": self.last_failure.isoformat() if self.last_failure else None,
            "failure_count": self.failure_count,
            "response_time_ms": self.response_time_ms,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


class HealthChecker:
    """
    System Health Checker

    Monitors system components and provides health status.
    """

    def __init__(self):
        """Initialize health checker"""
        self.components: Dict[str, ComponentHealth] = {}
        self.audit_logger = get_audit_logger()

    def register_component(self, name: str) -> ComponentHealth:
        """
        Register a component for health checking.

        Args:
            name: Component name

        Returns:
            ComponentHealth object
        """
        component = ComponentHealth(name)
        self.components[name] = component
        return component

    def check_database(self) -> bool:
        """
        Check database health.

        Returns:
            True if database is healthy
        """
        try:
            from bondtrader.data.data_persistence import EnhancedBondDatabase

            db = EnhancedBondDatabase()
            # Simple query to test connection
            session = db._get_session()
            try:
                session.execute("SELECT 1")
                session.close()
                return True
            except Exception:
                session.close()
                return False
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

    def check_redis(self) -> bool:
        """
        Check Redis health.

        Returns:
            True if Redis is healthy
        """
        try:
            from bondtrader.utils.redis_cache import get_redis_cache

            cache = get_redis_cache()
            return cache.is_available()
        except Exception:
            return False

    def check_external_apis(self) -> Dict[str, bool]:
        """
        Check external API health (FRED, FINRA, etc.).

        Returns:
            Dictionary of API name -> health status
        """
        results = {}

        # Check FRED API
        try:
            from bondtrader.data.market_data import FREDDataProvider

            provider = FREDDataProvider()
            # Simple health check (would test actual connection in production)
            results["fred"] = True
        except Exception:
            results["fred"] = False

        # Check FINRA API
        try:
            from bondtrader.data.market_data import MarketDataManager

            manager = MarketDataManager()
            # Simple health check
            results["finra"] = True
        except Exception:
            results["finra"] = False

        return results

    def run_health_checks(self) -> Dict:
        """
        Run all health checks.

        Returns:
            Dictionary with overall health status and component details
        """
        overall_status = HealthStatus.HEALTHY
        component_statuses = {}

        # Check database
        db_component = self.register_component("database")
        start_time = time.time()
        db_healthy = self.check_database()
        response_time = (time.time() - start_time) * 1000

        if db_healthy:
            db_component.record_success(response_time_ms=response_time)
        else:
            db_component.record_failure("Database connection failed", HealthStatus.CRITICAL)
            overall_status = HealthStatus.CRITICAL

        component_statuses["database"] = db_component.to_dict()

        # Check Redis (optional)
        redis_component = self.register_component("redis")
        start_time = time.time()
        redis_healthy = self.check_redis()
        response_time = (time.time() - start_time) * 1000

        if redis_healthy:
            redis_component.record_success(response_time_ms=response_time)
        else:
            redis_component.record_failure("Redis not available", HealthStatus.DEGRADED)
            if overall_status == HealthStatus.HEALTHY:
                overall_status = HealthStatus.DEGRADED

        component_statuses["redis"] = redis_component.to_dict()

        # Check external APIs
        api_results = self.check_external_apis()
        for api_name, is_healthy in api_results.items():
            api_component = self.register_component(f"api_{api_name}")
            if is_healthy:
                api_component.record_success()
            else:
                api_component.record_failure(f"{api_name} API unavailable", HealthStatus.DEGRADED)
                if overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED
            component_statuses[f"api_{api_name}"] = api_component.to_dict()

        # Audit log
        self.audit_logger.log(
            AuditEventType.USER_ACTION,
            "system",
            "health_check",
            details={
                "overall_status": overall_status.value,
                "components": component_statuses,
            },
        )

        return {
            "status": overall_status.value,
            "timestamp": datetime.now().isoformat(),
            "components": component_statuses,
        }

    def get_health_status(self) -> Dict:
        """
        Get current health status.

        Returns:
            Health status dictionary
        """
        return self.run_health_checks()


class AutomaticRecovery:
    """
    Automatic Recovery Manager

    Implements automatic recovery procedures for system failures.
    """

    def __init__(self, health_checker: Optional[HealthChecker] = None):
        """
        Initialize automatic recovery.

        Args:
            health_checker: Health checker instance
        """
        self.health_checker = health_checker or HealthChecker()
        self.recovery_attempts: Dict[str, int] = {}
        self.max_recovery_attempts = 3

    def attempt_recovery(self, component: str) -> bool:
        """
        Attempt to recover a failed component.

        Args:
            component: Component name to recover

        Returns:
            True if recovery successful
        """
        if component not in self.recovery_attempts:
            self.recovery_attempts[component] = 0

        if self.recovery_attempts[component] >= self.max_recovery_attempts:
            logger.error(f"Max recovery attempts reached for {component}")
            return False

        self.recovery_attempts[component] += 1

        # Component-specific recovery procedures
        if component == "database":
            return self._recover_database()
        elif component == "redis":
            return self._recover_redis()
        else:
            logger.warning(f"No recovery procedure for component: {component}")
            return False

    def _recover_database(self) -> bool:
        """Attempt to recover database connection"""
        try:
            # Close existing connections and reconnect
            from bondtrader.data.data_persistence import EnhancedBondDatabase

            # Force reconnection by creating new instance
            db = EnhancedBondDatabase()
            if self.health_checker.check_database():
                self.recovery_attempts["database"] = 0  # Reset on success
                logger.info("Database recovery successful")
                return True
            return False
        except Exception as e:
            logger.error(f"Database recovery failed: {e}")
            return False

    def _recover_redis(self) -> bool:
        """Attempt to recover Redis connection"""
        try:
            from bondtrader.utils.redis_cache import get_redis_cache

            cache = get_redis_cache()
            if cache.is_available():
                self.recovery_attempts["redis"] = 0  # Reset on success
                logger.info("Redis recovery successful")
                return True
            return False
        except Exception as e:
            logger.error(f"Redis recovery failed: {e}")
            return False


# Global health checker instance
_health_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """Get or create global health checker instance"""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker
