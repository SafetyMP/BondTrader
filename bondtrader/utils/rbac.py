"""
Role-Based Access Control (RBAC) System
Implements granular permissions for financial system security

CRITICAL: Required for SOX compliance and security in Fortune 10 financial institutions
"""

from enum import Enum
from typing import Dict, List, Optional, Set

from bondtrader.core.audit import AuditEventType, get_audit_logger
from bondtrader.utils.auth import AuthorizationError
from bondtrader.utils.utils import logger


class Role(str, Enum):
    """System roles with hierarchical permissions"""

    READ_ONLY = "read_only"  # Can only read data
    TRADER = "trader"  # Can create bonds, view valuations
    RISK_MANAGER = "risk_manager"  # Can view risk metrics, run reports
    ADMIN = "admin"  # Full access including user management
    AUDITOR = "auditor"  # Read-only access to audit logs and reports


class Permission(str, Enum):
    """Granular permissions for resources"""

    # Bond permissions
    BOND_READ = "bond:read"
    BOND_CREATE = "bond:create"
    BOND_UPDATE = "bond:update"
    BOND_DELETE = "bond:delete"

    # Valuation permissions
    VALUATION_READ = "valuation:read"
    VALUATION_CALCULATE = "valuation:calculate"

    # Risk permissions
    RISK_READ = "risk:read"
    RISK_CALCULATE = "risk:calculate"
    RISK_EXPORT = "risk:export"

    # ML permissions
    ML_PREDICT = "ml:predict"
    ML_TRAIN = "ml:train"
    ML_DEPLOY = "ml:deploy"

    # Arbitrage permissions
    ARBITRAGE_READ = "arbitrage:read"
    ARBITRAGE_EXECUTE = "arbitrage:execute"

    # Portfolio permissions
    PORTFOLIO_READ = "portfolio:read"
    PORTFOLIO_CREATE = "portfolio:create"
    PORTFOLIO_UPDATE = "portfolio:update"

    # User management permissions
    USER_READ = "user:read"
    USER_CREATE = "user:create"
    USER_UPDATE = "user:update"
    USER_DELETE = "user:delete"

    # Audit permissions
    AUDIT_READ = "audit:read"
    AUDIT_EXPORT = "audit:export"

    # Configuration permissions
    CONFIG_READ = "config:read"
    CONFIG_UPDATE = "config:update"


# Role to permissions mapping
ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
    Role.READ_ONLY: {
        Permission.BOND_READ,
        Permission.VALUATION_READ,
        Permission.RISK_READ,
        Permission.ARBITRAGE_READ,
        Permission.PORTFOLIO_READ,
    },
    Role.TRADER: {
        Permission.BOND_READ,
        Permission.BOND_CREATE,
        Permission.BOND_UPDATE,
        Permission.VALUATION_READ,
        Permission.VALUATION_CALCULATE,
        Permission.RISK_READ,
        Permission.ML_PREDICT,
        Permission.ARBITRAGE_READ,
        Permission.ARBITRAGE_EXECUTE,
        Permission.PORTFOLIO_READ,
        Permission.PORTFOLIO_CREATE,
        Permission.PORTFOLIO_UPDATE,
    },
    Role.RISK_MANAGER: {
        Permission.BOND_READ,
        Permission.VALUATION_READ,
        Permission.VALUATION_CALCULATE,
        Permission.RISK_READ,
        Permission.RISK_CALCULATE,
        Permission.RISK_EXPORT,
        Permission.ML_PREDICT,
        Permission.ARBITRAGE_READ,
        Permission.PORTFOLIO_READ,
        Permission.AUDIT_READ,
    },
    Role.ADMIN: {
        # Admin has all permissions
        *[p for p in Permission],
    },
    Role.AUDITOR: {
        Permission.BOND_READ,
        Permission.VALUATION_READ,
        Permission.RISK_READ,
        Permission.AUDIT_READ,
        Permission.AUDIT_EXPORT,
        Permission.CONFIG_READ,
    },
}


class RBACManager:
    """
    Role-Based Access Control Manager

    Manages roles, permissions, and access control for the system.
    """

    def __init__(self):
        """Initialize RBAC manager"""
        self.audit_logger = get_audit_logger()

    def get_permissions(self, roles: List[str]) -> Set[Permission]:
        """
        Get all permissions for a set of roles.

        Args:
            roles: List of role names

        Returns:
            Set of permissions
        """
        permissions = set()
        for role_name in roles:
            try:
                role = Role(role_name)
                permissions.update(ROLE_PERMISSIONS.get(role, set()))
            except ValueError:
                logger.warning(f"Unknown role: {role_name}")
        return permissions

    def has_permission(self, roles: List[str], permission: Permission) -> bool:
        """
        Check if user with given roles has a specific permission.

        Args:
            roles: List of role names
            permission: Permission to check

        Returns:
            True if user has permission
        """
        user_permissions = self.get_permissions(roles)
        return permission in user_permissions

    def require_permission(self, permission: Permission):
        """
        Decorator to require a specific permission.

        Usage:
            @rbac_manager.require_permission(Permission.BOND_CREATE)
            def create_bond(...):
                ...
        """
        from functools import wraps

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Extract user roles from context (implementation depends on framework)
                # For FastAPI, would get from request context
                # For Streamlit, would get from session state
                user_roles = kwargs.get("user_roles", [])
                if not user_roles:
                    # Try to get from function context
                    import inspect

                    frame = inspect.currentframe()
                    try:
                        # This is a simplified version - actual implementation
                        # would get from request/session context
                        user_roles = []
                    finally:
                        del frame

                if not self.has_permission(user_roles, permission):
                    # Log unauthorized access attempt
                    self.audit_logger.log(
                        AuditEventType.USER_ACTION,
                        "system",
                        "unauthorized_access_attempt",
                        details={
                            "permission": permission.value,
                            "roles": user_roles,
                            "function": func.__name__,
                        },
                        compliance_tags=["SOX"],
                    )
                    raise AuthorizationError(f"Permission denied: {permission.value} required")

                return func(*args, **kwargs)

            return wrapper

        return decorator

    def check_permission(self, roles: List[str], permission: Permission, resource_id: Optional[str] = None) -> bool:
        """
        Check permission and log the access attempt.

        Args:
            roles: User roles
            permission: Permission to check
            resource_id: Optional resource ID being accessed

        Returns:
            True if permission granted
        """
        has_access = self.has_permission(roles, permission)

        # Log access attempt
        self.audit_logger.log(
            AuditEventType.DATA_ACCESSED if has_access else AuditEventType.USER_ACTION,
            resource_id or "system",
            "permission_check",
            details={
                "permission": permission.value,
                "granted": has_access,
                "roles": roles,
            },
            compliance_tags=["SOX"],
        )

        return has_access


# Global RBAC manager instance
_rbac_manager: Optional[RBACManager] = None


def get_rbac_manager() -> RBACManager:
    """Get or create global RBAC manager instance"""
    global _rbac_manager
    if _rbac_manager is None:
        _rbac_manager = RBACManager()
    return _rbac_manager
