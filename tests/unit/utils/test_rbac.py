"""
Unit tests for RBAC (Role-Based Access Control) utilities
"""

import pytest

from bondtrader.core.audit import AuditEventType
from bondtrader.utils.auth import AuthorizationError
from bondtrader.utils.rbac import Permission, RBACManager, Role


@pytest.mark.unit
class TestRBACManager:
    """Test RBACManager functionality"""

    @pytest.fixture
    def rbac_manager(self):
        """Create RBAC manager"""
        return RBACManager()

    def test_get_permissions_read_only(self, rbac_manager):
        """Test getting permissions for read-only role"""
        permissions = rbac_manager.get_permissions([Role.READ_ONLY.value])
        assert Permission.BOND_READ in permissions
        assert Permission.BOND_CREATE not in permissions

    def test_get_permissions_trader(self, rbac_manager):
        """Test getting permissions for trader role"""
        permissions = rbac_manager.get_permissions([Role.TRADER.value])
        assert Permission.BOND_CREATE in permissions
        assert Permission.ARBITRAGE_EXECUTE in permissions

    def test_get_permissions_admin(self, rbac_manager):
        """Test getting permissions for admin role"""
        permissions = rbac_manager.get_permissions([Role.ADMIN.value])
        assert Permission.USER_CREATE in permissions
        assert Permission.CONFIG_UPDATE in permissions

    def test_get_permissions_multiple_roles(self, rbac_manager):
        """Test getting permissions for multiple roles"""
        permissions = rbac_manager.get_permissions([Role.READ_ONLY.value, Role.TRADER.value])
        # Should have permissions from both roles
        assert Permission.BOND_READ in permissions
        assert Permission.BOND_CREATE in permissions

    def test_get_permissions_invalid_role(self, rbac_manager):
        """Test getting permissions for invalid role"""
        permissions = rbac_manager.get_permissions(["invalid_role"])
        assert len(permissions) == 0

    def test_check_permission_has_permission(self, rbac_manager):
        """Test checking permission when user has it"""
        has_permission = rbac_manager.check_permission([Role.TRADER.value], Permission.BOND_CREATE)
        assert has_permission is True

    def test_check_permission_no_permission(self, rbac_manager):
        """Test checking permission when user doesn't have it"""
        has_permission = rbac_manager.check_permission([Role.READ_ONLY.value], Permission.BOND_CREATE)
        assert has_permission is False

    def test_check_permission_with_resource_id(self, rbac_manager):
        """Test checking permission with resource ID"""
        has_permission = rbac_manager.check_permission([Role.TRADER.value], Permission.BOND_CREATE, resource_id="BOND-001")
        assert has_permission is True

    def test_require_permission_decorator(self, rbac_manager):
        """Test require_permission decorator"""

        @rbac_manager.require_permission(Permission.BOND_CREATE)
        def create_bond(user_roles=None):
            return "success"

        # Test with user who has permission
        result = create_bond(user_roles=[Role.TRADER.value])
        assert result == "success"

    def test_require_permission_decorator_no_permission(self, rbac_manager):
        """Test require_permission decorator without permission"""

        @rbac_manager.require_permission(Permission.BOND_CREATE)
        def create_bond(user_roles=None):
            return "success"

        # Should raise AuthorizationError when user doesn't have permission
        with pytest.raises(AuthorizationError):
            create_bond(user_roles=[Role.READ_ONLY.value])


@pytest.mark.unit
class TestRoleEnum:
    """Test Role enum"""

    def test_role_values(self):
        """Test role enum values"""
        assert Role.READ_ONLY.value == "read_only"
        assert Role.TRADER.value == "trader"
        assert Role.ADMIN.value == "admin"

    def test_role_from_value(self):
        """Test creating role from value"""
        role = Role("read_only")
        assert role == Role.READ_ONLY


@pytest.mark.unit
class TestPermissionEnum:
    """Test Permission enum"""

    def test_permission_values(self):
        """Test permission enum values"""
        assert Permission.BOND_READ.value == "bond:read"
        assert Permission.BOND_CREATE.value == "bond:create"

    def test_permission_from_value(self):
        """Test creating permission from value"""
        permission = Permission("bond:read")
        assert permission == Permission.BOND_READ
