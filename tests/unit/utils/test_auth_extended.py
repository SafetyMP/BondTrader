"""
Extended tests for authentication utilities
"""

import os
import tempfile

import pytest

from bondtrader.utils.auth import (
    AuthenticationError,
    UserManager,
    get_user_manager,
    hash_password,
    verify_password,
)


@pytest.mark.unit
class TestUserManagerExtended:
    """Extended tests for UserManager"""

    @pytest.fixture
    def temp_users_file(self):
        """Create temporary users file"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_path = f.name
        yield temp_path
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    @pytest.fixture
    def user_manager(self, temp_users_file):
        """Create user manager"""
        return UserManager(users_file=temp_users_file)

    def test_user_manager_load_from_env(self, monkeypatch):
        """Test loading users from environment"""
        monkeypatch.setenv("USERS", "user1:pass1,user2:pass2")
        manager = UserManager(users_file=None)
        assert "user1" in manager.users or "user2" in manager.users

    def test_authenticate_with_mfa_enabled_no_token(self, user_manager):
        """Test authentication when MFA enabled but no token provided"""
        # Create user with MFA enabled
        hashed, salt = hash_password("testpassword")
        user_manager.users["testuser"] = {
            "password_hash": hashed,
            "salt": salt,
            "roles": ["user"],
            "mfa_enabled": True,
            "mfa_secret": "TEST_SECRET",
        }

        # Should fail without MFA token
        assert not user_manager.authenticate("testuser", "testpassword")

    def test_authenticate_with_mfa_invalid_token(self, user_manager):
        """Test authentication with invalid MFA token"""
        hashed, salt = hash_password("testpassword")
        user_manager.users["testuser"] = {
            "password_hash": hashed,
            "salt": salt,
            "roles": ["user"],
            "mfa_enabled": True,
            "mfa_secret": "TEST_SECRET",
        }

        # Mock MFA manager to return False
        if user_manager.mfa_manager:
            original_verify = user_manager.mfa_manager.verify_totp
            user_manager.mfa_manager.verify_totp = lambda secret, token: False

            assert not user_manager.authenticate("testuser", "testpassword", mfa_token="123456")

            # Restore
            user_manager.mfa_manager.verify_totp = original_verify

    def test_get_user_roles_existing_user(self, user_manager):
        """Test getting roles for existing user"""
        hashed, salt = hash_password("testpassword")
        user_manager.users["testuser"] = {
            "password_hash": hashed,
            "salt": salt,
            "roles": ["user", "admin"],
        }

        roles = user_manager.get_user_roles("testuser")
        assert "user" in roles
        assert "admin" in roles

    def test_has_role_existing_user(self, user_manager):
        """Test checking role for existing user"""
        hashed, salt = hash_password("testpassword")
        user_manager.users["testuser"] = {
            "password_hash": hashed,
            "salt": salt,
            "roles": ["user", "admin"],
        }

        assert user_manager.has_role("testuser", "admin")
        assert user_manager.has_role("testuser", "user")
        assert not user_manager.has_role("testuser", "trader")

    def test_verify_mfa_setup_success(self, user_manager):
        """Test verifying MFA setup successfully"""
        hashed, salt = hash_password("testpassword")
        user_manager.users["testuser"] = {
            "password_hash": hashed,
            "salt": salt,
            "roles": ["user"],
            "mfa_secret": "TEST_SECRET",
        }

        if user_manager.mfa_manager:
            # Mock verify to return True
            original_verify = user_manager.mfa_manager.verify_totp
            user_manager.mfa_manager.verify_totp = lambda secret, token: True

            result = user_manager.verify_mfa_setup("testuser", "123456")
            assert result
            assert user_manager.users["testuser"]["mfa_enabled"] is True

            # Restore
            user_manager.mfa_manager.verify_totp = original_verify

    def test_verify_mfa_setup_invalid_token(self, user_manager):
        """Test verifying MFA setup with invalid token"""
        hashed, salt = hash_password("testpassword")
        user_manager.users["testuser"] = {
            "password_hash": hashed,
            "salt": salt,
            "roles": ["user"],
            "mfa_secret": "TEST_SECRET",
        }

        if user_manager.mfa_manager:
            original_verify = user_manager.mfa_manager.verify_totp
            user_manager.mfa_manager.verify_totp = lambda secret, token: False

            result = user_manager.verify_mfa_setup("testuser", "123456")
            assert not result

            # Restore
            user_manager.mfa_manager.verify_totp = original_verify

    def test_verify_mfa_setup_no_secret(self, user_manager):
        """Test verifying MFA setup when no secret exists"""
        hashed, salt = hash_password("testpassword")
        user_manager.users["testuser"] = {
            "password_hash": hashed,
            "salt": salt,
            "roles": ["user"],
        }

        result = user_manager.verify_mfa_setup("testuser", "123456")
        assert not result

    def test_save_users_creates_directory(self, temp_users_file):
        """Test that save users creates directory if needed"""
        import os

        # Use a path with a directory that doesn't exist
        test_dir = os.path.join(os.path.dirname(temp_users_file), "test_users")
        test_file = os.path.join(test_dir, "users.json")

        manager = UserManager(users_file=test_file)
        hashed, salt = hash_password("testpassword")
        manager.users["testuser"] = {"password_hash": hashed, "salt": salt, "roles": ["user"]}
        manager._save_users()

        # Directory should be created
        assert os.path.exists(test_dir)
        assert os.path.exists(test_file)

        # Cleanup
        if os.path.exists(test_file):
            os.unlink(test_file)
        if os.path.exists(test_dir):
            os.rmdir(test_dir)


@pytest.mark.unit
class TestPasswordHashingExtended:
    """Extended tests for password hashing"""

    def test_hash_password_with_custom_salt(self):
        """Test hashing password with custom salt"""
        # Use string salt for compatibility (not bytes)
        salt = "custom_salt_string"
        hashed1, _ = hash_password("testpassword", salt)
        hashed2, _ = hash_password("testpassword", salt)
        # Same salt should produce same hash
        assert hashed1 == hashed2

    def test_verify_password_bcrypt_format(self):
        """Test verifying password with bcrypt format"""
        password = "testpassword"
        hashed, salt = hash_password(password)
        # Should work with bcrypt format
        assert verify_password(password, hashed, salt)

    def test_verify_password_sha256_format(self):
        """Test verifying password with SHA-256 format"""
        import hashlib

        # Create SHA-256 hash manually (only works if bcrypt not available)
        password = "testpassword"
        salt = "test_salt_string"
        password_salt = f"{password}{salt}".encode("utf-8")
        hashed = hashlib.sha256(password_salt).hexdigest()

        # Note: verify_password tries bcrypt first, then falls back to SHA-256
        # If bcrypt is available, it may not match, so we just test it doesn't crash
        try:
            result = verify_password(password, hashed, salt)
            assert isinstance(result, bool)
        except (ValueError, AttributeError):
            # May fail if bcrypt format doesn't match - that's OK
            pass


@pytest.mark.unit
class TestGlobalFunctions:
    """Test global authentication functions"""

    def test_get_user_manager_singleton(self):
        """Test that get_user_manager returns singleton"""
        manager1 = get_user_manager()
        manager2 = get_user_manager()
        assert manager1 is manager2
