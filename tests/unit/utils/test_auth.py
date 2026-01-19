"""
Unit tests for authentication utilities
"""

import os
import tempfile
from pathlib import Path

import pytest

from bondtrader.utils.auth import UserManager, hash_password, verify_password


@pytest.mark.unit
class TestPasswordHashing:
    """Test password hashing functions"""

    def test_hash_password_generates_hash(self):
        """Test that password hashing generates a hash"""
        hashed, salt = hash_password("test_password")
        assert hashed is not None
        assert len(hashed) > 0

    def test_verify_password_correct(self):
        """Test verifying correct password"""
        password = "test_password"
        hashed, salt = hash_password(password)
        assert verify_password(password, hashed, salt)

    def test_verify_password_incorrect(self):
        """Test verifying incorrect password"""
        hashed, salt = hash_password("test_password")
        assert not verify_password("wrong_password", hashed, salt)

    def test_hash_password_different_salts(self):
        """Test that same password hashed twice produces different hashes"""
        hashed1, salt1 = hash_password("test_password")
        hashed2, salt2 = hash_password("test_password")
        # Hashes should be different due to salt
        assert hashed1 != hashed2 or salt1 != salt2

    def test_verify_password_with_salt(self):
        """Test verifying password with provided salt"""
        password = "test_password"
        salt = b"test_salt"
        hashed, returned_salt = hash_password(password, salt)
        # Use the returned salt (which may be converted to string format)
        assert verify_password(password, hashed, returned_salt)


@pytest.mark.unit
class TestUserManager:
    """Test UserManager functionality"""

    @pytest.fixture
    def temp_users_file(self):
        """Create temporary users file"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_path = f.name
        yield temp_path
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    @pytest.fixture
    def user_manager_empty(self):
        """Create empty user manager"""
        return UserManager(users_file=None)

    @pytest.fixture
    def user_manager_file(self, temp_users_file):
        """Create user manager with file"""
        return UserManager(users_file=temp_users_file)

    def test_user_manager_initialization(self, user_manager_empty):
        """Test user manager initialization"""
        assert user_manager_empty.users is not None
        assert isinstance(user_manager_empty.users, dict)

    def test_authenticate_nonexistent_user(self, user_manager_empty):
        """Test authentication for non-existent user"""
        assert not user_manager_empty.authenticate("nonexistent", "password")

    def test_has_role_nonexistent(self, user_manager_empty):
        """Test checking role for non-existent user"""
        assert not user_manager_empty.has_role("nonexistent", "admin")

    def test_get_user_roles_nonexistent(self, user_manager_empty):
        """Test getting roles for non-existent user"""
        roles = user_manager_empty.get_user_roles("nonexistent")
        assert roles == []

    def test_enable_mfa(self, user_manager_file):
        """Test enabling MFA for user"""
        # Create user first by adding to users dict
        hashed, salt = hash_password("testpassword")
        user_manager_file.users["testuser"] = {
            "password_hash": hashed,
            "salt": salt,
            "roles": ["user"],
        }

        if user_manager_file.mfa_manager:
            secret, qr_code = user_manager_file.enable_mfa("testuser")
            assert secret is not None
            assert qr_code is not None
            assert "mfa_secret" in user_manager_file.users["testuser"]

    def test_enable_mfa_nonexistent_user(self, user_manager_file):
        """Test enabling MFA for non-existent user"""
        with pytest.raises(Exception):  # Should raise AuthenticationError
            user_manager_file.enable_mfa("nonexistent")

    def test_get_user_manager_singleton(self):
        """Test get_user_manager returns singleton"""
        from bondtrader.utils.auth import get_user_manager

        manager1 = get_user_manager()
        manager2 = get_user_manager()
        # Should be same instance
        assert manager1 is manager2

    def test_user_manager_load_from_file(self, temp_users_file):
        """Test loading users from file"""
        import json

        users_data = {
            "user1": {"password_hash": "hash1", "salt": "salt1", "roles": ["user"]},
            "user2": {"password_hash": "hash2", "salt": "salt2", "roles": ["admin"]},
        }
        with open(temp_users_file, "w") as f:
            json.dump(users_data, f)

        manager = UserManager(users_file=temp_users_file)
        assert "user1" in manager.users
        assert "user2" in manager.users

    def test_user_manager_save_to_file(self, temp_users_file):
        """Test saving users to file"""
        manager = UserManager(users_file=temp_users_file)
        # Add user directly to users dict (UserManager doesn't have create_user method)
        hashed, salt = hash_password("testpassword")
        manager.users["testuser"] = {"password_hash": hashed, "salt": salt, "roles": ["user"]}
        manager._save_users()

        # Verify file was created and has content
        assert os.path.exists(temp_users_file)
        import json

        with open(temp_users_file, "r") as f:
            data = json.load(f)
            assert "testuser" in data
