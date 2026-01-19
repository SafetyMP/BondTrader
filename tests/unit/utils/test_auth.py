"""
Tests for authentication utilities
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from bondtrader.utils.auth import (
    AuthenticationError,
    AuthorizationError,
    UserManager,
    get_user_manager,
    hash_password,
    verify_password,
)


class TestPasswordHashing:
    """Test password hashing functionality"""

    def test_hash_password_generates_salt(self):
        """Test that hash_password generates a salt if not provided"""
        hashed1, salt1 = hash_password("test_password")
        hashed2, salt2 = hash_password("test_password")

        # Different salts should produce different hashes
        assert salt1 != salt2
        assert hashed1 != hashed2

    def test_hash_password_with_salt(self):
        """Test that hash_password uses provided salt"""
        salt = "test_salt"
        hashed1, salt1 = hash_password("test_password", salt)
        hashed2, salt2 = hash_password("test_password", salt)

        # Same salt should produce same hash
        assert salt1 == salt2 == salt
        assert hashed1 == hashed2

    def test_verify_password_correct(self):
        """Test password verification with correct password"""
        password = "test_password"
        hashed, salt = hash_password(password)

        assert verify_password(password, hashed, salt) is True

    def test_verify_password_incorrect(self):
        """Test password verification with incorrect password"""
        password = "test_password"
        hashed, salt = hash_password(password)

        assert verify_password("wrong_password", hashed, salt) is False

    def test_verify_password_timing_attack_protection(self):
        """Test that verify_password uses constant-time comparison"""
        password = "test_password"
        hashed, salt = hash_password(password)

        # Should not raise exception even with wrong password
        result = verify_password("wrong_password", hashed, salt)
        assert result is False


class TestUserManager:
    """Test UserManager functionality"""

    def test_user_manager_init(self):
        """Test UserManager initialization"""
        manager = UserManager()
        assert manager is not None

    def test_user_manager_from_file(self):
        """Test loading users from file"""
        # Create temporary users file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            hashed, salt = hash_password("test_password")
            users_data = {"test_user": {"password_hash": hashed, "salt": salt, "roles": ["user"]}}
            json.dump(users_data, f)
            temp_file = f.name

        try:
            manager = UserManager(users_file=temp_file)
            assert manager.authenticate("test_user", "test_password") is True
            assert manager.authenticate("test_user", "wrong_password") is False
        finally:
            os.unlink(temp_file)

    def test_user_manager_from_env(self):
        """Test loading users from environment variable"""
        # Set environment variable
        os.environ["USERS"] = "test_user:test_password,admin:admin_pass"

        try:
            manager = UserManager()
            assert manager.authenticate("test_user", "test_password") is True
            assert manager.authenticate("admin", "admin_pass") is True
            assert manager.authenticate("test_user", "wrong_password") is False
        finally:
            if "USERS" in os.environ:
                del os.environ["USERS"]

    def test_user_manager_default_admin(self):
        """Test default admin user creation"""
        os.environ["ENABLE_DEFAULT_ADMIN"] = "true"
        os.environ["DEFAULT_ADMIN_PASSWORD"] = "test_admin_pass"

        try:
            # Clear any existing users
            if "USERS" in os.environ:
                del os.environ["USERS"]

            manager = UserManager()
            assert manager.authenticate("admin", "test_admin_pass") is True
        finally:
            if "ENABLE_DEFAULT_ADMIN" in os.environ:
                del os.environ["ENABLE_DEFAULT_ADMIN"]
            if "DEFAULT_ADMIN_PASSWORD" in os.environ:
                del os.environ["DEFAULT_ADMIN_PASSWORD"]

    def test_has_role(self):
        """Test role checking"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            hashed, salt = hash_password("test_password")
            users_data = {
                "admin_user": {"password_hash": hashed, "salt": salt, "roles": ["admin", "user"]},
                "regular_user": {"password_hash": hashed, "salt": salt, "roles": ["user"]},
            }
            json.dump(users_data, f)
            temp_file = f.name

        try:
            manager = UserManager(users_file=temp_file)
            assert manager.has_role("admin_user", "admin") is True
            assert manager.has_role("admin_user", "user") is True
            assert manager.has_role("regular_user", "admin") is False
            assert manager.has_role("regular_user", "user") is True
            assert manager.has_role("nonexistent", "user") is False
        finally:
            os.unlink(temp_file)

    def test_get_user_roles(self):
        """Test getting user roles"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            hashed, salt = hash_password("test_password")
            users_data = {"test_user": {"password_hash": hashed, "salt": salt, "roles": ["admin", "user", "analyst"]}}
            json.dump(users_data, f)
            temp_file = f.name

        try:
            manager = UserManager(users_file=temp_file)
            roles = manager.get_user_roles("test_user")
            assert "admin" in roles
            assert "user" in roles
            assert "analyst" in roles
            assert len(roles) == 3

            # Non-existent user
            assert manager.get_user_roles("nonexistent") == []
        finally:
            os.unlink(temp_file)


class TestGetUserManager:
    """Test global user manager instance"""

    def test_get_user_manager_singleton(self):
        """Test that get_user_manager returns singleton"""
        manager1 = get_user_manager()
        manager2 = get_user_manager()
        assert manager1 is manager2
