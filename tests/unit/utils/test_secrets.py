"""
Tests for secrets management utilities
"""

import base64
import json
import os
import tempfile

import pytest

from bondtrader.utils.secrets import (
    SecretsManager,
    get_secrets_manager,
)


class TestSecretsManager:
    """Test SecretsManager functionality"""

    def test_secrets_manager_env_backend(self):
        """Test environment variable backend"""
        os.environ["TEST_SECRET"] = "test_value"

        try:
            manager = SecretsManager(backend="env")
            assert manager.get_secret("TEST_SECRET") == "test_value"
            assert manager.get_secret("NONEXISTENT", "default") == "default"
        finally:
            if "TEST_SECRET" in os.environ:
                del os.environ["TEST_SECRET"]

    def test_secrets_manager_file_backend(self):
        """Test encrypted file backend"""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_file = f.name

        try:
            # Set master password
            os.environ["SECRETS_MASTER_PASSWORD"] = "test_master_password"

            manager = SecretsManager(backend="file", secrets_file=temp_file)

            # Set a secret
            manager.set_secret("TEST_KEY", "test_value")

            # Get the secret
            assert manager.get_secret("TEST_KEY") == "test_value"

            # Get non-existent secret
            assert manager.get_secret("NONEXISTENT", "default") == "default"
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
            if "SECRETS_MASTER_PASSWORD" in os.environ:
                del os.environ["SECRETS_MASTER_PASSWORD"]

    def test_secrets_manager_file_persistence(self):
        """Test that file backend persists secrets"""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_file = f.name

        try:
            os.environ["SECRETS_MASTER_PASSWORD"] = "test_master_password"

            # Create manager and set secret
            manager1 = SecretsManager(backend="file", secrets_file=temp_file)
            manager1.set_secret("PERSISTENT_KEY", "persistent_value")

            # Create new manager instance (simulating restart)
            manager2 = SecretsManager(backend="file", secrets_file=temp_file)

            # Should still be able to read the secret
            assert manager2.get_secret("PERSISTENT_KEY") == "persistent_value"
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
            if "SECRETS_MASTER_PASSWORD" in os.environ:
                del os.environ["SECRETS_MASTER_PASSWORD"]

    def test_get_api_key(self):
        """Test getting API keys"""
        os.environ["FRED_API_KEY"] = "fred_key_value"
        os.environ["FINRA_API_KEY"] = "finra_key_value"

        try:
            manager = SecretsManager(backend="env")

            # Test service-specific key
            assert manager.get_api_key("fred") == "fred_key_value"
            assert manager.get_api_key("finra") == "finra_key_value"

            # Test non-existent key
            assert manager.get_api_key("nonexistent") is None
        finally:
            if "FRED_API_KEY" in os.environ:
                del os.environ["FRED_API_KEY"]
            if "FINRA_API_KEY" in os.environ:
                del os.environ["FINRA_API_KEY"]

    def test_require_secret(self):
        """Test requiring a secret (raises if not found)"""
        manager = SecretsManager(backend="env")

        # Should raise if secret not found
        with pytest.raises(ValueError, match="Required secret"):
            manager.require_secret("NONEXISTENT_SECRET")

        # Should return value if found
        os.environ["REQUIRED_SECRET"] = "required_value"
        try:
            assert manager.require_secret("REQUIRED_SECRET") == "required_value"
        finally:
            if "REQUIRED_SECRET" in os.environ:
                del os.environ["REQUIRED_SECRET"]

    def test_set_secret_file_only(self):
        """Test that set_secret only works for file backend"""
        manager = SecretsManager(backend="env")

        with pytest.raises(ValueError, match="set_secret only supported"):
            manager.set_secret("KEY", "VALUE")


class TestGetSecretsManager:
    """Test global secrets manager instance"""

    def test_get_secrets_manager_singleton(self):
        """Test that get_secrets_manager returns singleton"""
        # Clear environment to ensure clean state
        if "SECRETS_BACKEND" in os.environ:
            old_backend = os.environ["SECRETS_BACKEND"]
            del os.environ["SECRETS_BACKEND"]
        else:
            old_backend = None

        try:
            manager1 = get_secrets_manager()
            manager2 = get_secrets_manager()
            assert manager1 is manager2
        finally:
            if old_backend:
                os.environ["SECRETS_BACKEND"] = old_backend
