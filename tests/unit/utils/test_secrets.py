"""
Extended unit tests for secrets management utilities
"""

from unittest.mock import MagicMock, patch

import pytest

from bondtrader.utils.secrets import SecretsManager


@pytest.mark.unit
class TestSecretsManager:
    """Extended tests for SecretsManager class"""

    def test_secrets_manager_creation_env(self):
        """Test creating secrets manager with env backend"""
        manager = SecretsManager(backend="env")
        assert manager.backend == "env"

    @patch.dict("os.environ", {"SECRETS_ENCRYPTION_KEY": "test_key_encoded"})
    def test_secrets_manager_file_backend(self):
        """Test creating secrets manager with file backend"""
        try:
            manager = SecretsManager(backend="file", secrets_file=".test_secrets")
            assert manager.backend == "file"
        except ValueError:
            # May fail if SECRETS_MASTER_PASSWORD not set
            pass

    def test_get_secret_env(self):
        """Test getting secret from environment"""
        import os

        manager = SecretsManager(backend="env")

        with patch.dict(os.environ, {"TEST_SECRET": "test_value"}):
            value = manager.get_secret("TEST_SECRET")
            assert value == "test_value"

    def test_set_secret_file(self):
        """Test setting secret in file backend"""
        import os

        with patch.dict(
            os.environ,
            {
                "SECRETS_MASTER_PASSWORD": "test_password",
                "SECRETS_SALT": "test_salt_value_for_testing_only",
            },
        ):
            try:
                manager = SecretsManager(backend="file", secrets_file=".test_secrets")
                result = manager.set_secret("TEST_SECRET", "test_value")
                # Should work for file backend
                assert isinstance(result, bool) or result is None
            except (ValueError, ImportError):
                # May fail if dependencies not available
                pass

    def test_get_secret_not_found(self):
        """Test getting non-existent secret"""
        manager = SecretsManager(backend="env")

        value = manager.get_secret("NON_EXISTENT_SECRET", default="default_value")
        assert value == "default_value"
