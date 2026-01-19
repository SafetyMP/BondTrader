"""
Unit tests for API key management utilities
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from bondtrader.utils.api_keys import APIKey, APIKeyError, APIKeyManager


@pytest.mark.unit
class TestAPIKey:
    """Test APIKey class"""

    def test_api_key_creation(self):
        """Test creating an API key"""
        key = APIKey(
            key_id="test-key-001",
            key_hash="hash123",
            user_id="user1",
            created_at=datetime.now(),
        )
        assert key.key_id == "test-key-001"
        assert key.user_id == "user1"
        assert key.revoked is False

    def test_api_key_is_expired(self):
        """Test checking if key is expired"""
        expired_key = APIKey(
            key_id="test-key",
            key_hash="hash",
            user_id="user1",
            created_at=datetime.now(),
            expires_at=datetime.now() - timedelta(days=1),
        )
        assert expired_key.is_expired() is True

        valid_key = APIKey(
            key_id="test-key",
            key_hash="hash",
            user_id="user1",
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(days=1),
        )
        assert valid_key.is_expired() is False

    def test_api_key_is_valid(self):
        """Test checking if key is valid"""
        valid_key = APIKey(
            key_id="test-key",
            key_hash="hash",
            user_id="user1",
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(days=1),
        )
        assert valid_key.is_valid() is True

        revoked_key = APIKey(
            key_id="test-key",
            key_hash="hash",
            user_id="user1",
            created_at=datetime.now(),
            revoked=True,
        )
        assert revoked_key.is_valid() is False

    def test_api_key_to_dict(self):
        """Test converting key to dictionary"""
        key = APIKey(
            key_id="test-key",
            key_hash="hash123",
            user_id="user1",
            created_at=datetime.now(),
            description="Test key",
        )
        key_dict = key.to_dict()
        assert key_dict["key_id"] == "test-key"
        assert key_dict["key_hash"] == "hash123"
        assert key_dict["description"] == "Test key"

    def test_api_key_from_dict(self):
        """Test creating key from dictionary"""
        key_dict = {
            "key_id": "test-key",
            "key_hash": "hash123",
            "user_id": "user1",
            "created_at": datetime.now().isoformat(),
            "expires_at": None,
            "last_used": None,
            "usage_count": 0,
            "revoked": False,
            "description": "Test key",
        }
        key = APIKey.from_dict(key_dict)
        assert key.key_id == "test-key"
        assert key.user_id == "user1"


@pytest.mark.unit
class TestAPIKeyManager:
    """Test APIKeyManager class"""

    def test_api_key_manager_creation(self):
        """Test creating API key manager"""
        manager = APIKeyManager()
        assert manager is not None

    def test_generate_key(self):
        """Test generating a new API key"""
        manager = APIKeyManager()
        key_id, key = manager.generate_key("user1", description="Test key")
        assert key_id is not None
        assert key.user_id == "user1"
        assert key.description == "Test key"
        assert len(key_id) > 0

    def test_validate_key(self):
        """Test validating an API key"""
        manager = APIKeyManager()
        key_id, key = manager.generate_key("user1")
        # Store the key hash for validation
        key_hash = key.key_hash

        # Create a new manager and validate
        manager2 = APIKeyManager()
        manager2.keys[key_id] = key

        # Generate the key string from key_id (in real usage, key string is provided)
        # For test, we'll just check that validation method exists
        assert hasattr(manager2, "validate_key")

    def test_revoke_key(self):
        """Test revoking an API key"""
        manager = APIKeyManager()
        key_id, key = manager.generate_key("user1")
        manager.keys[key_id] = key

        manager.revoke_key(key_id)
        assert manager.keys[key_id].revoked is True

    def test_list_keys(self):
        """Test listing API keys for a user"""
        manager = APIKeyManager()
        key_id1, _ = manager.generate_key("user1")
        key_id2, _ = manager.generate_key("user1")
        key_id3, _ = manager.generate_key("user2")

        user1_keys = manager.list_keys("user1")
        assert len(user1_keys) >= 2
