"""
API Key Management with Rotation and Expiration
Implements secure API key lifecycle management

CRITICAL: Required for Fortune 10 financial institution security standards
"""

import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from bondtrader.core.audit import AuditEventType, get_audit_logger
from bondtrader.utils.utils import logger


class APIKeyError(Exception):
    """API key related error"""

    pass


class APIKey:
    """API Key with metadata"""

    def __init__(
        self,
        key_id: str,
        key_hash: str,
        user_id: str,
        created_at: datetime,
        expires_at: Optional[datetime] = None,
        last_used: Optional[datetime] = None,
        usage_count: int = 0,
        revoked: bool = False,
        description: Optional[str] = None,
    ):
        self.key_id = key_id
        self.key_hash = key_hash  # SHA-256 hash of the actual key
        self.user_id = user_id
        self.created_at = created_at
        self.expires_at = expires_at
        self.last_used = last_used
        self.usage_count = usage_count
        self.revoked = revoked
        self.description = description

    def is_expired(self) -> bool:
        """Check if key is expired"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def is_valid(self) -> bool:
        """Check if key is valid (not revoked and not expired)"""
        return not self.revoked and not self.is_expired()

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        return {
            "key_id": self.key_id,
            "key_hash": self.key_hash,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "usage_count": self.usage_count,
            "revoked": self.revoked,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "APIKey":
        """Create from dictionary"""
        return cls(
            key_id=data["key_id"],
            key_hash=data["key_hash"],
            user_id=data["user_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=(
                datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None
            ),
            last_used=datetime.fromisoformat(data["last_used"]) if data.get("last_used") else None,
            usage_count=data.get("usage_count", 0),
            revoked=data.get("revoked", False),
            description=data.get("description"),
        )


class APIKeyManager:
    """
    API Key Manager with rotation and expiration support.

    CRITICAL: Manages API key lifecycle for secure access control.
    """

    def __init__(self, keys_file: Optional[str] = None):
        """
        Initialize API key manager.

        Args:
            keys_file: Path to JSON file storing API keys
        """
        self.keys_file = keys_file or "api_keys.json"
        self.keys: Dict[str, APIKey] = {}  # key_id -> APIKey
        self.key_lookup: Dict[str, str] = {}  # key_hash -> key_id (for fast lookup)
        self.audit_logger = get_audit_logger()
        self._load_keys()

    def _load_keys(self):
        """Load API keys from file"""
        import json
        import os

        if os.path.exists(self.keys_file):
            try:
                with open(self.keys_file, "r") as f:
                    data = json.load(f)
                    for key_data in data.get("keys", []):
                        key = APIKey.from_dict(key_data)
                        self.keys[key.key_id] = key
                        self.key_lookup[key.key_hash] = key.key_id
            except Exception as e:
                logger.error(f"Failed to load API keys: {e}")

    def _save_keys(self):
        """Save API keys to file"""
        import json
        import os

        try:
            os.makedirs(os.path.dirname(self.keys_file) or ".", exist_ok=True)
            data = {"keys": [key.to_dict() for key in self.keys.values()]}
            with open(self.keys_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save API keys: {e}")

    def generate_key(
        self,
        user_id: str,
        expires_in_days: Optional[int] = 90,
        description: Optional[str] = None,
    ) -> Tuple[str, APIKey]:
        """
        Generate a new API key.

        Args:
            user_id: User ID for the key
            expires_in_days: Days until expiration (None = never expires)
            description: Optional description

        Returns:
            Tuple of (plain_text_key, APIKey object)

        Note: The plain text key is only returned once and should be stored securely.
        """
        # Generate random key (32 bytes = 256 bits)
        plain_key = secrets.token_urlsafe(32)

        # Hash the key (we never store the plain text)
        key_hash = hashlib.sha256(plain_key.encode()).hexdigest()

        # Generate key ID
        key_id = secrets.token_urlsafe(16)

        # Calculate expiration
        expires_at = None
        if expires_in_days:
            expires_at = datetime.now() + timedelta(days=expires_in_days)

        # Create API key object
        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            user_id=user_id,
            created_at=datetime.now(),
            expires_at=expires_at,
            description=description,
        )

        # Store
        self.keys[key_id] = api_key
        self.key_lookup[key_hash] = key_id
        self._save_keys()

        # Audit log
        self.audit_logger.log(
            AuditEventType.USER_ACTION,
            user_id,
            "api_key_created",
            details={
                "key_id": key_id,
                "expires_at": expires_at.isoformat() if expires_at else None,
            },
            compliance_tags=["SOX"],
        )

        return plain_key, api_key

    def verify_key(self, api_key: str) -> Optional[APIKey]:
        """
        Verify an API key.

        Args:
            api_key: Plain text API key

        Returns:
            APIKey object if valid, None otherwise
        """
        # Hash the provided key
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        # Look up key
        key_id = self.key_lookup.get(key_hash)
        if not key_id:
            return None

        api_key_obj = self.keys.get(key_id)
        if not api_key_obj:
            return None

        # Check if valid
        if not api_key_obj.is_valid():
            return None

        # Update usage
        api_key_obj.last_used = datetime.now()
        api_key_obj.usage_count += 1
        self._save_keys()

        # Audit log
        self.audit_logger.log(
            AuditEventType.DATA_ACCESSED,
            api_key_obj.user_id,
            "api_key_used",
            details={"key_id": key_id},
            compliance_tags=["SOX"],
        )

        return api_key_obj

    def revoke_key(self, key_id: str, user_id: Optional[str] = None) -> bool:
        """
        Revoke an API key.

        Args:
            key_id: Key ID to revoke
            user_id: Optional user ID (for authorization check)

        Returns:
            True if key was revoked
        """
        if key_id not in self.keys:
            return False

        key = self.keys[key_id]

        # Check authorization
        if user_id and key.user_id != user_id:
            return False

        # Revoke
        key.revoked = True
        self._save_keys()

        # Audit log
        self.audit_logger.log(
            AuditEventType.USER_ACTION,
            key.user_id,
            "api_key_revoked",
            details={"key_id": key_id},
            compliance_tags=["SOX"],
        )

        return True

    def rotate_key(self, key_id: str, user_id: Optional[str] = None) -> Tuple[str, APIKey]:
        """
        Rotate an API key (revoke old, create new).

        Args:
            key_id: Key ID to rotate
            user_id: Optional user ID (for authorization check)

        Returns:
            Tuple of (new_plain_text_key, new_APIKey object)
        """
        if key_id not in self.keys:
            raise APIKeyError(f"Key {key_id} not found")

        old_key = self.keys[key_id]

        # Check authorization
        if user_id and old_key.user_id != user_id:
            raise APIKeyError("Unauthorized")

        # Revoke old key
        old_key.revoked = True

        # Create new key with same expiration policy
        expires_in_days = None
        if old_key.expires_at:
            days_remaining = (old_key.expires_at - datetime.now()).days
            expires_in_days = max(30, days_remaining)  # At least 30 days

        new_plain_key, new_key = self.generate_key(
            user_id=old_key.user_id,
            expires_in_days=expires_in_days,
            description=f"Rotated from {key_id}",
        )

        # Audit log
        self.audit_logger.log(
            AuditEventType.USER_ACTION,
            old_key.user_id,
            "api_key_rotated",
            details={"old_key_id": key_id, "new_key_id": new_key.key_id},
            compliance_tags=["SOX"],
        )

        self._save_keys()

        return new_plain_key, new_key

    def list_keys(self, user_id: Optional[str] = None) -> List[APIKey]:
        """
        List API keys.

        Args:
            user_id: Optional user ID to filter by

        Returns:
            List of API keys (without plain text)
        """
        keys = list(self.keys.values())

        if user_id:
            keys = [k for k in keys if k.user_id == user_id]

        return keys

    def get_expiring_keys(self, days_ahead: int = 7) -> List[APIKey]:
        """
        Get keys expiring within specified days.

        Args:
            days_ahead: Number of days to look ahead

        Returns:
            List of expiring keys
        """
        cutoff = datetime.now() + timedelta(days=days_ahead)
        return [
            key
            for key in self.keys.values()
            if key.expires_at and key.expires_at <= cutoff and not key.revoked
        ]


# Global API key manager instance
_api_key_manager: Optional[APIKeyManager] = None


def get_api_key_manager() -> APIKeyManager:
    """Get or create global API key manager instance"""
    global _api_key_manager
    if _api_key_manager is None:
        _api_key_manager = APIKeyManager()
    return _api_key_manager
