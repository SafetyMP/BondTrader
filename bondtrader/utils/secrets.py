"""
Secrets Management Utilities
Provides secure secrets management for API keys and credentials
"""

import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class SecretsManager:
    """
    Secrets management with encryption support

    Supports multiple backends:
    - Environment variables (default)
    - Encrypted file storage
    - AWS Secrets Manager (if boto3 available)
    - HashiCorp Vault (if hvac available)
    """

    def __init__(self, backend: str = "env", secrets_file: Optional[str] = None):
        """
        Initialize secrets manager

        Args:
            backend: Backend to use ('env', 'file', 'aws', 'vault')
            secrets_file: Path to encrypted secrets file (for 'file' backend)
        """
        self.backend = backend
        self.secrets_file = secrets_file or os.getenv("SECRETS_FILE", ".secrets.encrypted")
        self._encryption_key: Optional[bytes] = None
        self._cipher: Optional[Fernet] = None

        # Initialize backend
        if backend == "file":
            self._init_file_backend()
        elif backend == "aws":
            self._init_aws_backend()
        elif backend == "vault":
            self._init_vault_backend()

    def _init_file_backend(self):
        """Initialize file-based encrypted storage"""
        # Get encryption key from environment
        key_env = os.getenv("SECRETS_ENCRYPTION_KEY")
        if key_env:
            # Use provided key
            key = base64.urlsafe_b64decode(key_env.encode())
        else:
            # Generate key from master password
            master_password = os.getenv("SECRETS_MASTER_PASSWORD")
            if not master_password:
                raise ValueError(
                    "SECRETS_MASTER_PASSWORD environment variable must be set. " "Never use default passwords in production."
                )
            salt = os.getenv("SECRETS_SALT")
            if not salt:
                raise ValueError(
                    "SECRETS_SALT environment variable must be set for secure encryption. " "Generate a random salt value."
                )
            salt = salt.encode()

            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(master_password.encode()))

        self._cipher = Fernet(key)

    def _init_aws_backend(self):
        """Initialize AWS Secrets Manager backend"""
        try:
            import boto3

            self._aws_client = boto3.client("secretsmanager")
            self._aws_secret_name = os.getenv("AWS_SECRET_NAME", "bondtrader/secrets")
        except ImportError:
            raise ImportError("boto3 required for AWS backend. Install with: pip install boto3")

    def _init_vault_backend(self):
        """Initialize HashiCorp Vault backend"""
        try:
            import hvac

            vault_url = os.getenv("VAULT_ADDR", "http://localhost:8200")
            vault_token = os.getenv("VAULT_TOKEN")

            if not vault_token:
                raise ValueError("VAULT_TOKEN environment variable required for Vault backend")

            self._vault_client = hvac.Client(url=vault_url, token=vault_token)
            self._vault_path = os.getenv("VAULT_SECRET_PATH", "secret/bondtrader")
        except ImportError:
            raise ImportError("hvac required for Vault backend. Install with: pip install hvac")

    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get a secret value

        Args:
            key: Secret key
            default: Default value if not found

        Returns:
            Secret value or default
        """
        if self.backend == "env":
            return os.getenv(key, default)

        elif self.backend == "file":
            return self._get_file_secret(key, default)

        elif self.backend == "aws":
            return self._get_aws_secret(key, default)

        elif self.backend == "vault":
            return self._get_vault_secret(key, default)

        return default

    def _get_file_secret(self, key: str, default: Optional[str]) -> Optional[str]:
        """Get secret from encrypted file"""
        if not os.path.exists(self.secrets_file):
            return default

        try:
            with open(self.secrets_file, "rb") as f:
                encrypted_data = f.read()

            if not self._cipher:
                raise ValueError("Encryption not initialized")

            decrypted_data = self._cipher.decrypt(encrypted_data)
            secrets = json.loads(decrypted_data.decode())
            return secrets.get(key, default)

        except Exception:
            # If decryption fails, fall back to environment
            return os.getenv(key, default)

    def _get_aws_secret(self, key: str, default: Optional[str]) -> Optional[str]:
        """Get secret from AWS Secrets Manager"""
        try:
            response = self._aws_client.get_secret_value(SecretId=self._aws_secret_name)
            secrets = json.loads(response["SecretString"])
            return secrets.get(key, default)
        except Exception:
            # Fall back to environment
            return os.getenv(key, default)

    def _get_vault_secret(self, key: str, default: Optional[str]) -> Optional[str]:
        """Get secret from HashiCorp Vault"""
        try:
            secret_response = self._vault_client.secrets.kv.v2.read_secret_version(path=self._vault_path)
            secrets = secret_response["data"]["data"]
            return secrets.get(key, default)
        except Exception:
            # Fall back to environment
            return os.getenv(key, default)

    def set_secret(self, key: str, value: str):
        """
        Set a secret value (file backend only)

        Args:
            key: Secret key
            value: Secret value
        """
        if self.backend != "file":
            raise ValueError("set_secret only supported for 'file' backend")

        # Load existing secrets
        secrets: Dict[str, str] = {}
        if os.path.exists(self.secrets_file):
            try:
                with open(self.secrets_file, "rb") as f:
                    encrypted_data = f.read()
                if self._cipher:
                    decrypted_data = self._cipher.decrypt(encrypted_data)
                    secrets = json.loads(decrypted_data.decode())
            except Exception:
                pass

        # Update secret
        secrets[key] = value

        # Save encrypted
        if not self._cipher:
            raise ValueError("Encryption not initialized")

        encrypted_data = self._cipher.encrypt(json.dumps(secrets).encode())

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.secrets_file) or ".", exist_ok=True)

        with open(self.secrets_file, "wb") as f:
            f.write(encrypted_data)

    def get_api_key(self, service: str) -> Optional[str]:
        """
        Get API key for a service

        Args:
            service: Service name (e.g., 'fred', 'finra')

        Returns:
            API key or None
        """
        # Try service-specific key first
        key = self.get_secret(f"{service.upper()}_API_KEY")
        if key:
            return key

        # Try generic API key
        return self.get_secret("API_KEY")

    def require_secret(self, key: str) -> str:
        """
        Get a required secret (raises if not found)

        Args:
            key: Secret key

        Returns:
            Secret value

        Raises:
            ValueError: If secret not found
        """
        value = self.get_secret(key)
        if value is None:
            raise ValueError(f"Required secret '{key}' not found")
        return value


# Global secrets manager instance
_secrets_manager: Optional[SecretsManager] = None


def get_secrets_manager() -> SecretsManager:
    """Get or create global secrets manager instance"""
    global _secrets_manager
    if _secrets_manager is None:
        backend = os.getenv("SECRETS_BACKEND", "env")
        _secrets_manager = SecretsManager(backend=backend)
    return _secrets_manager


def get_api_key(service: str) -> Optional[str]:
    """
    Convenience function to get API key

    Args:
        service: Service name

    Returns:
        API key or None
    """
    return get_secrets_manager().get_api_key(service)
