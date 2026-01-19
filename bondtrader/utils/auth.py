"""
Authentication and Authorization Utilities
Provides authentication for dashboard and API endpoints

CRITICAL: Uses bcrypt for secure password hashing (industry standard for financial systems)
"""

import os
import time
from functools import wraps
from typing import Dict, List, Optional, Tuple

import streamlit as st

from bondtrader.utils.utils import logger

# Try to import bcrypt, fall back to SHA-256 if not available (for backward compatibility)
try:
    import bcrypt

    HAS_BCRYPT = True
except ImportError:
    HAS_BCRYPT = False
    import hashlib
    import hmac


class AuthenticationError(Exception):
    """Authentication error"""

    pass


class AuthorizationError(Exception):
    """Authorization error"""

    pass


def hash_password(password: str, salt: Optional[bytes] = None) -> Tuple[str, str]:
    """
    Hash a password using bcrypt (preferred) or SHA-256 (fallback).

    CRITICAL: bcrypt is the industry standard for financial systems.
    Uses 12 rounds (2^12 = 4096 iterations) for security.

    Args:
        password: Plain text password
        salt: Optional salt (generated if not provided)

    Returns:
        Tuple of (hashed_password, salt) - both as strings for storage
    """
    if HAS_BCRYPT:
        # Use bcrypt (industry standard, much more secure than SHA-256)
        if salt is None:
            # Generate salt with bcrypt (includes salt in hash)
            hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12))
            return hashed.decode("utf-8"), ""  # bcrypt includes salt in hash
        else:
            # Use provided salt (for migration scenarios)
            # Handle both bytes and string salt
            if isinstance(salt, bytes):
                salt_bytes = salt
            elif isinstance(salt, str):
                # If it's a string, try to use it as bcrypt salt (must be valid bcrypt salt format)
                # If not valid format, treat as custom salt and hash with SHA-256 instead
                if salt.startswith("$2"):
                    salt_bytes = salt.encode("utf-8")
                    hashed = bcrypt.hashpw(password.encode("utf-8"), salt_bytes)
                    return hashed.decode("utf-8"), ""
                else:
                    # Invalid bcrypt salt format, fall through to SHA-256
                    pass
            else:
                salt_bytes = str(salt).encode("utf-8")

            # If we get here, salt wasn't valid bcrypt format, use SHA-256 fallback
            if not isinstance(salt, str):
                salt = salt_bytes.decode("utf-8", errors="ignore") if isinstance(salt_bytes, bytes) else str(salt_bytes)
            password_salt = f"{password}{salt}".encode("utf-8")
            hashed = hashlib.sha256(password_salt).hexdigest()
            return hashed, salt if isinstance(salt, str) else salt_bytes.decode("utf-8", errors="ignore")
    else:
        # Fallback to SHA-256 (less secure, but backward compatible)
        if salt is None:
            salt = os.urandom(32).hex()

        # Combine password and salt
        password_salt = f"{password}{salt}".encode("utf-8")

        # Hash using SHA-256
        hashed = hashlib.sha256(password_salt).hexdigest()

        return hashed, salt


def verify_password(password: str, hashed_password: str, salt: str) -> bool:
    """
    Verify a password against a hash.

    Supports both bcrypt (preferred) and SHA-256 (legacy) formats.

    Args:
        password: Plain text password to verify
        hashed_password: Stored hash
        salt: Salt used in original hash (empty string for bcrypt)

    Returns:
        True if password matches
    """
    if HAS_BCRYPT:
        # Check if hash looks like bcrypt (starts with $2b$ or $2a$)
        if hashed_password.startswith("$2"):
            try:
                # bcrypt includes salt in hash, so we don't need separate salt
                # Convert hashed_password to bytes if it's a string
                hash_bytes = hashed_password.encode("utf-8") if isinstance(hashed_password, str) else hashed_password
                return bcrypt.checkpw(password.encode("utf-8"), hash_bytes)
            except Exception:
                return False
        else:
            # Legacy SHA-256 hash - verify using old method
            computed_hash, _ = hash_password(password, salt)
            return hmac.compare_digest(computed_hash, hashed_password)
    else:
        # Fallback to SHA-256 verification
        computed_hash, _ = hash_password(password, salt)
        return hmac.compare_digest(computed_hash, hashed_password)


class UserManager:
    """
    User management for authentication with MFA support.

    CRITICAL: Supports MFA (Multi-Factor Authentication) for financial system security.
    """

    def __init__(self, users_file: Optional[str] = None):
        """
        Initialize user manager

        Args:
            users_file: Path to JSON file storing users (defaults to environment or in-memory)
        """
        self.users_file = users_file or os.getenv("USERS_FILE", None)
        self.users: Dict[str, Dict] = {}
        self._load_users()

        # Import MFA manager
        try:
            from bondtrader.utils.mfa import get_mfa_manager

            self.mfa_manager = get_mfa_manager()
        except ImportError:
            self.mfa_manager = None

    def _load_users(self):
        """Load users from file or environment"""
        import json

        # Try to load from file
        if self.users_file and os.path.exists(self.users_file):
            try:
                with open(self.users_file, "r") as f:
                    self.users = json.load(f)
                return
            except Exception:
                pass

        # Load from environment variables (for simple deployments)
        # Format: USERS=user1:password1,user2:password2
        users_env = os.getenv("USERS", "")
        if users_env:
            for user_pass in users_env.split(","):
                if ":" in user_pass:
                    username, password = user_pass.split(":", 1)
                    hashed, salt = hash_password(password)
                    self.users[username] = {
                        "password_hash": hashed,
                        "salt": salt,
                        "roles": ["user"],
                    }

        # Default admin user if no users configured (for development only)
        # SECURITY: Only enable in development, never in production
        if not self.users and os.getenv("ENABLE_DEFAULT_ADMIN", "false").lower() == "true":
            default_password = os.getenv("DEFAULT_ADMIN_PASSWORD")
            if not default_password:
                raise ValueError(
                    "DEFAULT_ADMIN_PASSWORD environment variable must be set when "
                    "ENABLE_DEFAULT_ADMIN=true. Never use default passwords in production."
                )
            hashed, salt = hash_password(default_password)
            self.users["admin"] = {
                "password_hash": hashed,
                "salt": salt,
                "roles": ["admin", "user"],
            }

    def authenticate(self, username: str, password: str, mfa_token: Optional[str] = None) -> bool:
        """
        Authenticate a user with optional MFA.

        CRITICAL: If user has MFA enabled, mfa_token is required.

        Args:
            username: Username
            password: Plain text password
            mfa_token: Optional MFA token (required if MFA enabled for user)

        Returns:
            True if authentication successful
        """
        if username not in self.users:
            return False

        user = self.users[username]

        # Verify password
        if not verify_password(password, user["password_hash"], user["salt"]):
            return False

        # Check if MFA is enabled for this user
        mfa_enabled = user.get("mfa_enabled", False)
        mfa_secret = user.get("mfa_secret")

        if mfa_enabled and mfa_secret:
            # MFA is required
            if not mfa_token:
                logger.warning(f"MFA token required for user {username}")
                return False

            # Verify MFA token
            if not self.mfa_manager or not self.mfa_manager.verify_totp(mfa_secret, mfa_token):
                logger.warning(f"Invalid MFA token for user {username}")
                return False

        return True

    def enable_mfa(self, username: str) -> Tuple[str, bytes]:
        """
        Enable MFA for a user and return secret and QR code.

        Args:
            username: Username

        Returns:
            Tuple of (secret, qr_code_image_bytes)
        """
        if username not in self.users:
            raise AuthenticationError(f"User {username} not found")

        if not self.mfa_manager:
            raise AuthenticationError("MFA manager not available. Install pyotp and qrcode.")

        # Generate secret
        secret = self.mfa_manager.generate_secret(username)

        # Generate QR code
        _, qr_code = self.mfa_manager.generate_qr_code(username, secret)

        # Store secret (but don't enable MFA until verified)
        self.users[username]["mfa_secret"] = secret
        self._save_users()

        return secret, qr_code

    def verify_mfa_setup(self, username: str, token: str) -> bool:
        """
        Verify MFA setup by checking a token, then enable MFA.

        Args:
            username: Username
            token: TOTP token to verify

        Returns:
            True if token is valid and MFA is now enabled
        """
        if username not in self.users:
            return False

        user = self.users[username]
        secret = user.get("mfa_secret")

        if not secret:
            return False

        if not self.mfa_manager:
            return False

        # Verify token
        if self.mfa_manager.verify_totp(secret, token):
            # Enable MFA and generate backup codes
            self.users[username]["mfa_enabled"] = True
            backup_codes = self.mfa_manager.generate_backup_codes()
            self.users[username]["mfa_backup_codes"] = backup_codes
            self._save_users()
            return True

        return False

    def _save_users(self):
        """Save users to file"""
        if not self.users_file:
            return

        import json

        try:
            os.makedirs(os.path.dirname(self.users_file) or ".", exist_ok=True)
            with open(self.users_file, "w") as f:
                json.dump(self.users, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save users: {e}")

    def has_role(self, username: str, role: str) -> bool:
        """Check if user has a specific role"""
        if username not in self.users:
            return False
        return role in self.users[username].get("roles", [])

    def get_user_roles(self, username: str) -> List[str]:
        """Get all roles for a user"""
        if username not in self.users:
            return []
        return self.users[username].get("roles", [])


# Global user manager instance
_user_manager: Optional[UserManager] = None


def get_user_manager() -> UserManager:
    """Get or create global user manager instance"""
    global _user_manager
    if _user_manager is None:
        _user_manager = UserManager()
    return _user_manager


def require_auth(func):
    """
    Decorator to require authentication for Streamlit functions

    Usage:
        @require_auth
        def my_page():
            st.write("Protected content")
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check if user is authenticated
        if "authenticated" not in st.session_state or not st.session_state.authenticated:
            # Show login form
            st.title("🔐 Authentication Required")
            st.info("Please log in to access the dashboard")

            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")

            if st.button("Login"):
                user_manager = get_user_manager()
                if user_manager.authenticate(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.user_roles = user_manager.get_user_roles(username)
                    st.rerun()
                else:
                    st.error("Invalid username or password")

            return

        # User is authenticated, proceed
        return func(*args, **kwargs)

    return wrapper


def require_role(role: str):
    """
    Decorator to require a specific role

    Usage:
        @require_role("admin")
        def admin_page():
            st.write("Admin only content")
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # First check authentication
            if "authenticated" not in st.session_state or not st.session_state.authenticated:
                st.error("Authentication required")
                return

            # Check role
            user_roles = st.session_state.get("user_roles", [])
            if role not in user_roles:
                st.error(f"Access denied. Required role: {role}")
                return

            return func(*args, **kwargs)

        return wrapper

    return decorator


def logout():
    """Logout current user"""
    if "authenticated" in st.session_state:
        del st.session_state.authenticated
    if "username" in st.session_state:
        del st.session_state.username
    if "user_roles" in st.session_state:
        del st.session_state.user_roles
