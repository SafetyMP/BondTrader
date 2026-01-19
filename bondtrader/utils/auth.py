"""
Authentication and Authorization Utilities
Provides authentication for dashboard and API endpoints
"""

import hashlib
import hmac
import os
import time
from functools import wraps
from typing import Dict, List, Optional

import streamlit as st


class AuthenticationError(Exception):
    """Authentication error"""

    pass


class AuthorizationError(Exception):
    """Authorization error"""

    pass


from typing import Tuple


def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """
    Hash a password using SHA-256 with salt

    Args:
        password: Plain text password
        salt: Optional salt (generated if not provided)

    Returns:
        Tuple of (hashed_password, salt)
    """
    if salt is None:
        salt = os.urandom(32).hex()

    # Combine password and salt
    password_salt = f"{password}{salt}".encode("utf-8")

    # Hash using SHA-256
    hashed = hashlib.sha256(password_salt).hexdigest()

    return hashed, salt


def verify_password(password: str, hashed_password: str, salt: str) -> bool:
    """
    Verify a password against a hash

    Args:
        password: Plain text password to verify
        hashed_password: Stored hash
        salt: Salt used in original hash

    Returns:
        True if password matches
    """
    computed_hash, _ = hash_password(password, salt)
    return hmac.compare_digest(computed_hash, hashed_password)


class UserManager:
    """Simple user management for authentication"""

    def __init__(self, users_file: Optional[str] = None):
        """
        Initialize user manager

        Args:
            users_file: Path to JSON file storing users (defaults to environment or in-memory)
        """
        self.users_file = users_file or os.getenv("USERS_FILE", None)
        self.users: Dict[str, Dict] = {}
        self._load_users()

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
                    self.users[username] = {"password_hash": hashed, "salt": salt, "roles": ["user"]}

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
            self.users["admin"] = {"password_hash": hashed, "salt": salt, "roles": ["admin", "user"]}

    def authenticate(self, username: str, password: str) -> bool:
        """
        Authenticate a user

        Args:
            username: Username
            password: Plain text password

        Returns:
            True if authentication successful
        """
        if username not in self.users:
            return False

        user = self.users[username]
        return verify_password(password, user["password_hash"], user["salt"])

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
