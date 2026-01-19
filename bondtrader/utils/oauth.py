"""
OAuth 2.0 / OpenID Connect Support
Enterprise SSO integration for authentication

CRITICAL: Required for enterprise deployments in Fortune 10 financial institutions
"""

import os
from typing import Any, Dict, Optional

import requests

from bondtrader.utils.utils import logger

# OAuth libraries availability
try:
    from authlib.integrations.requests_client import OAuth2Session
    from authlib.jose import jwt

    _authlib_available = True
except ImportError:
    _authlib_available = False
    logger.warning("Authlib not available. Install with: pip install authlib requests")


class OAuth2Provider:
    """OAuth 2.0 provider configuration"""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        authorization_endpoint: str,
        token_endpoint: str,
        userinfo_endpoint: Optional[str] = None,
        issuer: Optional[str] = None,
    ):
        """
        Initialize OAuth 2.0 provider.

        Args:
            client_id: OAuth client ID
            client_secret: OAuth client secret
            authorization_endpoint: Authorization endpoint URL
            token_endpoint: Token endpoint URL
            userinfo_endpoint: Optional userinfo endpoint URL
            issuer: Optional issuer URL for OpenID Connect
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.authorization_endpoint = authorization_endpoint
        self.token_endpoint = token_endpoint
        self.userinfo_endpoint = userinfo_endpoint
        self.issuer = issuer

    @classmethod
    def from_environment(cls, provider_name: str = "OAUTH") -> Optional["OAuth2Provider"]:
        """
        Create provider from environment variables.

        Args:
            provider_name: Prefix for environment variables (e.g., "OAUTH" for OAUTH_CLIENT_ID)

        Returns:
            OAuth2Provider instance or None if not configured
        """
        client_id = os.getenv(f"{provider_name}_CLIENT_ID")
        client_secret = os.getenv(f"{provider_name}_CLIENT_SECRET")
        auth_endpoint = os.getenv(f"{provider_name}_AUTHORIZATION_ENDPOINT")
        token_endpoint = os.getenv(f"{provider_name}_TOKEN_ENDPOINT")

        if not all([client_id, client_secret, auth_endpoint, token_endpoint]):
            return None

        return cls(
            client_id=client_id,
            client_secret=client_secret,
            authorization_endpoint=auth_endpoint,
            token_endpoint=token_endpoint,
            userinfo_endpoint=os.getenv(f"{provider_name}_USERINFO_ENDPOINT"),
            issuer=os.getenv(f"{provider_name}_ISSUER"),
        )


class OAuth2Manager:
    """
    OAuth 2.0 / OpenID Connect Manager

    Handles OAuth 2.0 authentication and token validation.
    """

    def __init__(self, provider: Optional[OAuth2Provider] = None):
        """
        Initialize OAuth 2.0 manager.

        Args:
            provider: OAuth 2.0 provider configuration
        """
        self.provider = provider or OAuth2Provider.from_environment()

    def get_authorization_url(
        self, redirect_uri: str, state: Optional[str] = None, scopes: Optional[list] = None
    ) -> Optional[str]:
        """
        Get OAuth 2.0 authorization URL.

        Args:
            redirect_uri: Redirect URI after authorization
            state: Optional state parameter for CSRF protection
            scopes: Optional list of scopes to request

        Returns:
            Authorization URL or None if provider not configured
        """
        if not self.provider or not _authlib_available:
            return None

        if scopes is None:
            scopes = ["openid", "profile", "email"]

        try:
            session = OAuth2Session(
                self.provider.client_id,
                redirect_uri=redirect_uri,
                scope=scopes,
            )
            url, state = session.authorization_url(
                self.provider.authorization_endpoint,
                state=state,
            )
            return url
        except Exception as e:
            logger.error(f"Failed to generate authorization URL: {e}")
            return None

    def exchange_code_for_token(self, code: str, redirect_uri: str) -> Optional[Dict[str, Any]]:
        """
        Exchange authorization code for access token.

        Args:
            code: Authorization code from callback
            redirect_uri: Redirect URI used in authorization

        Returns:
            Token response dictionary or None if failed
        """
        if not self.provider or not _authlib_available:
            return None

        try:
            session = OAuth2Session(
                self.provider.client_id,
                self.provider.client_secret,
                redirect_uri=redirect_uri,
            )
            token = session.fetch_token(
                self.provider.token_endpoint,
                code=code,
            )
            return token
        except Exception as e:
            logger.error(f"Failed to exchange code for token: {e}")
            return None

    def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Validate and decode JWT token (OpenID Connect).

        Args:
            token: JWT token to validate

        Returns:
            Decoded token claims or None if invalid
        """
        if not self.provider or not _authlib_available:
            return None

        try:
            # Get JWKS (JSON Web Key Set) from issuer
            if self.provider.issuer:
                jwks_url = f"{self.provider.issuer}/.well-known/jwks.json"
                jwks = requests.get(jwks_url, timeout=5).json()
            else:
                # Fallback: try to decode without verification (not recommended for production)
                logger.warning("No issuer configured, skipping token verification")
                return jwt.decode(token, options={"verify_signature": False})

            # Decode and verify token
            claims = jwt.decode(token, jwks)
            return claims
        except Exception as e:
            logger.error(f"Token validation failed: {e}")
            return None

    def get_user_info(self, access_token: str) -> Optional[Dict[str, Any]]:
        """
        Get user information from userinfo endpoint.

        Args:
            access_token: OAuth 2.0 access token

        Returns:
            User information dictionary or None if failed
        """
        if not self.provider or not self.provider.userinfo_endpoint:
            return None

        try:
            response = requests.get(
                self.provider.userinfo_endpoint,
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=5,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get user info: {e}")
            return None

    def refresh_token(self, refresh_token: str) -> Optional[Dict[str, Any]]:
        """
        Refresh access token using refresh token.

        Args:
            refresh_token: OAuth 2.0 refresh token

        Returns:
            New token response dictionary or None if failed
        """
        if not self.provider or not _authlib_available:
            return None

        try:
            session = OAuth2Session(
                self.provider.client_id,
                self.provider.client_secret,
            )
            token = session.refresh_token(
                self.provider.token_endpoint,
                refresh_token=refresh_token,
            )
            return token
        except Exception as e:
            logger.error(f"Failed to refresh token: {e}")
            return None


# Global OAuth manager instance
_oauth_manager: Optional[OAuth2Manager] = None


def get_oauth_manager() -> Optional[OAuth2Manager]:
    """Get or create global OAuth manager instance"""
    global _oauth_manager
    if _oauth_manager is None:
        _oauth_manager = OAuth2Manager()
    return _oauth_manager
