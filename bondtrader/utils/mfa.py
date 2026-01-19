"""
Multi-Factor Authentication (MFA) Support
Implements TOTP (Time-based One-Time Password) for financial system security

CRITICAL: Required for Fortune 10 financial institution security standards
"""

import base64
import io
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

# Try to import MFA libraries
try:
    import pyotp
    import qrcode

    HAS_MFA_LIBS = True
except ImportError:
    HAS_MFA_LIBS = False

from bondtrader.core.audit import AuditEventType, get_audit_logger
from bondtrader.utils.utils import logger


class MFAError(Exception):
    """MFA-related error"""

    pass


class MFAManager:
    """
    Multi-Factor Authentication Manager

    Supports TOTP (Time-based One-Time Password) using industry-standard algorithms.
    """

    def __init__(self):
        """Initialize MFA manager"""
        if not HAS_MFA_LIBS:
            logger.warning(
                "MFA libraries (pyotp, qrcode) not available. "
                "Install with: pip install pyotp qrcode[pil]"
            )

    def generate_secret(self, username: str, issuer: str = "BondTrader") -> str:
        """
        Generate a TOTP secret for a user.

        Args:
            username: Username for the secret
            issuer: Issuer name (appears in authenticator apps)

        Returns:
            Base32-encoded secret key
        """
        if not HAS_MFA_LIBS:
            raise MFAError("MFA libraries not available. Install pyotp and qrcode.")

        # Generate a random secret (160 bits = 32 base32 characters)
        secret = pyotp.random_base32()
        return secret

    def generate_qr_code(
        self, username: str, secret: str, issuer: str = "BondTrader"
    ) -> Tuple[str, bytes]:
        """
        Generate QR code for MFA setup.

        Args:
            username: Username
            secret: TOTP secret
            issuer: Issuer name

        Returns:
            Tuple of (provisioning_uri, qr_code_image_bytes)
        """
        if not HAS_MFA_LIBS:
            raise MFAError("MFA libraries not available. Install pyotp and qrcode.")

        # Create TOTP object
        totp = pyotp.TOTP(secret)

        # Generate provisioning URI (for authenticator apps)
        provisioning_uri = totp.provisioning_uri(name=username, issuer_name=issuer)

        # Generate QR code
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(provisioning_uri)
        qr.make(fit=True)

        # Create image
        img = qr.make_image(fill_color="black", back_color="white")

        # Convert to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        return provisioning_uri, img_bytes.getvalue()

    def verify_totp(self, secret: str, token: str, window: int = 1) -> bool:
        """
        Verify a TOTP token.

        Args:
            secret: TOTP secret
            token: 6-digit token from authenticator app
            window: Time window for verification (default: 1 = current 30-second window)

        Returns:
            True if token is valid
        """
        if not HAS_MFA_LIBS:
            raise MFAError("MFA libraries not available. Install pyotp and qrcode.")

        try:
            totp = pyotp.TOTP(secret)
            # Verify with time window (allows for clock skew)
            return totp.verify(token, valid_window=window)
        except Exception as e:
            logger.error(f"Error verifying TOTP token: {e}")
            return False

    def generate_backup_codes(self, count: int = 10) -> list[str]:
        """
        Generate backup codes for account recovery.

        Args:
            count: Number of backup codes to generate

        Returns:
            List of backup codes (8-digit codes)
        """
        import secrets

        codes = []
        for _ in range(count):
            # Generate 8-digit code
            code = f"{secrets.randbelow(100000000):08d}"
            codes.append(code)

        return codes

    def verify_backup_code(self, code: str, backup_codes: list[str]) -> Tuple[bool, list[str]]:
        """
        Verify and consume a backup code.

        Args:
            code: Backup code to verify
            backup_codes: List of valid backup codes

        Returns:
            Tuple of (is_valid, remaining_codes)
        """
        if code in backup_codes:
            # Remove used code
            remaining = [c for c in backup_codes if c != code]
            return True, remaining
        return False, backup_codes


# Global MFA manager instance
_mfa_manager: Optional[MFAManager] = None


def get_mfa_manager() -> MFAManager:
    """Get or create global MFA manager instance"""
    global _mfa_manager
    if _mfa_manager is None:
        _mfa_manager = MFAManager()
    return _mfa_manager
