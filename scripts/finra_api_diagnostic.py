"""
FINRA API Detailed Diagnostic
Checks credential format and provides troubleshooting steps
"""

import base64
import os
import sys
from pathlib import Path

# Load .env file
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()

finra_key = os.getenv("FINRA_API_KEY", "")
finra_password = os.getenv("FINRA_API_PASSWORD", "")

print("=" * 70)
print("FINRA API CREDENTIAL DIAGNOSTIC")
print("=" * 70)

print(f"\n1. API Key Format:")
print(f"   Length: {len(finra_key)} characters")
print(f"   First 10 chars: {finra_key[:10] if finra_key else 'N/A'}...")
print(f"   Contains spaces: {'Yes' if ' ' in finra_key else 'No'}")
print(f"   Contains special chars: {'Yes' if any(c in finra_key for c in ['@', '#', '$', '%']) else 'No'}")

print(f"\n2. API Password Format:")
print(f"   Length: {len(finra_password)} characters")
print(f"   First 5 chars: {finra_password[:5] if finra_password else 'N/A'}...")
print(f"   Contains spaces: {'Yes' if ' ' in finra_password else 'No'}")
print(f"   Contains hyphens: {'Yes' if '-' in finra_password else 'No'}")

print(f"\n3. Base64 Encoding Test:")
if finra_key and finra_password:
    credentials = f"{finra_key}:{finra_password}"
    b64 = base64.b64encode(credentials.encode()).decode()
    print(f"   Credentials string length: {len(credentials)}")
    print(f"   Base64 length: {len(b64)}")
    print(f"   Base64 first 20 chars: {b64[:20]}...")

    # Decode to verify
    try:
        decoded = base64.b64decode(b64).decode()
        if decoded == credentials:
            print(f"   ✓ Base64 encoding/decoding works correctly")
        else:
            print(f"   ✗ Base64 encoding mismatch!")
    except (ValueError, UnicodeDecodeError) as e:
        import binascii

        print(f"   ✗ Base64 decoding failed: {e}")

print(f"\n4. Common Issues:")
print(f"   - API keys might need to be activated in FINRA portal")
print(f"   - Credentials might be for a different environment (prod vs mock)")
print(f"   - API key format might be different (some APIs use different formats)")
print(f"   - Account might need additional entitlements/permissions")

print(f"\n5. Next Steps:")
print(f"   1. Verify credentials in FINRA Developer Portal")
print(f"   2. Check if credentials are for production or mock environment")
print(f"   3. Ensure account has TRACE data access permissions")
print(f"   4. Try using FINRA API Console to test credentials manually")
print(f"   5. Contact FINRA support if credentials are confirmed correct")

print(f"\n6. Alternative Approach:")
print(f"   Since FINRA TRACE data requires special licensing and may have")
print(f"   different authentication, consider:")
print(f"   - Using FRED for Treasury data (already working)")
print(f"   - Using synthetic corporate bond data for training")
print(f"   - Contacting FINRA for API documentation specific to your account")

print("\n" + "=" * 70)
