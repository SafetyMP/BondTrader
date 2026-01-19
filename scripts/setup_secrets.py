#!/usr/bin/env python3
"""
Setup script for secrets management
Helps configure and manage encrypted secrets
"""

import os
import sys
from pathlib import Path

from bondtrader.utils.secrets import SecretsManager


def setup_file_backend():
    """Set up encrypted file backend for secrets"""
    print("Setting up encrypted file backend for secrets...")

    # Get configuration
    secrets_file = input("Secrets file path (default: .secrets.encrypted): ").strip() or ".secrets.encrypted"
    master_password = input("Master password (for encryption): ").strip()

    if not master_password:
        print("❌ Master password is required")
        return False

    # Confirm password
    confirm_password = input("Confirm master password: ").strip()
    if master_password != confirm_password:
        print("❌ Passwords do not match")
        return False

    # Set environment variables
    print("\n📝 Add these to your .env file:")
    print(f"   SECRETS_BACKEND=file")
    print(f"   SECRETS_FILE={secrets_file}")
    print(f"   SECRETS_MASTER_PASSWORD={master_password}")

    # Test the setup
    try:
        os.environ["SECRETS_BACKEND"] = "file"
        os.environ["SECRETS_FILE"] = secrets_file
        os.environ["SECRETS_MASTER_PASSWORD"] = master_password

        manager = SecretsManager(backend="file", secrets_file=secrets_file)

        # Test with a sample secret
        test_key = input("\nEnter a test key name (or Enter to skip): ").strip()
        if test_key:
            test_value = input(f"Enter value for {test_key}: ").strip()
            if test_value:
                manager.set_secret(test_key, test_value)
                retrieved = manager.get_secret(test_key)
                if retrieved == test_value:
                    print(f"✅ Test successful! Secret '{test_key}' stored and retrieved correctly.")
                else:
                    print(f"❌ Test failed! Retrieved value doesn't match.")
                    return False

        print("\n✅ Encrypted file backend configured successfully!")
        return True
    except Exception as e:
        print(f"❌ Error setting up file backend: {e}")
        return False


def add_secret_interactive():
    """Interactively add a secret"""
    backend = os.getenv("SECRETS_BACKEND", "env")

    if backend == "env":
        print("Using environment variable backend.")
        print("To add secrets, set environment variables directly:")
        print("  export FRED_API_KEY=your_key")
        return

    try:
        secrets_file = os.getenv("SECRETS_FILE", ".secrets.encrypted")
        manager = SecretsManager(backend=backend, secrets_file=secrets_file)

        key = input("Secret key name: ").strip()
        if not key:
            print("❌ Key name is required")
            return

        value = input(f"Value for {key}: ").strip()
        if not value:
            print("❌ Value is required")
            return

        if backend == "file":
            manager.set_secret(key, value)
            print(f"✅ Secret '{key}' stored successfully!")
        else:
            print(f"❌ set_secret only supported for 'file' backend")
    except Exception as e:
        print(f"❌ Error adding secret: {e}")


def list_secrets():
    """List available secrets (keys only, not values)"""
    backend = os.getenv("SECRETS_BACKEND", "env")

    if backend == "env":
        print("Environment variable backend - listing common API key variables:")
        common_keys = ["FRED_API_KEY", "FINRA_API_KEY", "API_KEY"]
        for key in common_keys:
            value = os.getenv(key)
            if value:
                print(f"  ✅ {key}: {'*' * min(len(value), 20)}")
            else:
                print(f"  ⚪ {key}: (not set)")
    else:
        print("⚠️  Secret listing not available for this backend")
        print("   Use get_secret() in code to retrieve values")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Setup and manage secrets for BondTrader")
    parser.add_argument("--setup-file", action="store_true", help="Set up encrypted file backend")
    parser.add_argument("--add", action="store_true", help="Add a secret interactively")
    parser.add_argument("--list", action="store_true", help="List available secrets")

    args = parser.parse_args()

    if args.setup_file:
        setup_file_backend()
    elif args.add:
        add_secret_interactive()
    elif args.list:
        list_secrets()
    else:
        print("Secrets Management Setup")
        print("=" * 50)
        print("\nOptions:")
        print("  --setup-file    Set up encrypted file backend")
        print("  --add           Add a secret interactively")
        print("  --list          List available secrets")
        print("\nFor environment variable backend, set variables directly:")
        print("  export FRED_API_KEY=your_key")


if __name__ == "__main__":
    main()
