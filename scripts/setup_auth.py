#!/usr/bin/env python3
"""
Setup script for authentication
Creates users file from environment variables or interactive input
"""

import json
import os
import sys
from pathlib import Path

from bondtrader.utils.auth import UserManager, hash_password


def create_users_file(users_file: str = "config/users.json"):
    """Create users file from environment or interactive input"""
    users_file_path = Path(users_file)
    users_file_path.parent.mkdir(parents=True, exist_ok=True)

    users = {}

    # Check if users already exist
    if users_file_path.exists():
        print(f"Users file already exists: {users_file_path}")
        response = input("Overwrite? (y/N): ").strip().lower()
        if response != "y":
            print("Cancelled.")
            return

    # Try to load from environment
    users_env = os.getenv("USERS", "")
    if users_env:
        print("Loading users from USERS environment variable...")
        for user_pass in users_env.split(","):
            if ":" in user_pass:
                username, password = user_pass.split(":", 1)
                hashed, salt = hash_password(password)
                users[username] = {"password_hash": hashed, "salt": salt, "roles": ["user"]}
                print(f"  Added user: {username}")

    # Interactive mode
    print("\nInteractive user creation (press Enter to finish):")
    while True:
        username = input("\nUsername (or Enter to finish): ").strip()
        if not username:
            break

        if username in users:
            print(f"User {username} already exists. Skipping.")
            continue

        password = input(f"Password for {username}: ").strip()
        if not password:
            print("Password cannot be empty. Skipping.")
            continue

        roles_input = input(f"Roles for {username} (comma-separated, default: user): ").strip()
        roles = [r.strip() for r in roles_input.split(",")] if roles_input else ["user"]

        hashed, salt = hash_password(password)
        users[username] = {"password_hash": hashed, "salt": salt, "roles": roles}
        print(f"  Added user: {username} with roles: {', '.join(roles)}")

    if not users:
        print("No users created. Exiting.")
        return

    # Save to file
    with open(users_file_path, "w") as f:
        json.dump(users, f, indent=2)

    print(f"\n✅ Users file created: {users_file_path}")
    print(f"   Total users: {len(users)}")
    print(f"\nTo use this file, set:")
    print(f"   export USERS_FILE={users_file_path.absolute()}")


def verify_users_file(users_file: str = "config/users.json"):
    """Verify users file and test authentication"""
    users_file_path = Path(users_file)

    if not users_file_path.exists():
        print(f"❌ Users file not found: {users_file_path}")
        return False

    try:
        manager = UserManager(users_file=str(users_file_path))

        print(f"✅ Users file is valid: {users_file_path}")
        print(f"   Users found: {len(manager.users)}")

        # Test authentication
        print("\nTesting authentication:")
        for username in manager.users:
            test_password = input(f"  Password for {username} (or Enter to skip): ").strip()
            if test_password:
                if manager.authenticate(username, test_password):
                    print(f"    ✅ {username}: Authentication successful")
                else:
                    print(f"    ❌ {username}: Authentication failed")

        return True
    except Exception as e:
        print(f"❌ Error loading users file: {e}")
        return False


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Setup authentication for BondTrader")
    parser.add_argument("--create", action="store_true", help="Create users file")
    parser.add_argument("--verify", action="store_true", help="Verify users file")
    parser.add_argument("--file", default="config/users.json", help="Path to users file (default: config/users.json)")

    args = parser.parse_args()

    if args.create:
        create_users_file(args.file)
    elif args.verify:
        verify_users_file(args.file)
    else:
        # Default: create if doesn't exist, verify if exists
        users_file_path = Path(args.file)
        if users_file_path.exists():
            verify_users_file(args.file)
        else:
            create_users_file(args.file)


if __name__ == "__main__":
    main()
