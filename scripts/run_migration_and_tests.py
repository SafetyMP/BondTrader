#!/usr/bin/env python3
"""
Run PostgreSQL Migration and Tests
Executes migration, load tests, and chaos tests

Usage:
    python3 scripts/run_migration_and_tests.py [--skip-migration] [--skip-load] [--skip-chaos]
"""

import os
import subprocess
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_migration():
    """Run PostgreSQL migration"""
    print("=" * 60)
    print("Running PostgreSQL Migration")
    print("=" * 60)

    try:
        # Check if PostgreSQL is configured
        from bondtrader.data.postgresql_support import get_database_type
        from scripts.migrate_to_postgresql import migrate_schema_to_postgresql

        db_type = get_database_type()

        if db_type != "postgresql":
            print("⚠️  DATABASE_TYPE is not set to 'postgresql'")
            print("Skipping migration. To run migration:")
            print("  export DATABASE_TYPE=postgresql")
            print("  export POSTGRES_HOST=localhost")
            print("  export POSTGRES_PORT=5432")
            print("  export POSTGRES_DB=bondtrader")
            print("  export POSTGRES_USER=bondtrader")
            print("  export POSTGRES_PASSWORD=your_password")
            return False

        success = migrate_schema_to_postgresql()
        if success:
            print("✅ Migration successful!")
        return success
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return False


def run_load_tests():
    """Run load tests"""
    print("\n" + "=" * 60)
    print("Running Load Tests")
    print("=" * 60)

    try:
        result = subprocess.run(
            [
                "python3",
                "-m",
                "pytest",
                "tests/load/test_load.py",
                "-v",
                "-m",
                "slow",
                "--tb=short",
            ],
            cwd=project_root,
            capture_output=True,
            text=True,
        )

        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        if result.returncode == 0:
            print("✅ Load tests passed!")
        else:
            print(f"⚠️  Load tests exited with code {result.returncode}")

        return result.returncode == 0
    except Exception as e:
        print(f"❌ Load tests failed: {e}")
        return False


def run_chaos_tests():
    """Run chaos tests"""
    print("\n" + "=" * 60)
    print("Running Chaos Tests")
    print("=" * 60)

    try:
        result = subprocess.run(
            ["python3", "-m", "pytest", "tests/chaos/test_chaos.py", "-v", "--tb=short"],
            cwd=project_root,
            capture_output=True,
            text=True,
        )

        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        if result.returncode == 0:
            print("✅ Chaos tests passed!")
        else:
            print(f"⚠️  Chaos tests exited with code {result.returncode}")

        return result.returncode == 0
    except Exception as e:
        print(f"❌ Chaos tests failed: {e}")
        return False


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Run migration and tests")
    parser.add_argument("--skip-migration", action="store_true", help="Skip PostgreSQL migration")
    parser.add_argument("--skip-load", action="store_true", help="Skip load tests")
    parser.add_argument("--skip-chaos", action="store_true", help="Skip chaos tests")

    args = parser.parse_args()

    results = {}

    # Run migration
    if not args.skip_migration:
        results["migration"] = run_migration()
    else:
        print("⏭️  Skipping migration")
        results["migration"] = None

    # Run load tests
    if not args.skip_load:
        results["load_tests"] = run_load_tests()
    else:
        print("⏭️  Skipping load tests")
        results["load_tests"] = None

    # Run chaos tests
    if not args.skip_chaos:
        results["chaos_tests"] = run_chaos_tests()
    else:
        print("⏭️  Skipping chaos tests")
        results["chaos_tests"] = None

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    for test_name, result in results.items():
        if result is None:
            status = "⏭️  Skipped"
        elif result:
            status = "✅ Passed"
        else:
            status = "❌ Failed"
        print(f"{test_name:20s}: {status}")

    # Exit with error if any test failed
    failed = [name for name, result in results.items() if result is False]
    if failed:
        print(f"\n⚠️  {len(failed)} test(s) failed: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("\n✅ All tests completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
