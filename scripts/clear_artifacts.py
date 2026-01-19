#!/usr/bin/env python3
"""
Clear All Binary Artifacts and Generated Data
Removes trained models, datasets, evaluation results, and other generated files
"""

import os
import shutil
import sys
from pathlib import Path
from typing import List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bondtrader.config import get_config


def find_artifact_files() -> List[Path]:
    """Find all artifact files to remove"""
    artifacts = []

    # Binary file extensions
    binary_extensions = {".joblib", ".pkl", ".h5", ".hdf5", ".pb", ".onnx"}

    # Database files
    db_files = ["bonds.db", "test.db", "*.db"]

    # Directories to scan
    root = Path(__file__).parent.parent

    # Find all binary files
    for ext in binary_extensions:
        artifacts.extend(root.rglob(f"*{ext}"))

    # Find database files
    for db_pattern in db_files:
        artifacts.extend(root.glob(db_pattern))
        artifacts.extend(root.rglob(db_pattern))

    # Find coverage files
    artifacts.extend(root.rglob("coverage.xml"))
    artifacts.extend(root.rglob(".coverage"))
    artifacts.extend(root.rglob(".coverage.*"))

    # Filter out files in .git, .venv, etc.
    filtered = []
    for artifact in artifacts:
        parts = artifact.parts
        if ".git" not in parts and ".venv" not in parts and "venv" not in parts:
            if "__pycache__" not in parts:
                filtered.append(artifact)

    return filtered


def find_artifact_directories() -> List[Path]:
    """Find all artifact directories to remove"""
    config = get_config()
    root = Path(__file__).parent.parent

    directories = [
        root / config.model_dir,
        root / config.data_dir,
        root / config.evaluation_data_dir,
        root / config.evaluation_results_dir,
        root / config.checkpoint_dir,
        root / "mlruns",  # MLflow tracking
        root / "htmlcov",  # Coverage reports
        root / ".pytest_cache",
        root / ".mypy_cache",
        root / ".hypothesis",
    ]

    # Filter to only existing directories
    return [d for d in directories if d.exists() and d.is_dir()]


def clear_artifacts(dry_run: bool = False) -> dict:
    """
    Clear all binary artifacts and generated data

    Args:
        dry_run: If True, only report what would be deleted without deleting

    Returns:
        Dictionary with statistics about what was cleared
    """
    stats = {"files_removed": 0, "directories_removed": 0, "total_size_mb": 0.0, "errors": []}

    print("=" * 70)
    print("CLEARING BINARY ARTIFACTS AND GENERATED DATA")
    print("=" * 70)

    if dry_run:
        print("\n🔍 DRY RUN MODE - No files will be deleted\n")

    # Find and remove files
    print("\n📁 Finding artifact files...")
    artifact_files = find_artifact_files()

    if artifact_files:
        print(f"   Found {len(artifact_files)} artifact files")

        for file_path in artifact_files:
            try:
                if file_path.exists():
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    stats["total_size_mb"] += size_mb

                    if dry_run:
                        print(f"   [DRY RUN] Would remove: {file_path} ({size_mb:.2f} MB)")
                    else:
                        file_path.unlink()
                        stats["files_removed"] += 1
                        if stats["files_removed"] % 10 == 0:
                            print(f"   Removed {stats['files_removed']} files...")
            except Exception as e:
                error_msg = f"Error removing {file_path}: {e}"
                stats["errors"].append(error_msg)
                print(f"   ⚠️  {error_msg}")
    else:
        print("   No artifact files found")

    # Find and remove directories
    print("\n📂 Finding artifact directories...")
    artifact_dirs = find_artifact_directories()

    if artifact_dirs:
        print(f"   Found {len(artifact_dirs)} artifact directories")

        for dir_path in artifact_dirs:
            try:
                if dir_path.exists() and dir_path.is_dir():
                    # Calculate size
                    total_size = sum(f.stat().st_size for f in dir_path.rglob("*") if f.is_file())
                    size_mb = total_size / (1024 * 1024)
                    stats["total_size_mb"] += size_mb

                    if dry_run:
                        print(f"   [DRY RUN] Would remove: {dir_path} ({size_mb:.2f} MB)")
                    else:
                        shutil.rmtree(dir_path)
                        stats["directories_removed"] += 1
                        print(f"   ✓ Removed: {dir_path}")
            except Exception as e:
                error_msg = f"Error removing {dir_path}: {e}"
                stats["errors"].append(error_msg)
                print(f"   ⚠️  {error_msg}")
    else:
        print("   No artifact directories found")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if dry_run:
        print(f"\n📊 Would remove:")
    else:
        print(f"\n📊 Removed:")

    print(f"   Files: {stats['files_removed']}")
    print(f"   Directories: {stats['directories_removed']}")
    print(f"   Total size: {stats['total_size_mb']:.2f} MB")

    if stats["errors"]:
        print(f"\n⚠️  Errors encountered: {len(stats['errors'])}")
        for error in stats["errors"][:5]:  # Show first 5 errors
            print(f"   - {error}")
        if len(stats["errors"]) > 5:
            print(f"   ... and {len(stats['errors']) - 5} more errors")

    if not dry_run:
        print("\n✅ Artifacts cleared successfully!")
        print("\n💡 To regenerate models, run:")
        print("   python scripts/refresh_models.py")
        print("   or")
        print("   make refresh-models")
    else:
        print("\n💡 Run without --dry-run to actually remove files")

    return stats


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Clear all binary artifacts and generated data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (see what would be deleted)
  python scripts/clear_artifacts.py --dry-run

  # Actually clear artifacts
  python scripts/clear_artifacts.py

  # Clear and show verbose output
  python scripts/clear_artifacts.py --verbose
        """,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting",
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Show verbose output")

    args = parser.parse_args()

    try:
        stats = clear_artifacts(dry_run=args.dry_run)

        if stats["errors"]:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
