#!/usr/bin/env python3
"""
Hard wipe MLflow database - deletes the entire SQLite database file.

This script completely removes the MLflow database file, forcing it to be
re-initialized on next use. This is useful when soft deletes don't work
or you want a completely fresh start.

WARNING: This will delete ALL experiments, runs, and metadata!
"""

import sys
import os
import shutil
from pathlib import Path

# Default MLflow database path (mounted from Podman container)
# Container mounts /var/lib/mlflow -> /mlflow/backend
# Database is at sqlite:///mlflow/backend/mlflow.db in container
# Which corresponds to /var/lib/mlflow/mlflow.db on host
DEFAULT_DB_PATH = "/var/lib/mlflow/mlflow.db"
DEFAULT_ARTIFACTS_PATH = "/var/lib/mlflow/artifacts"

def hard_wipe_mlflow_db(
    db_path: str = DEFAULT_DB_PATH,
    artifacts_path: str | None = None,
    backup: bool = False,
    force: bool = False,
) -> bool:
    """
    Hard wipe MLflow database by deleting the database file.
    
    Args:
        db_path: Path to MLflow SQLite database file
        artifacts_path: Optional path to artifacts directory (will be deleted if provided)
        backup: If True, create a backup before deleting
    
    Returns:
        True if successful, False otherwise
    """
    db_file = Path(db_path)
    artifacts_dir = Path(artifacts_path) if artifacts_path else None
    
    # Check if database exists
    if not db_file.exists():
        print(f"Database file not found at: {db_path}")
        print("Database may already be wiped or path is incorrect.")
        return False
    
    # Get file size for info
    db_size_mb = db_file.stat().st_size / (1024 * 1024)
    print(f"Found MLflow database: {db_path}")
    print(f"  Size: {db_size_mb:.2f} MB")
    
    # Check for artifacts
    artifacts_size_mb = 0.0
    if artifacts_dir and artifacts_dir.exists():
        artifacts_size_mb = sum(
            f.stat().st_size for f in artifacts_dir.rglob("*") if f.is_file()
        ) / (1024 * 1024)
        print(f"Found artifacts directory: {artifacts_path}")
        print(f"  Size: {artifacts_size_mb:.2f} MB")
    
    # Confirmation
    total_size = db_size_mb + artifacts_size_mb
    print(f"\n⚠️  WARNING: This will delete:")
    print(f"   - Database file: {db_path} ({db_size_mb:.2f} MB)")
    if artifacts_dir and artifacts_dir.exists():
        print(f"   - Artifacts directory: {artifacts_path} ({artifacts_size_mb:.2f} MB)")
    print(f"   - Total: {total_size:.2f} MB")
    print(f"\n⚠️  ALL experiments, runs, and metadata will be PERMANENTLY DELETED!")
    
    if not force:
        response = input("\nType 'yes' to confirm deletion: ")
        if response.lower() != 'yes':
            print("Aborted")
            return False
    else:
        print("\n⚠️  --force flag: skipping confirmation")
    
    # Create backup if requested
    if backup:
        backup_path = Path(f"{db_path}.backup")
        print(f"\nCreating backup: {backup_path}")
        try:
            shutil.copy2(db_file, backup_path)
            print(f"✓ Backup created: {backup_path}")
        except Exception as e:
            print(f"⚠️  Warning: Could not create backup: {e}")
            if not force:
                response = input("Continue without backup? (yes/no): ")
                if response.lower() != 'yes':
                    print("Aborted")
                    return False
    
    # IMPORTANT: Stop MLflow server FIRST to prevent it from recreating the database
    print("\n" + "=" * 70)
    print("Step 1: Stopping MLflow server...")
    print("=" * 70)
    try:
        import subprocess
        print("   Stopping MLflow server (requires sudo)...")
        result = subprocess.run(
            ["sudo", "systemctl", "stop", "mlflow-server"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print("✓ MLflow server stopped")
            import time
            time.sleep(1)  # Wait for server to fully stop
        else:
            print(f"⚠️  Could not stop server automatically:")
            print(f"   stdout: {result.stdout}")
            print(f"   stderr: {result.stderr}")
            if not force:
                response = input("Continue anyway? (yes/no): ")
                if response.lower() != 'yes':
                    print("Aborted")
                    return False
    except FileNotFoundError:
        print("⚠️  'sudo' command not found.")
        if not force:
            response = input("Continue without stopping server? (yes/no): ")
            if response.lower() != 'yes':
                print("Aborted")
                return False
    except Exception as e:
        print(f"⚠️  Could not stop server: {e}")
        if not force:
            response = input("Continue anyway? (yes/no): ")
            if response.lower() != 'yes':
                print("Aborted")
                return False
    
    # Delete database file
    try:
        print(f"\nDeleting database file: {db_path}")
        # Check if we need sudo
        if not os.access(db_file, os.W_OK):
            print(f"⚠️  No write permission. You may need to run with sudo:")
            print(f"  sudo python {sys.argv[0]} --db-path {db_path}")
            response = input("Try anyway? (yes/no): ")
            if response.lower() != 'yes':
                print("Aborted")
                return False
        
        db_file.unlink()
        print(f"✓ Database file deleted")
    except PermissionError:
        print(f"\nERROR: Permission denied.")
        print(f"The database file is owned by root.")
        print(f"Please run with sudo:")
        print(f"  sudo python {sys.argv[0]} --db-path {db_path}")
        return False
    except Exception as e:
        print(f"ERROR: Could not delete database: {e}")
        return False
    
    # Delete artifacts if requested
    if artifacts_dir and artifacts_dir.exists():
        try:
            print(f"\nDeleting artifacts directory: {artifacts_path}")
            shutil.rmtree(artifacts_dir)
            print(f"✓ Artifacts directory deleted")
        except PermissionError:
            print(f"⚠️  Warning: Could not delete artifacts (permission denied)")
            print(f"  You may need to delete manually: {artifacts_path}")
        except Exception as e:
            print(f"⚠️  Warning: Could not delete artifacts: {e}")
    
    print("\n" + "=" * 70)
    print("Step 2: Database file deleted!")
    print("=" * 70)
    
    # Verify database is deleted
    if db_file.exists():
        print(f"\n⚠️  WARNING: Database file still exists!")
        print(f"   This should not happen. Please check manually.")
        return False
    else:
        print(f"\n✓ Verified: Database file is deleted")
    
    # Restart MLflow server
    print("\n" + "=" * 70)
    print("Step 3: Restarting MLflow server...")
    print("=" * 70)
    print("   Server will create a fresh, empty database on first access.")
    
    try:
        import subprocess
        print("   Starting MLflow server (requires sudo)...")
        result = subprocess.run(
            ["sudo", "systemctl", "start", "mlflow-server"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print("✓ MLflow server started")
            # Wait for server to fully start
            import time
            time.sleep(3)
            print("   Server is ready. Database will be created on first access.")
        else:
            print(f"⚠️  Could not start server automatically:")
            print(f"   stdout: {result.stdout}")
            print(f"   stderr: {result.stderr}")
            print("   Please start manually: sudo systemctl start mlflow-server")
    except FileNotFoundError:
        print("⚠️  'sudo' command not found. Please start manually:")
        print("   sudo systemctl start mlflow-server")
    except Exception as e:
        print(f"⚠️  Could not start server: {e}")
        print("   Please start manually: sudo systemctl start mlflow-server")
    
    # Final verification: database should not exist yet
    if db_file.exists():
        print(f"\n⚠️  NOTE: Database file was created immediately.")
        print(f"   This is normal - MLflow creates it on server start.")
        print(f"   It should be empty (no experiments/runs).")
    else:
        print(f"\n✓ Database file will be created on first MLflow access.")
        print(f"   It will be completely empty (no experiments/runs).")
    
    print("\n" + "=" * 70)
    print("✓ Hard wipe complete!")
    print("=" * 70)
    print("\nThe MLflow database has been deleted and server restarted.")
    print("\n⚠️  IMPORTANT:")
    print("   - MLflow will create a 'Default' experiment on first access")
    print("   - This is normal MLflow behavior - it's an empty experiment")
    print("   - Your actual experiments will be created when you start training")
    print("\nNext steps:")
    print("  1. Verify in MLflow UI (http://localhost:5000)")
    print("  2. You should see only an empty 'Default' experiment (this is normal)")
    print("  3. Start your training: python scripts/optimize_models.py --models swissbert --n-trials 60")
    print("  4. Your experiments will be created automatically during training")
    
    return True

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Hard wipe MLflow database (deletes entire database file)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Wipe default database
  python scripts/hard_wipe_mlflow_db.py

  # Wipe with custom path
  python scripts/hard_wipe_mlflow_db.py --db-path /custom/path/mlflow.db

  # Wipe database and artifacts
  python scripts/hard_wipe_mlflow_db.py --wipe-artifacts

  # Create backup before wiping
  python scripts/hard_wipe_mlflow_db.py --backup
        """
    )
    
    parser.add_argument(
        "--db-path",
        type=str,
        default=DEFAULT_DB_PATH,
        help=f"Path to MLflow SQLite database (default: {DEFAULT_DB_PATH})"
    )
    parser.add_argument(
        "--artifacts-path",
        type=str,
        default=None,
        help="Path to MLflow artifacts directory (optional, won't delete if not specified)"
    )
    parser.add_argument(
        "--wipe-artifacts",
        action="store_true",
        help="Also wipe artifacts directory (default: {DEFAULT_ARTIFACTS_PATH})"
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create backup of database before wiping"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompt (use with caution!)"
    )
    
    args = parser.parse_args()
    
    # Determine artifacts path
    artifacts_path = args.artifacts_path
    if args.wipe_artifacts and artifacts_path is None:
        artifacts_path = DEFAULT_ARTIFACTS_PATH
    
    success = hard_wipe_mlflow_db(
        db_path=args.db_path,
        artifacts_path=artifacts_path,
        backup=args.backup,
        force=args.force,
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

