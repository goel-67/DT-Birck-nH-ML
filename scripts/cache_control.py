"""
Cache Control Script for Pareto optimization system.
Provides easy command-line interface for cache management.
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.cache_manager import CacheManager
from src.core.config import (
    CACHE_FRESH_START, CACHE_INCREMENTAL, CACHE_BACKUP_BEFORE_OVERWRITE,
    DATABASE_FRESH_START, DATABASE_INCREMENTAL
)


def main():
    parser = argparse.ArgumentParser(description="Cache Control for Pareto System")
    parser.add_argument("command", choices=[
        "status", "fresh-start", "incremental", "backup", "restore", 
        "cleanup", "list-backups", "set-mode"
    ], help="Command to execute")
    
    parser.add_argument("--backup-name", help="Backup name for restore command")
    parser.add_argument("--keep-count", type=int, default=5, help="Number of backups to keep")
    parser.add_argument("--mode", choices=["fresh-start", "incremental"], help="Set cache mode")
    parser.add_argument("--database-mode", choices=["fresh-start", "incremental"], help="Set database mode")
    
    args = parser.parse_args()
    
    cache_manager = CacheManager()
    
    if args.command == "status":
        show_status(cache_manager)
    
    elif args.command == "fresh-start":
        set_fresh_start_mode()
    
    elif args.command == "incremental":
        set_incremental_mode()
    
    elif args.command == "backup":
        create_backup(cache_manager)
    
    elif args.command == "restore":
        if not args.backup_name:
            print("❌ Error: --backup-name required for restore command")
            return
        restore_backup(cache_manager, args.backup_name)
    
    elif args.command == "cleanup":
        cleanup_backups(cache_manager, args.keep_count)
    
    elif args.command == "list-backups":
        list_backups(cache_manager)
    
    elif args.command == "set-mode":
        if not args.mode:
            print("❌ Error: --mode required for set-mode command")
            return
        set_cache_mode(args.mode, args.database_mode)


def show_status(cache_manager):
    """Show current cache status"""
    status = cache_manager.get_cache_status()
    
    print("🗄️ Cache Status")
    print("=" * 50)
    print(f"Cache Root: {status['cache_root']}")
    print(f"Cache Exists: {'✅ Yes' if status['cache_exists'] else '❌ No'}")
    print(f"Cache Size: {status['cache_size_mb']:.2f} MB")
    print(f"Backup Count: {status['backup_count']}")
    print()
    
    print("⚙️ Configuration")
    print("-" * 30)
    config = status['config']
    print(f"Cache Fresh Start: {'✅ Enabled' if config['fresh_start'] else '❌ Disabled'}")
    print(f"Cache Incremental: {'✅ Enabled' if config['incremental'] else '❌ Disabled'}")
    print(f"Backup Before Overwrite: {'✅ Enabled' if config['backup_before_overwrite'] else '❌ Disabled'}")
    print(f"Validate On Start: {'✅ Enabled' if config['validate_on_start'] else '❌ Disabled'}")
    print(f"Database Fresh Start: {'✅ Enabled' if config['database_fresh_start'] else '❌ Disabled'}")
    print(f"Database Incremental: {'✅ Enabled' if config['database_incremental'] else '❌ Disabled'}")


def set_fresh_start_mode():
    """Set system to fresh start mode"""
    print("🔄 Setting Fresh Start Mode...")
    
    # Set environment variables
    os.environ["CACHE_FRESH_START"] = "true"
    os.environ["CACHE_INCREMENTAL"] = "false"
    os.environ["DATABASE_FRESH_START"] = "true"
    os.environ["DATABASE_INCREMENTAL"] = "false"
    
    print("✅ Fresh start mode enabled!")
    print("📝 Next run will:")
    print("   - Create backup of existing cache (if enabled)")
    print("   - Remove all existing cache and database")
    print("   - Recreate everything from scratch")


def set_incremental_mode():
    """Set system to incremental mode"""
    print("🔄 Setting Incremental Mode...")
    
    # Set environment variables
    os.environ["CACHE_FRESH_START"] = "false"
    os.environ["CACHE_INCREMENTAL"] = "true"
    os.environ["DATABASE_FRESH_START"] = "false"
    os.environ["DATABASE_INCREMENTAL"] = "true"
    
    print("✅ Incremental mode enabled!")
    print("📝 Next run will:")
    print("   - Use existing cache and database")
    print("   - Add only new data")
    print("   - Preserve all existing calculations")


def create_backup(cache_manager):
    """Create a manual backup"""
    print("📦 Creating manual backup...")
    
    if not cache_manager.cache_root.exists():
        print("❌ No cache to backup")
        return
    
    backup_name = f"manual_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_path = cache_manager.backup_dir / backup_name
    
    try:
        # Create backup directory
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # Copy all files and directories except 'backups'
        for item in cache_manager.cache_root.iterdir():
            if item.name != 'backups':
                if item.is_file():
                    import shutil
                    shutil.copy2(item, backup_path / item.name)
                elif item.is_dir():
                    import shutil
                    shutil.copytree(item, backup_path / item.name)
        
        print(f"✅ Backup created: {backup_name}")
    except Exception as e:
        print(f"❌ Backup failed: {e}")


def restore_backup(cache_manager, backup_name):
    """Restore from backup"""
    print(f"🔄 Restoring from backup: {backup_name}")
    
    if cache_manager.restore_backup(backup_name):
        print("✅ Backup restored successfully!")
    else:
        print("❌ Backup restore failed!")


def cleanup_backups(cache_manager, keep_count):
    """Clean up old backups"""
    print(f"🧹 Cleaning up backups (keeping {keep_count} most recent)...")
    
    removed = cache_manager.cleanup_old_backups(keep_count)
    if removed > 0:
        print(f"✅ Removed {removed} old backups")
    else:
        print("✅ No old backups to remove")


def list_backups(cache_manager):
    """List available backups"""
    backups = cache_manager.get_backup_list()
    
    if not backups:
        print("📭 No backups found")
        return
    
    print("📦 Available Backups")
    print("=" * 50)
    
    for i, backup in enumerate(backups, 1):
        print(f"{i}. {backup['name']}")
        print(f"   Size: {backup['size_mb']:.2f} MB")
        print(f"   Created: {backup['created']}")
        print()


def set_cache_mode(cache_mode, database_mode=None):
    """Set specific cache and database modes"""
    print(f"🔄 Setting cache mode: {cache_mode}")
    
    if cache_mode == "fresh-start":
        os.environ["CACHE_FRESH_START"] = "true"
        os.environ["CACHE_INCREMENTAL"] = "false"
    else:
        os.environ["CACHE_FRESH_START"] = "false"
        os.environ["CACHE_INCREMENTAL"] = "true"
    
    if database_mode:
        print(f"🔄 Setting database mode: {database_mode}")
        if database_mode == "fresh-start":
            os.environ["DATABASE_FRESH_START"] = "true"
            os.environ["DATABASE_INCREMENTAL"] = "false"
        else:
            os.environ["DATABASE_FRESH_START"] = "false"
            os.environ["DATABASE_INCREMENTAL"] = "true"
    
    print("✅ Mode set successfully!")


if __name__ == "__main__":
    main()
