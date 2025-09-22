"""
Cache Manager for Pareto optimization system.
Handles cache operations with fresh start and incremental modes.
"""

import os
import shutil
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

from src.core.config import (
    ROOT_DIR, CACHE_FRESH_START, CACHE_INCREMENTAL, 
    CACHE_BACKUP_BEFORE_OVERWRITE, CACHE_VALIDATE_ON_START,
    DATABASE_FRESH_START, DATABASE_INCREMENTAL, CACHE_MANAGEMENT_MODE,
    CACHE_REGENERATE_PLOTS
)


class CacheManager:
    """Manages cache operations with flexible modes"""
    
    def __init__(self):
        """Initialize cache manager"""
        self.cache_root = Path(ROOT_DIR)
        self.backup_dir = self.cache_root / "backups"
        # Only create backup directory if cache root exists
        if self.cache_root.exists():
            self.backup_dir.mkdir(exist_ok=True)
    
    def get_cache_status(self) -> Dict[str, any]:
        """Get current cache status and configuration"""
        return {
            'cache_root': str(self.cache_root),
            'cache_exists': self.cache_root.exists(),
            'cache_size_mb': self._get_cache_size_mb(),
            'config': {
                'fresh_start': CACHE_FRESH_START,
                'incremental': CACHE_INCREMENTAL,
                'backup_before_overwrite': CACHE_BACKUP_BEFORE_OVERWRITE,
                'validate_on_start': CACHE_VALIDATE_ON_START,
                'database_fresh_start': DATABASE_FRESH_START,
                'database_incremental': DATABASE_INCREMENTAL
            },
            'backup_count': len(list(self.backup_dir.glob("*backup*")))
        }
    
    def should_use_fresh_start(self) -> bool:
        """Determine if fresh start mode should be used"""
        # Check CACHE_MANAGEMENT_MODE first (primary control)
        if CACHE_MANAGEMENT_MODE == "full_rebuild":
            print("[cache] Fresh start mode enabled via CACHE_MANAGEMENT_MODE=full_rebuild")
            return True
        
        # Fallback to legacy CACHE_FRESH_START setting
        if CACHE_FRESH_START:
            print("[cache] Fresh start mode enabled via CACHE_FRESH_START")
            return True
        
        # Incremental mode logic
        if CACHE_MANAGEMENT_MODE == "incremental":
            if not self.cache_root.exists():
                # Check if database has data before deciding on fresh start
                if self._database_has_data():
                    print("[cache] No cache directory but database has data - using incremental mode")
                    return False
                else:
                    print("[cache] No existing cache found, using fresh start")
                    return True
            
            if CACHE_VALIDATE_ON_START:
                # Check if cache is corrupted
                if not self._validate_cache_integrity():
                    print("[cache] Cache validation failed, using fresh start")
                    return True
            
            print("[cache] Using incremental mode")
            return False
        
        # Default to incremental mode
        print("[cache] Using incremental mode (default)")
        return False
    
    def _database_has_data(self) -> bool:
        """Check if database has data"""
        import sqlite3
        
        db_path = os.path.join(os.getcwd(), "pareto_cache.db")
        if not os.path.exists(db_path):
            return False
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM iterations")
            iteration_count = cursor.fetchone()[0]
            conn.close()
            
            return iteration_count > 0
        except Exception:
            return False
    
    def prepare_cache(self) -> bool:
        """Prepare cache based on configuration"""
        try:
            if self.should_use_fresh_start():
                return self._prepare_fresh_start()
            else:
                return self._prepare_incremental()
        except Exception as e:
            print(f"[cache] Error preparing cache: {e}")
            return False
    
    def _prepare_fresh_start(self) -> bool:
        """Prepare fresh start mode"""
        print("[cache] 🗑️ Preparing fresh start mode...")
        
        # Backup existing cache if requested
        if CACHE_BACKUP_BEFORE_OVERWRITE and self.cache_root.exists():
            backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backup_path = self.backup_dir / backup_name
            print(f"[cache] 📦 Creating backup: {backup_path}")
            
            # Copy cache contents excluding the backups directory to avoid recursion
            try:
                # Create backup directory
                backup_path.mkdir(parents=True, exist_ok=True)
                
                # Copy all files and directories except 'backups'
                for item in self.cache_root.iterdir():
                    if item.name != 'backups':
                        if item.is_file():
                            shutil.copy2(item, backup_path / item.name)
                        elif item.is_dir():
                            shutil.copytree(item, backup_path / item.name)
            except Exception as e:
                print(f"[cache] ⚠️ Backup creation failed: {e}")
                # Continue without backup
        
        # Remove existing cache
        if self.cache_root.exists():
            print("[cache] 🗑️ Removing existing cache...")
            shutil.rmtree(self.cache_root)
        
        # Create fresh cache structure
        print("[cache] 📁 Creating fresh cache structure...")
        self._create_cache_structure()
        
        print("[cache] ✅ Fresh start mode prepared")
        return True
    
    def _prepare_incremental(self) -> bool:
        """Prepare incremental mode"""
        print("[cache] 🔄 Preparing incremental mode...")
        
        # Create cache structure if it doesn't exist
        if not self.cache_root.exists():
            print("[cache] 📁 Creating cache structure for incremental mode...")
            self._create_cache_structure()
            
            # If database has data but cache directory was empty, regenerate from database
            if self._database_has_data():
                print("[cache] 🔄 Database has data but cache was empty - triggering regeneration...")
                return self._trigger_cache_regeneration()
        
        # Validate cache integrity
        if CACHE_VALIDATE_ON_START:
            if not self._validate_cache_integrity():
                print("[cache] ⚠️ Cache validation failed, but continuing with incremental mode")
        
        print("[cache] ✅ Incremental mode prepared")
        return True
    
    def _trigger_cache_regeneration(self) -> bool:
        """Trigger cache regeneration from database"""
        try:
            # Import here to avoid circular imports
            from ..core.config import ROOT_DIR
            from .database_manager import DatabaseManager
            
            print("[cache] 🔄 Triggering cache regeneration from database...")
            if CACHE_REGENERATE_PLOTS:
                print("[cache] 📊 Plot generation enabled during cache regeneration")
            else:
                print("[cache] ⚡ Plot generation disabled during cache regeneration (faster startup)")
            
            db_manager = DatabaseManager()
            success = db_manager.regenerate_cache_from_database(ROOT_DIR)
            
            if success:
                print("[cache] ✅ Cache regeneration completed successfully")
                return True
            else:
                print("[cache] ❌ Cache regeneration failed")
                return False
                
        except Exception as e:
            print(f"[cache] ❌ Error during cache regeneration: {e}")
            return False
    
    def _create_cache_structure(self) -> None:
        """Create the cache directory structure"""
        directories = [
            "iterations",
            "rolling", 
            "snapshots",
            "backups"
        ]
        
        for dir_name in directories:
            dir_path = self.cache_root / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create iteration subdirectories based on actual data
        # We'll create them dynamically as needed, but start with a reasonable number
        # The main system will determine the actual number of iterations
        for i in range(1, 6):  # Support up to 5 iterations by default
            iteration_dir = self.cache_root / "iterations" / f"iteration_{i}"
            plots_dir = iteration_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
    
    def _validate_cache_integrity(self) -> bool:
        """Validate cache integrity"""
        try:
            # Check if essential directories exist
            essential_dirs = ["iterations", "rolling", "snapshots"]
            for dir_name in essential_dirs:
                dir_path = self.cache_root / dir_name
                if not dir_path.exists():
                    print(f"[cache] ❌ Missing essential directory: {dir_name}")
                    return False
            
            # Check if manifest exists (try both locations)
            manifest_path = self.cache_root / "manifest.json"
            if not manifest_path.exists():
                manifest_path = self.cache_root / "manifests" / "latest.json"
                if not manifest_path.exists():
                    print("[cache] ❌ Missing manifest.json")
                    return False
            
            # Validate manifest structure
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            required_keys = ["updated_at", "code_version", "new_completed_lots"]
            for key in required_keys:
                if key not in manifest:
                    print(f"[cache] ❌ Missing required manifest key: {key}")
                    return False
            
            print("[cache] ✅ Cache integrity validation passed")
            return True
            
        except Exception as e:
            print(f"[cache] ❌ Cache validation error: {e}")
            return False
    
    def _get_cache_size_mb(self) -> float:
        """Get cache size in MB"""
        try:
            if not self.cache_root.exists():
                return 0.0
            
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(self.cache_root):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
            
            return total_size / (1024 * 1024)  # Convert to MB
        except Exception:
            return 0.0
    
    def get_backup_list(self) -> List[Dict[str, any]]:
        """Get list of available backups"""
        backups = []
        for backup_dir in self.backup_dir.glob("*backup*"):
            try:
                backup_info = {
                    'name': backup_dir.name,
                    'path': str(backup_dir),
                    'size_mb': self._get_dir_size_mb(backup_dir),
                    'created': datetime.fromtimestamp(backup_dir.stat().st_ctime).isoformat()
                }
                backups.append(backup_info)
            except Exception as e:
                print(f"[cache] Error reading backup {backup_dir.name}: {e}")
        
        return sorted(backups, key=lambda x: x['created'], reverse=True)
    
    def restore_backup(self, backup_name: str) -> bool:
        """Restore cache from backup"""
        try:
            backup_path = self.backup_dir / backup_name
            if not backup_path.exists():
                print(f"[cache] ❌ Backup not found: {backup_name}")
                return False
            
            # Remove current cache
            if self.cache_root.exists():
                shutil.rmtree(self.cache_root)
            
            # Restore from backup
            print(f"[cache] 🔄 Restoring from backup: {backup_name}")
            shutil.copytree(backup_path, self.cache_root)
            
            print("[cache] ✅ Backup restored successfully")
            return True
            
        except Exception as e:
            print(f"[cache] ❌ Error restoring backup: {e}")
            return False
    
    def _get_dir_size_mb(self, dir_path: Path) -> float:
        """Get directory size in MB"""
        try:
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(dir_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
            return total_size / (1024 * 1024)
        except Exception:
            return 0.0
    
    def ensure_iteration_directory(self, iteration_num: int) -> bool:
        """Ensure iteration directory exists for a specific iteration number"""
        try:
            iteration_dir = self.cache_root / "iterations" / f"iteration_{iteration_num}"
            plots_dir = iteration_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            print(f"[cache] Error creating iteration directory {iteration_num}: {e}")
            return False
    
    def cleanup_old_backups(self, keep_count: int = 5) -> int:
        """Clean up old backups, keeping only the most recent ones"""
        try:
            backups = self.get_backup_list()
            if len(backups) <= keep_count:
                return 0
            
            # Remove oldest backups
            backups_to_remove = backups[keep_count:]
            removed_count = 0
            
            for backup in backups_to_remove:
                backup_path = Path(backup['path'])
                if backup_path.exists():
                    shutil.rmtree(backup_path)
                    removed_count += 1
                    print(f"[cache] 🗑️ Removed old backup: {backup['name']}")
            
            print(f"[cache] ✅ Cleaned up {removed_count} old backups")
            return removed_count
            
        except Exception as e:
            print(f"[cache] ❌ Error cleaning up backups: {e}")
            return 0
