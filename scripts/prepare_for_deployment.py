#!/usr/bin/env python3
"""
Prepare repository for deployment by checking what can be safely committed to Git
and what needs to be manually copied to nanoHUB VM.
"""

import os
import sys
from pathlib import Path

def check_git_status():
    """Check what files would be committed to Git"""
    print("🔍 Checking Git status...")
    
    # Files that should be committed (safe)
    safe_files = [
        "src/",
        "scripts/", 
        "run.py",
        "setup.py",
        "README.md",
        "SYSTEM_DOCUMENTATION.md",
        "docs/",
        "tests/",
        "data/",  # Assuming CSV files are not sensitive
        "config/requirements.txt",
        "config/*.env.example",
        ".gitignore"
    ]
    
    # Files that should NOT be committed (sensitive/generated)
    sensitive_files = [
        "config/*.env",  # Contains SharePoint credentials
        "dt-cache/",     # Runtime cache
        "pareto_cache.db",  # Database
        "database_exports/",  # Generated exports
        "plots/",        # Generated plots
        "misc_temp_files/",   # Temporary files
        "notebooks/*/model_outputs/",  # Generated outputs
        "notebooks/*/plots/",  # Generated plots
        "parity_test_outputs/",  # Generated outputs
        "__pycache__/",  # Python cache
        "*.log"          # Log files
    ]
    
    print("✅ Files safe to commit to Git:")
    for file_pattern in safe_files:
        print(f"  - {file_pattern}")
    
    print("\n❌ Files that should NOT be committed (sensitive/generated):")
    for file_pattern in sensitive_files:
        print(f"  - {file_pattern}")
    
    return safe_files, sensitive_files

def check_deployment_package():
    """Check if deployment package is ready"""
    print("\n📦 Checking nanoHUB deployment package...")
    
    deployment_dir = Path("nanohub-deployment")
    if not deployment_dir.exists():
        print("❌ nanohub-deployment directory not found!")
        return False
    
    config_dir = deployment_dir / "config"
    if not config_dir.exists():
        print("❌ nanohub-deployment/config directory not found!")
        return False
    
    # Check for required config files
    required_files = [
        "config/development.env",
        "config/production.env", 
        "config/pareto_config.env",
        "config/fresh_start.env",
        "config/incremental.env",
        "config/requirements.txt"
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = deployment_dir / file_path
        if not full_path.exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing files in deployment package:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False
    
    print("✅ Deployment package is ready!")
    print("\n📋 To deploy to nanoHUB VM:")
    print("1. Copy the entire 'nanohub-deployment' folder to your nanoHUB VM")
    print("2. On the VM, copy config files: cp -r nanohub-deployment/config/* config/")
    print("3. Install dependencies: pip install -r config/requirements.txt")
    print("4. Run the application: python run.py")
    
    return True

def main():
    print("🚀 Pareto Optimization - Deployment Preparation Check")
    print("=" * 60)
    
    # Check Git safety
    safe_files, sensitive_files = check_git_status()
    
    # Check deployment package
    deployment_ready = check_deployment_package()
    
    print("\n" + "=" * 60)
    if deployment_ready:
        print("✅ Repository is ready for deployment!")
        print("\n📝 Next steps:")
        print("1. Commit safe files to Git: git add . && git commit -m 'Initial commit'")
        print("2. Push to GitHub: git push origin main")
        print("3. Copy nanohub-deployment/ to your nanoHUB VM")
        print("4. Follow the deployment instructions on the VM")
    else:
        print("❌ Repository needs preparation before deployment!")
        print("Please fix the issues above before proceeding.")

if __name__ == "__main__":
    main()
