@echo off
REM nanoHUB Deployment Script for Windows
REM Run this script on your local machine to prepare for nanoHUB deployment

echo 🚀 Starting nanoHUB deployment preparation...

REM Check if we're in the right directory
if not exist "run.py" (
    echo ❌ Error: run.py not found. Please run this script from the repository root.
    pause
    exit /b 1
)

REM Pull latest changes from GitHub
echo 📥 Pulling latest changes from GitHub...
git pull origin main

REM Install/update Python dependencies
echo 📦 Installing Python dependencies...
pip install -r config/requirements.txt

REM Check if config files exist
if not exist "config\production.env" (
    echo ⚠️  Warning: config\production.env not found. Make sure you've copied the sensitive config files.
)

REM Set environment for production
set ENVIRONMENT=production

REM Run a quick test to make sure everything works
echo 🧪 Running quick system check...
python -c "import sys; sys.path.append('.'); from src.core.config import *; print('✅ Configuration loaded successfully'); print(f'✅ Environment: {ENVIRONMENT}'); print(f'✅ Cache directory: {CACHE_DIRECTORY}'); print(f'✅ Dataset: {DATASET_CSV}')"

echo ✅ Deployment preparation completed successfully!
echo 🎯 You can now copy the nanohub-deployment folder to your nanoHUB VM
pause
