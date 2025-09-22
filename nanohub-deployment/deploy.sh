#!/bin/bash

# nanoHUB Deployment Script
# Run this script on your nanoHUB VM to deploy updates

set -e  # Exit on any error

echo "🚀 Starting nanoHUB deployment..."

# Check if we're in the right directory
if [ ! -f "run.py" ]; then
    echo "❌ Error: run.py not found. Please run this script from the repository root."
    exit 1
fi

# Pull latest changes from GitHub
echo "📥 Pulling latest changes from GitHub..."
git pull origin main

# Install/update Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r config/requirements.txt

# Check if config files exist
if [ ! -f "config/production.env" ]; then
    echo "⚠️  Warning: config/production.env not found. Make sure you've copied the sensitive config files."
fi

# Set environment for production
export ENVIRONMENT=production

# Run a quick test to make sure everything works
echo "🧪 Running quick system check..."
python -c "
import sys
sys.path.append('.')
from src.core.config import *
print('✅ Configuration loaded successfully')
print(f'✅ Environment: {ENVIRONMENT}')
print(f'✅ Cache directory: {CACHE_DIRECTORY}')
print(f'✅ Dataset: {DATASET_CSV}')
"

echo "✅ Deployment completed successfully!"
echo "🎯 You can now run: python run.py"
