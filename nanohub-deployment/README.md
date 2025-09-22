# nanoHUB Deployment Package

This directory contains all the sensitive configuration files and data that need to be manually copied to your nanoHUB VM. **DO NOT COMMIT THIS DIRECTORY TO GIT**.

## Files to Copy to nanoHUB VM

### 1. Configuration Files (`config/` directory)
Copy the entire `config/` directory to your nanoHUB VM at the same location:
```
config/
├── development.env     # Development settings (if needed)
├── production.env      # Production settings for nanoHUB
├── pareto_config.env   # Pareto-specific configuration
├── fresh_start.env     # Fresh start configuration
├── incremental.env     # Incremental mode configuration
└── requirements.txt    # Python dependencies
```

### 2. Sensitive Data Files
If you have any Excel files with recipes or other sensitive data, copy them to:
```
data/
├── [your-sensitive-excel-files].xlsx
└── [other-sensitive-data-files]
```

## nanoHUB VM Setup Instructions

### Step 1: Clone the Repository
```bash
git clone <your-github-repo-url> pareto-optimization
cd pareto-optimization
```

### Step 2: Copy Sensitive Files
Copy all files from this `nanohub-deployment` directory to the corresponding locations in your cloned repository:
```bash
# Copy config files
cp -r nanohub-deployment/config/* config/

# Copy any sensitive data files (if applicable)
cp -r nanohub-deployment/data/* data/
```

### Step 3: Install Dependencies
```bash
pip install -r config/requirements.txt
```

### Step 4: Set Environment Variables (if needed)
You may need to set additional environment variables on the nanoHUB VM:
```bash
export ENVIRONMENT=production
```

### Step 5: Run the Application
```bash
python run.py
```

## Configuration Notes

- The `production.env` file is configured for nanoHUB deployment with appropriate settings
- SharePoint credentials should be filled in the config files
- Cache management is set to `full_rebuild` by default for production
- Logging is set to `INFO` level for production

## Security Reminders

- Never commit the actual `.env` files to Git
- Keep SharePoint credentials secure
- Regularly rotate credentials if needed
- Monitor logs for any security issues

## Troubleshooting

If you encounter issues:
1. Check that all config files are in the correct locations
2. Verify that SharePoint credentials are correctly set
3. Ensure all Python dependencies are installed
4. Check the logs for any error messages
