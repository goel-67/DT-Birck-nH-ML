# Deployment Guide for Pareto Optimization System

This guide explains how to safely deploy the Pareto optimization system to GitHub and set it up on a nanoHUB VM.

## 🎯 Overview

The system is designed with a clean separation between:
- **Public code** (safe for GitHub)
- **Sensitive configuration** (manually copied to nanoHUB)
- **Generated data** (runtime cache, not versioned)

## 📁 Directory Structure

### What Gets Committed to GitHub
```
├── src/                    # Source code
├── scripts/                # Utility scripts
├── run.py                  # Main entry point
├── setup.py                # Package setup
├── README.md               # Documentation
├── SYSTEM_DOCUMENTATION.md # System docs
├── docs/                   # Additional documentation
├── tests/                  # Test files
├── data/                   # Non-sensitive data files
├── config/
│   ├── requirements.txt    # Python dependencies
│   ├── *.env.example       # Configuration templates
└── .gitignore              # Git ignore rules
```

### What Does NOT Get Committed (Sensitive/Generated)
```
├── config/*.env            # Contains SharePoint credentials
├── dt-cache/               # Runtime cache
├── pareto_cache.db         # SQLite database
├── database_exports/       # Generated exports
├── plots/                  # Generated plots
├── misc_temp_files/        # Temporary files
├── notebooks/*/outputs/    # Generated outputs
├── parity_test_outputs/    # Generated outputs
├── __pycache__/            # Python cache
├── *.log                   # Log files
└── nanohub-deployment/     # Deployment package
```

## 🚀 Deployment Process

### Step 1: Prepare Repository for GitHub

1. **Verify Git Safety**
   ```bash
   python scripts/prepare_for_deployment.py
   ```

2. **Initialize Git Repository** (if not already done)
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   ```

3. **Create GitHub Repository**
   - Create a new repository on GitHub
   - Add the remote:
   ```bash
   git remote add origin <your-github-repo-url>
   git branch -M main
   git push -u origin main
   ```

### Step 2: Set Up nanoHUB VM

1. **Clone Repository on nanoHUB VM**
   ```bash
   git clone <your-github-repo-url> pareto-optimization
   cd pareto-optimization
   ```

2. **Copy Sensitive Configuration**
   
   From your local machine, copy the `nanohub-deployment` folder to your nanoHUB VM, then:
   ```bash
   # Copy configuration files
   cp -r nanohub-deployment/config/* config/
   
   # Make deployment script executable
   chmod +x nanohub-deployment/deploy.sh
   ```

3. **Install Dependencies**
   ```bash
   pip install -r config/requirements.txt
   ```

4. **Configure Environment**
   ```bash
   export ENVIRONMENT=production
   ```

5. **Run the Application**
   ```bash
   python run.py
   ```

### Step 3: Update Deployment Process

When you make changes to the code:

1. **On Local Machine:**
   ```bash
   # Make your changes
   git add .
   git commit -m "Your changes"
   git push origin main
   ```

2. **On nanoHUB VM:**
   ```bash
   # Pull latest changes
   git pull origin main
   
   # Update dependencies if needed
   pip install -r config/requirements.txt
   
   # Run the application
   python run.py
   ```

## 🔧 Configuration Management

### Environment Files

The system uses multiple environment files for different purposes:

- `development.env` - Local development settings
- `production.env` - Production settings for nanoHUB
- `pareto_config.env` - Pareto-specific configuration
- `fresh_start.env` - Fresh start configuration
- `incremental.env` - Incremental mode configuration

### SharePoint Integration

The system integrates with SharePoint for recipe management. Credentials are stored in environment variables:

- `GRAPH_CLIENT_ID` - Azure AD application client ID
- `GRAPH_CLIENT_SECRET` - Azure AD application secret
- `GRAPH_TENANT_ID` - Azure AD tenant ID
- `GRAPH_TENANT_NAME` - SharePoint tenant name
- `GRAPH_SITE_NAME` - SharePoint site name
- `RECIPES_FILE_PATH` - Path to recipes Excel file

## 🔒 Security Best Practices

1. **Never commit `.env` files** - They contain sensitive credentials
2. **Use environment-specific configurations** - Different settings for dev/prod
3. **Rotate credentials regularly** - Update SharePoint credentials periodically
4. **Monitor logs** - Check for any security issues in application logs
5. **Keep dependencies updated** - Regularly update Python packages

## 🐛 Troubleshooting

### Common Issues

1. **Configuration not loading**
   - Check that `.env` files are in the correct locations
   - Verify environment variables are set correctly

2. **SharePoint authentication fails**
   - Verify credentials are correct in config files
   - Check that the SharePoint site is accessible

3. **Cache issues**
   - Delete `dt-cache/` and `pareto_cache.db` to start fresh
   - Set `CACHE_MANAGEMENT_MODE=full_rebuild`

4. **Dependencies issues**
   - Update requirements: `pip install -r config/requirements.txt`
   - Check Python version compatibility

### Debug Mode

Enable debug logging for troubleshooting:
```bash
export LOG_LEVEL=DEBUG
python run.py
```

## 📋 Deployment Checklist

- [ ] Repository initialized and pushed to GitHub
- [ ] Sensitive files excluded from Git (`.gitignore` working)
- [ ] Configuration templates created (`.env.example` files)
- [ ] nanoHUB deployment package created
- [ ] nanoHUB VM has repository cloned
- [ ] Sensitive config files copied to nanoHUB VM
- [ ] Dependencies installed on nanoHUB VM
- [ ] Environment variables set correctly
- [ ] Application runs successfully on nanoHUB VM
- [ ] SharePoint integration working
- [ ] Cache system functioning properly

## 🔄 Update Workflow

For regular updates:

1. **Local Development:**
   - Make changes to code
   - Test locally with `development.env`
   - Update any sensitive config if needed

2. **Deploy to GitHub:**
   - Commit and push code changes
   - Update config templates if needed

3. **Deploy to nanoHUB:**
   - Pull changes on VM
   - Update sensitive config if needed
   - Test the deployment

This workflow ensures that sensitive configuration remains secure while allowing easy code updates through Git.
