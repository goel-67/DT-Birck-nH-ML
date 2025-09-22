# 🚀 Copy This Entire Folder to Your nanoHUB VM

## What You Need to Do

### 1. Copy This Folder
Copy the entire `nanohub-deployment` folder to your nanoHUB VM. You can use:
- SCP: `scp -r nanohub-deployment/ user@nanohub-vm:/path/to/destination/`
- SFTP: Upload the entire folder via SFTP client
- Manual copy: If you have direct access to the VM

### 2. On Your nanoHUB VM

#### Step A: Clone the Repository
```bash
git clone <your-github-repo-url> pareto-optimization
cd pareto-optimization
```

#### Step B: Copy Configuration Files
```bash
# Copy all config files from the deployment package
cp -r nanohub-deployment/config/* config/
```

#### Step C: Install Dependencies
```bash
pip install -r config/requirements.txt
```

#### Step D: Set Environment
```bash
export ENVIRONMENT=production
```

#### Step E: Run the Application
```bash
python run.py
```

### 3. For Future Updates

When you push changes to GitHub, update your nanoHUB VM:
```bash
# Pull latest changes
git pull origin main

# Update dependencies if needed
pip install -r config/requirements.txt

# Run the application
python run.py
```

## What's in This Package

- `config/` - All your sensitive configuration files with SharePoint credentials
- `deploy.sh` - Linux deployment script for the VM
- `deploy.bat` - Windows deployment script for your local machine
- `README.md` - Detailed deployment instructions
- `COPY_TO_NANOHUB.md` - This file (quick reference)

## ⚠️ Important Security Notes

- **DO NOT** commit this folder to Git
- **DO NOT** share the contents of this folder publicly
- Keep your SharePoint credentials secure
- Regularly rotate credentials if needed

## 🆘 Need Help?

1. Check the main `DEPLOYMENT.md` file in the repository root
2. Run `python scripts/prepare_for_deployment.py` to check your setup
3. Check the logs if the application doesn't start: `tail -f *.log`

---

**You're all set!** 🎉
