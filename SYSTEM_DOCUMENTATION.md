# Pareto Optimization System - Complete Documentation

## **📋 Table of Contents**
1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Data Management](#data-management)
4. [Database System](#database-system)
5. [Cache Management System](#cache-management-system)
6. [Cache Validation](#cache-validation)
7. [Configuration Management](#configuration-management)
8. [Core Components](#core-components)
9. [Machine Learning Models](#machine-learning-models)
10. [Optimization Engine](#optimization-engine)
11. [Visualization System](#visualization-system)
12. [Production Features](#production-features)
13. [File Structure](#file-structure)
14. [Usage Instructions](#usage-instructions)
15. [Command Reference](#command-reference)
16. [API Integration](#api-integration)

---

## **🎯 System Overview**

The Pareto Optimization System is a comprehensive machine learning and optimization platform designed for semiconductor manufacturing process optimization. It implements an iterative Pareto front optimization approach using advanced ML models to propose optimal recipe parameters.

### **Key Features:**
- **Iterative Optimization**: Sequential improvement of Pareto fronts
- **Multi-Model Support**: Random Forest, Extra Trees with iteration-specific selection
- **Comprehensive Caching**: File cache + SQLite database for reliability
- **Production-Grade**: Cache validation, configuration management, monitoring
- **nanoHUB Integration**: Ready for cloud deployment and API access
- **Real-time Processing**: Daily automated optimization cycles

---

## **🏗️ Architecture**

### **High-Level Architecture:**
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │  Core System   │    │   Outputs       │
│                 │    │                 │    │                 │
│ • SharePoint    │───▶│ • Main Engine  │───▶│ • Plots         │
│ • Azure Storage │    │ • ML Models    │    │ • Database      │
│ • Local CSV     │    │ • Optimizer    │    │ • Cache Files   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │  Cache System   │
                       │                 │
                       │ • File Cache    │
                       │ • SQLite DB     │
                       │ • Validation    │
                       └─────────────────┘
```

### **Data Flow:**
1. **Ingestion**: Load dataset from Azure/local + SharePoint recipes
2. **Processing**: Train models, generate candidates, optimize Pareto front
3. **Caching**: Save results to both file cache and SQLite database
4. **Validation**: Ensure data integrity and consistency
5. **Output**: Generate plots, exports, and API-ready data

---

## **📊 Data Management**

### **Data Sources:**
- **`full_dataset.csv`**: Complete historical manufacturing data (107 points)
- **SharePoint Excel**: Real-time recipe status and completion data
- **Azure Storage**: Future integration for automated data ingestion

### **Data Structure:**
```python
# Core Dataset Columns
{
    'LOTNAME': 'etch8/22/2024',
    'FIMAP_FILE': 'fimap_data.csv',
    'AvgEtchRate': 45.2,
    'RangeEtchRate': 2.1,
    'Range_nm': 8.5,
    'run_date': '2024-08-22',
    # Feature columns for ML
    'Etch_AvgO2Flow': 25.0,
    'Etch_Avgcf4Flow': 35.0,
    'Etch_Avg_Rf1_Pow': 50.0,
    'Etch_Avg_Rf2_Pow': 300.0,
    'Etch_AvgPres': 15.0
}
```

### **Iteration System:**
- **Training Logic**: For iteration N, train on all data where `run_date < Date_Completed` of first point in iteration N
- **Points per Iteration**: 3 new recipes proposed per iteration
- **Model Selection**: Iteration-specific model configuration (RF/Extra Trees)

---

## **🗄️ Database System**

### **SQLite Database Schema:**

#### **Core Tables:**
1. **`iterations`** - Iteration metadata and training info
2. **`highlight_lots`** - Lots highlighted for each iteration
3. **`pareto_fronts`** - Pareto front data points
4. **`model_predictions`** - ML model predictions and actual results
5. **`historical_data`** - Complete dataset (107 points)
6. **`training_data_snapshots`** - Training data for each iteration
7. **`processing_logs`** - Processing history and performance metrics
8. **`system_metadata`** - System configuration and version info
9. **`data_versioning`** - Data change tracking (future use)

### **Database Features:**
- **Complete Data Storage**: All 107 historical points + iteration data
- **Incremental Updates**: Sequential data addition without recreation
- **Data Integrity**: Foreign key relationships and constraints
- **Export Capability**: CSV exports for all tables
- **Size**: ~80KB with comprehensive data coverage

### **Database Location:**
```
C:\Users\laksh\Downloads\Pareto_backup\pareto_cache.db
```

---

## **🔍 Cache Validation**

### **Validation System:**
The `CacheValidator` class provides comprehensive data integrity checks:

#### **Validation Checks:**
1. **File Cache Structure**: Required directories and files
2. **Database Structure**: Table existence and data counts
3. **Data Consistency**: File cache ↔ Database synchronization
4. **Iteration Completeness**: Required data for each iteration
5. **Data Freshness**: Timestamp validation and staleness detection
6. **File Integrity**: Hash validation for data corruption detection

#### **Validation Results:**
```python
{
    'timestamp': '2025-09-02T17:18:20.108894',
    'overall_status': 'passed|warning|failed',
    'checks': {
        'file cache structure': {'passed': True, 'errors': [], 'warnings': []},
        'database structure': {'passed': True, 'errors': [], 'warnings': []},
        # ... other checks
    },
    'errors': ['Critical issues found'],
    'warnings': ['Minor issues detected']
}
```

#### **Usage:**
```python
from src.data.cache_validator import CacheValidator
from src.data.database_manager import DatabaseManager

db_manager = DatabaseManager()
validator = CacheValidator(db_manager)
results = validator.validate_cache_integrity()
report = validator.generate_validation_report(results)
```

---

## **🗂️ Cache Management System**

### **Overview:**
The Cache Management System provides flexible control over cache operations with two primary modes: **Fresh Start** and **Incremental**. This system ensures data integrity while allowing developers to easily switch between modes during development and deployment.

### **Cache Modes:**

#### **1. Fresh Start Mode**
- **Purpose**: Complete cache recreation from scratch
- **Use Case**: Development, testing, or when cache corruption is detected
- **Behavior**: 
  - Creates backup of existing cache (if enabled)
  - Removes all existing cache and database
  - Recreates everything from scratch
  - Recalculates all iterations and plots

#### **2. Incremental Mode**
- **Purpose**: Use existing cache and add only new data
- **Use Case**: Production runs, daily updates
- **Behavior**:
  - Uses existing cache and database
  - Adds only new data
  - Preserves all existing calculations
  - Validates cache integrity before use

### **Cache Manager (`src/data/cache_manager.py`)**

#### **Key Methods:**
```python
class CacheManager:
    def prepare_cache(self) -> bool:
        """Prepare cache based on configuration"""
    
    def should_use_fresh_start(self) -> bool:
        """Determine if fresh start mode should be used"""
    
    def ensure_iteration_directory(self, iteration_num: int) -> bool:
        """Ensure iteration directory exists for specific iteration"""
    
    def get_cache_status(self) -> Dict[str, any]:
        """Get current cache status and configuration"""
    
    def create_backup(self) -> bool:
        """Create manual backup of current cache"""
    
    def restore_backup(self, backup_name: str) -> bool:
        """Restore cache from backup"""
    
    def cleanup_old_backups(self, keep_count: int = 5) -> int:
        """Clean up old backups, keeping only recent ones"""
```

#### **Configuration Variables:**
```python
# Cache Management
CACHE_FRESH_START = os.getenv("CACHE_FRESH_START", "false").lower() == "true"
CACHE_INCREMENTAL = os.getenv("CACHE_INCREMENTAL", "true").lower() == "true"
CACHE_BACKUP_BEFORE_OVERWRITE = os.getenv("CACHE_BACKUP_BEFORE_OVERWRITE", "true").lower() == "true"
CACHE_VALIDATE_ON_START = os.getenv("CACHE_VALIDATE_ON_START", "true").lower() == "true"
CACHE_AUTO_REPAIR = os.getenv("CACHE_AUTO_REPAIR", "false").lower() == "true"

# Database Management
DATABASE_FRESH_START = os.getenv("DATABASE_FRESH_START", "false").lower() == "true"
DATABASE_INCREMENTAL = os.getenv("DATABASE_INCREMENTAL", "true").lower() == "true"
```

### **Environment Configuration Files:**

#### **Fresh Start Mode (`config/fresh_start.env`):**
```bash
# Fresh Start Mode Configuration
CACHE_FRESH_START=true
CACHE_INCREMENTAL=false
CACHE_BACKUP_BEFORE_OVERWRITE=true
CACHE_VALIDATE_ON_START=true
CACHE_AUTO_REPAIR=false
DATABASE_FRESH_START=true
DATABASE_INCREMENTAL=false
ENVIRONMENT=development
```

#### **Incremental Mode (`config/incremental.env`):**
```bash
# Incremental Mode Configuration
CACHE_FRESH_START=false
CACHE_INCREMENTAL=true
CACHE_BACKUP_BEFORE_OVERWRITE=true
CACHE_VALIDATE_ON_START=true
CACHE_AUTO_REPAIR=false
DATABASE_FRESH_START=false
DATABASE_INCREMENTAL=true
ENVIRONMENT=development
```

### **Cache Structure:**
```
dt-cache/
├── iterations/
│   ├── iteration_1/
│   │   ├── plots/
│   │   │   ├── 1_pareto_proposed.png
│   │   │   ├── 2_parity_rate_horizontal.png
│   │   │   ├── 3_parity_range_horizontal.png
│   │   │   ├── 4_pareto_predicted_actual.png
│   │   │   ├── 5_parity_rate_actual.png
│   │   │   ├── 6_parity_range_actual.png
│   │   │   ├── 7_metrics_rmse.png
│   │   │   ├── 8_metrics_coverage.png
│   │   │   └── 9_debug_pareto_hypothetical.png
│   │   ├── highlight_lots.txt
│   │   ├── iteration_status.json
│   │   ├── pareto_front.csv
│   │   ├── selected_points.csv
│   │   └── summary_data.csv
│   └── iteration_2/
│       └── ...
├── rolling/
│   ├── loocv_predictions.csv
│   └── metrics_over_time.csv
├── snapshots/
│   └── 2025-09-03/
│       ├── plots/
│       └── proposals_2025-09-03.xlsx
├── backups/
│   ├── backup_20250903_182500/
│   └── manual_backup_20250903_183000/
└── manifests/
    └── latest.json
```

---

## **⚙️ Configuration Management**

### **Environment Configuration:**
The system supports multiple environment configurations:

#### **Configuration Files:**
- **`.env`** (root): Production credentials and secrets
- **`config/pareto_config.env`**: Local development settings
- **`config/production.env`**: Production environment settings
- **`config/development.env`**: Development environment settings

#### **Environment Variables:**
```bash
# Data Sources
AZURE_DATASET_URL=https://your-azure-storage.blob.core.windows.net/datasets/full_dataset.csv
AZURE_LAST_PROCESSED_TIMESTAMP_URL=https://your-azure-storage.blob.core.windows.net/datasets/last_processed_timestamp.txt
SHAREPOINT_CREDENTIALS_ENABLED=true

# Database Configuration
SQLITE_DB_PATH=pareto_cache.db
CACHE_DIRECTORY=dt-cache
DATABASE_BACKUP_ENABLED=true

# Processing Configuration
MAX_ITERATIONS=10
POINTS_PER_ITERATION=3
MODEL_CONFIGURATION_VERSION=v1.3.0

# nanoHUB Integration
NANOHUB_API_ENDPOINT=https://nanohub.org/api/v1
NANOHUB_CREDENTIALS_ENABLED=false

# Logging
LOG_LEVEL=INFO
LOG_FILE=pareto_optimization.log

# Performance
ENABLE_PARALLEL_PROCESSING=true
MAX_WORKERS=4
CACHE_VALIDATION_ENABLED=true
```

### **Configuration Loading:**
```python
# Automatic environment detection
ENV = os.getenv("ENVIRONMENT", "development")
if ENV == "production":
    load_dotenv("config/production.env")
else:
    load_dotenv("config/development.env")
```

---

## **🔧 Core Components**

### **1. Main Engine (`src/core/main.py`)**
**Purpose**: Orchestrates the entire optimization process

**Key Methods:**
- `run_main_optimization()`: Main execution flow
- `_process_iterations()`: Iteration processing logic
- `_create_comprehensive_iteration_plots()`: Plot generation
- `_save_iteration_data_with_excel_info()`: Data persistence

**Features:**
- SharePoint integration for recipe data
- LOOCV (Leave-One-Out Cross Validation)
- Comprehensive plot generation (9 plots per iteration)
- Database integration for data persistence

### **2. Data Manager (`src/data/data_manager.py`)**
**Purpose**: Handles all data operations and Excel integration

**Key Methods:**
- `load_dataset()`: Load and preprocess dataset
- `read_recipes_excel()`: SharePoint Excel integration
- `get_training_data_debug_info()`: Training data debugging
- `save_iteration_data_to_database()`: Database persistence

**Features:**
- Microsoft Graph API integration
- Training data filtering by date
- Debug information generation
- Database facade operations

### **3. Database Manager (`src/data/database_manager.py`)**
**Purpose**: SQLite database operations and management

**Key Methods:**
- `save_iteration_data()`: Save complete iteration data
- `get_iteration_data()`: Retrieve iteration information
- `export_to_csv()`: Export all tables to CSV
- `get_database_info()`: Database statistics

**Features:**
- Complete historical data storage
- Incremental data updates
- Processing logs and metadata
- Data versioning support

---

## **🤖 Machine Learning Models**

### **Model Configuration:**
The system uses iteration-specific model selection:

```python
# Iteration 1-2: Random Forest for both models
if iteration_num <= 2:
    rate_model = RandomForestRegressor(random_state=0, n_estimators=100)
    range_model = RandomForestRegressor(random_state=0, n_estimators=100)

# Iteration 3-4: Extra Trees for etch rate, Random Forest for range
elif iteration_num in [3, 4]:
    rate_model = ExtraTreesRegressor(random_state=0, n_estimators=100)
    range_model = RandomForestRegressor(random_state=0, n_estimators=100)

# Iteration 5+: Default configuration (to be specified)
else:
    rate_model = ExtraTreesRegressor(random_state=0, n_estimators=100)
    range_model = RandomForestRegressor(random_state=0, n_estimators=100)
```

### **Model Features:**
- **Target Variables**: `AvgEtchRate` (rate) and `Range_nm` (range)
- **Input Features**: 5 process parameters (O2, CF4, RF1, RF2, Pressure)
- **Training Data**: Historical data filtered by completion date
- **Validation**: LOOCV for model performance assessment

### **Model Performance:**
- **R² Score**: Model fit quality
- **RMSE**: Prediction accuracy
- **MAE**: Mean absolute error
- **Uncertainty Estimation**: Prediction confidence intervals

---

## **🎯 Optimization Engine**

### **Pareto Optimization (`src/optimization/pareto_optimizer.py`)**
**Purpose**: Pareto front calculation and optimization

**Key Methods:**
- `calculate_pareto_front()`: Pareto front computation
- `propose_recipes()`: Recipe proposal generation
- `evaluate_candidates()`: Candidate evaluation

**Features:**
- Sobol sequence sampling for candidate generation
- Multi-objective optimization (rate vs range)
- Dominance checking and Pareto front updates
- Uncertainty-aware optimization

### **Sampling System (`src/ml/sampler.py`)**
**Purpose**: Generate candidate points in feature space

**Features:**
- Sobol sequence for quasi-random sampling
- Feature range constraints
- Target rate filtering
- RF2 > RF1 constraint enforcement

---

## **📈 Visualization System**

### **Plotter (`src/visualization/plotter.py`)**
**Purpose**: Comprehensive plotting and visualization

**Plot Types (9 plots per iteration):**
1. **Pareto Front with Proposed Recipes**: Current front + new proposals
2. **Parity Plot - Etch Rate**: Predicted vs actual etch rates
3. **Parity Plot - Range**: Predicted vs actual ranges
4. **Pareto Front with Predicted and Actual**: Historical + predicted points
5. **Parity Plot - Actual Etch Rate**: Highlighted lots etch rates
6. **Parity Plot - Actual Range**: Highlighted lots ranges
7. **Metrics Plot - RMSE**: Model performance over time
8. **Metrics Plot - Coverage**: Pareto front coverage metrics
9. **Debug Pareto Front**: Hypothetical completion analysis

**Features:**
- Dynamic y-axis limits based on highlighted lots
- Enhanced legends with point counts
- Highlighting of current iteration lots
- Uncertainty visualization
- Professional formatting and styling

---

## **🚀 Production Features**

### **1. Cache Validation**
- **Automated Health Checks**: Daily validation before processing
- **Data Integrity**: Hash validation and corruption detection
- **Consistency Monitoring**: File cache ↔ Database synchronization
- **Freshness Tracking**: Data staleness detection

### **2. Configuration Management**
- **Environment-Specific**: Production vs development settings
- **Secure Credentials**: Environment variable management
- **Scalable Configuration**: Easy deployment configuration

### **3. Database Integration**
- **Complete Data Storage**: All historical data + iteration data
- **Incremental Updates**: Sequential data addition
- **Export Capability**: CSV exports for external analysis
- **API Ready**: Structured data for nanoHUB integration

### **4. Monitoring and Logging**
- **Processing Logs**: Detailed execution tracking
- **Performance Metrics**: Processing time and efficiency
- **Error Handling**: Comprehensive error capture and reporting
- **Validation Reports**: Detailed health check reports

### **5. nanoHUB Integration Ready**
- **Database Exports**: CSV files for API consumption
- **Structured Data**: Consistent data format
- **Cron Job Ready**: Daily automated execution
- **API Endpoints**: Ready for Daniel's nanoHUB API

---

## **📁 File Structure**

```
Pareto_backup/
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── main.py              # Main orchestration engine
│   │   └── config.py            # Configuration management
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_manager.py      # Data operations and Excel integration
│   │   ├── database_manager.py  # SQLite database operations
│   │   ├── cache_validator.py   # Cache validation system
│   │   └── excel_manager.py     # Excel file handling
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── ml_models.py         # Machine learning models
│   │   └── sampler.py           # Candidate sampling
│   ├── optimization/
│   │   ├── __init__.py
│   │   └── pareto_optimizer.py  # Pareto optimization engine
│   └── visualization/
│       ├── __init__.py
│       └── plotter.py           # Comprehensive plotting system
├── config/
│   ├── pareto_config.env        # Local configuration
│   ├── production.env            # Production settings
│   └── development.env           # Development settings
├── scripts/
│   ├── test_database_simple.py  # Database testing
│   ├── test_cache_validation.py # Cache validation testing
│   └── create_test_recipes.py   # Test data generation
├── data/
│   ├── full_dataset.csv         # Complete historical dataset
│   └── test_pareto_recipes.xlsx # Test recipe data
├── dt-cache/                    # File cache directory
│   ├── iterations/              # Iteration-specific data
│   ├── rolling/                # Rolling statistics
│   ├── snapshots/              # Daily snapshots
│   └── database_exports/       # CSV exports
├── pareto_cache.db             # SQLite database
├── .env                        # Root environment variables
├── requirements.txt            # Python dependencies
├── SYSTEM_DOCUMENTATION.md     # This documentation
└── README.md                   # Quick start guide
```

---

## **📖 Usage Instructions**

### **1. Cache Management:**
```bash
# Check current cache status
python scripts/cache_control.py status

# Set fresh start mode (recreate everything)
python scripts/cache_control.py fresh-start

# Set incremental mode (use existing cache)
python scripts/cache_control.py incremental

# Create backup before making changes
python scripts/cache_control.py backup

# List available backups
python scripts/cache_control.py list-backups
```

### **2. Basic Execution:**
```bash
# Run the complete optimization system
python -m src.core.main

# Run with fresh start (recreates everything)
python scripts/cache_control.py fresh-start
python -m src.core.main

# Run with incremental mode (uses existing cache)
python scripts/cache_control.py incremental
python -m src.core.main
```

### **3. Database Operations:**
```bash
# Test database functionality
python scripts/test_database_simple.py

# Test enhanced database with complete data
python scripts/test_enhanced_database.py

# Validate cache integrity
python scripts/test_cache_validation.py
```

### **4. Environment Setup:**
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp config/development.env .env
# Edit .env with your credentials

# Load specific cache mode
source config/fresh_start.env    # For fresh start mode
source config/incremental.env     # For incremental mode
```

### **5. Production Deployment:**
```bash
# Set production environment
export ENVIRONMENT=production

# Set incremental mode for production
python scripts/cache_control.py incremental

# Run with production settings
python -m src.core.main

# Create daily backup
python scripts/cache_control.py backup
```

### **6. Windows Users:**
```batch
# Use Windows batch script for cache control
cache_control.bat status
cache_control.bat fresh
cache_control.bat incremental
cache_control.bat backup
cache_control.bat list
```

---

## **⌨️ Command Reference**

### **Cache Management Commands:**

#### **1. Cache Control Script (`scripts/cache_control.py`)**
```bash
# Show current cache status
python scripts/cache_control.py status

# Set fresh start mode (recreate everything)
python scripts/cache_control.py fresh-start

# Set incremental mode (use existing cache)
python scripts/cache_control.py incremental

# Create manual backup
python scripts/cache_control.py backup

# List available backups
python scripts/cache_control.py list-backups

# Restore from backup
python scripts/cache_control.py restore --backup-name backup_20250903_182500

# Clean up old backups (keep 5 most recent)
python scripts/cache_control.py cleanup --keep-count 5

# Set specific cache mode
python scripts/cache_control.py set-mode --mode fresh-start
python scripts/cache_control.py set-mode --mode incremental
```

#### **2. Windows Batch Script (`cache_control.bat`)**
```batch
# Show cache status
cache_control.bat status

# Set fresh start mode
cache_control.bat fresh

# Set incremental mode
cache_control.bat incremental

# Create backup
cache_control.bat backup

# List backups
cache_control.bat list

# Restore from backup
cache_control.bat restore backup_name

# Clean up backups
cache_control.bat cleanup
```

### **Main System Commands:**

#### **1. Run Complete System:**
```bash
# Run with current configuration
python -m src.core.main

# Run with fresh start mode
python scripts/cache_control.py fresh-start
python -m src.core.main

# Run with incremental mode
python scripts/cache_control.py incremental
python -m src.core.main
```

#### **2. Database Operations:**
```bash
# Test database functionality
python scripts/test_database_simple.py

# Test enhanced database with complete data
python scripts/test_enhanced_database.py

# Export database to CSV files
python -c "
from src.data.database_manager import DatabaseManager
db = DatabaseManager()
db.export_to_csv('database_exports')
"
```

#### **3. Testing and Validation:**
```bash
# Test cache validation
python scripts/test_cache_validation.py

# Create test recipes
python scripts/create_test_recipes.py

# Test comprehensive plots
python tests/test_comprehensive_plots.py
```

### **Environment Management:**

#### **1. Configuration Files:**
```bash
# Load fresh start configuration
source config/fresh_start.env

# Load incremental configuration
source config/incremental.env

# Set environment variables manually
export CACHE_FRESH_START=true
export CACHE_INCREMENTAL=false
export DATABASE_FRESH_START=true
```

#### **2. Environment Setup:**
```bash
# Install dependencies
pip install -r requirements.txt

# Set up development environment
cp config/development.env .env
# Edit .env with your credentials

# Set up production environment
cp config/production.env .env
# Edit .env with production credentials
```

### **Development Workflow:**

#### **1. Fresh Development Session:**
```bash
# 1. Set fresh start mode
python scripts/cache_control.py fresh-start

# 2. Run system (creates everything from scratch)
python -m src.core.main

# 3. Check results
python scripts/cache_control.py status
```

#### **2. Incremental Development:**
```bash
# 1. Set incremental mode
python scripts/cache_control.py incremental

# 2. Run system (uses existing cache)
python -m src.core.main

# 3. Check what changed
python scripts/cache_control.py status
```

#### **3. Backup and Restore:**
```bash
# 1. Create backup before changes
python scripts/cache_control.py backup

# 2. Make changes and test
python -m src.core.main

# 3. If something goes wrong, restore
python scripts/cache_control.py list-backups
python scripts/cache_control.py restore --backup-name manual_backup_20250903_183000
```

### **Production Deployment:**

#### **1. Daily Production Run:**
```bash
# Set incremental mode for production
python scripts/cache_control.py incremental

# Run daily optimization
python -m src.core.main

# Check system status
python scripts/cache_control.py status
```

#### **2. Production Backup Strategy:**
```bash
# Create daily backup
python scripts/cache_control.py backup

# Clean up old backups (keep last 7 days)
python scripts/cache_control.py cleanup --keep-count 7
```

#### **3. Cron Job Setup:**
```bash
# Add to crontab for daily execution
0 6 * * * cd /path/to/pareto && python scripts/cache_control.py incremental && python -m src.core.main
```

### **Troubleshooting Commands:**

#### **1. Cache Issues:**
```bash
# Check cache integrity
python scripts/cache_control.py status

# Validate cache structure
python scripts/test_cache_validation.py

# Force fresh start if cache is corrupted
python scripts/cache_control.py fresh-start
![1757355403783](image/SYSTEM_DOCUMENTATION/1757355403783.png)
```

#### **2. Database Issues:**
```bash
# Check database status
python scripts/test_enhanced_database.py

# Export database for inspection
python -c "
from src.data.database_manager import DatabaseManager
db = DatabaseManager()
print(db.get_database_info())
"
```

#### **3. Plot Generation Issues:**
```bash
# Test plot generation
python tests/test_comprehensive_plots.py

# Check plot files exist
ls dt-cache/iterations/iteration_1/plots/
```

---

## **🔌 API Integration**

### **Database Exports:**
The system generates CSV exports for nanoHUB API integration:

```
database_exports/
├── historical_data.csv          # Complete dataset (107 points)
├── iterations.csv               # Iteration metadata
├── highlight_lots.csv           # Highlighted lots per iteration
├── pareto_fronts.csv            # Pareto front data
├── model_predictions.csv        # ML predictions
├── training_data_snapshots.csv  # Training data snapshots
├── processing_logs.csv          # Processing history
├── system_metadata.csv          # System configuration
└── data_versioning.csv          # Data change tracking
```

### **API Endpoints (Future):**
```python
# Example nanoHUB API endpoints
GET /api/v1/iterations           # List all iterations
GET /api/v1/iterations/{id}      # Get specific iteration
GET /api/v1/pareto-fronts        # Get Pareto front data
GET /api/v1/historical-data      # Get complete dataset
GET /api/v1/validation-status     # Get cache validation status
```

### **Cron Job Integration:**
```bash
# Daily execution for nanoHUB
0 6 * * * cd /path/to/pareto && python -m src.core.main
```

---

## **🔮 Future Enhancements**

### **Planned Features:**
1. **Azure Integration**: Automated dataset ingestion from Azure Storage
2. **Advanced Model Selection**: Iteration 5+ model configuration
3. **Real-time Monitoring**: Live performance dashboards
4. **Advanced Validation**: Machine learning-based anomaly detection
5. **API Development**: RESTful API endpoints for nanoHUB
6. **Performance Optimization**: Parallel processing and caching
7. **Advanced Analytics**: Statistical analysis and reporting
8. **User Interface**: Web-based dashboard for monitoring

### **Scalability Considerations:**
- **Database Optimization**: Indexing and query optimization
- **Caching Strategy**: Multi-level caching for performance
- **Load Balancing**: Distributed processing capabilities
- **Monitoring**: Comprehensive system monitoring and alerting

---

## **📞 Support and Maintenance**

### **Troubleshooting:**
1. **Cache Validation**: Run `python scripts/test_cache_validation.py`
2. **Database Issues**: Check `python scripts/test_database_simple.py`
3. **Configuration**: Verify environment variables in `.env`
4. **Dependencies**: Ensure all packages in `requirements.txt` are installed

### **Maintenance Tasks:**
- **Daily**: Run cache validation and processing
- **Weekly**: Review processing logs and performance metrics
- **Monthly**: Database optimization and cleanup
- **Quarterly**: System updates and configuration review

---

## **📄 Version History**

### **v1.3.0 (Current)**
- Enhanced database schema with complete historical data
- Cache validation system for data integrity
- Production-grade configuration management
- Comprehensive documentation
- nanoHUB integration preparation

### **v1.2.0**
- SQLite database integration
- Enhanced plotting with dynamic y-axis limits
- Iteration-specific model selection
- Training data debugging features

### **v1.1.0**
- SharePoint integration
- LOOCV implementation
- Comprehensive plotting system
- File cache organization

### **v1.0.0**
- Initial Pareto optimization system
- Basic ML model integration
- File-based caching
- Core optimization algorithms

---

## **Pareto Recipes Excel File Structure Documentation**

### **Overview**
The Pareto Recipes Excel file serves as the central repository for all experimental recipes, predictions, and workflow management. This document provides comprehensive details about every aspect of the Excel file structure, column mappings, data types, and workflow logic.

### **File Structure (18 Columns Total)**

#### **Feature Columns (Input Parameters)**
These columns contain the experimental input parameters that define each recipe:

- **`O2_flow`** (float)
  - **Range:** 10.0 - 90.0
  - **Description:** Oxygen flow rate in standard units
  - **Code Mapping:** `Etch_AvgO2Flow`

- **`cf4_flow`** (float)
  - **Range:** 10.0 - 90.0
  - **Description:** CF4 flow rate in standard units
  - **Code Mapping:** `Etch_Avgcf4Flow`

- **`Rf1_Pow`** (int)
  - **Range:** 0.0 - 100.0
  - **Description:** RF Power 1 setting
  - **Code Mapping:** `Etch_Avg_Rf1_Pow`

- **`Rf2_Pow`** (int)
  - **Range:** 50.0 - 700.0
  - **Description:** RF Power 2 setting (must be > Rf1_Pow)
  - **Code Mapping:** `Etch_Avg_Rf2_Pow`
  - **Constraint:** Enforced in code - only recipes with Rf2 > Rf1 are proposed

- **`Pressure`** (float)
  - **Range:** 1.0 - 100.0
  - **Description:** Chamber pressure setting
  - **Code Mapping:** `Etch_AvgPres`

#### **Fixed Parameters**
These columns contain constant values that don't vary between recipes:

- **`Chamber_temp`** (int)
  - **Value:** Always 50
  - **Description:** Chamber temperature setting

- **`Electrode_temp`** (int)
  - **Value:** Always 15
  - **Description:** Electrode temperature setting

- **`Etch_time`** (int)
  - **Value:** Always 5
  - **Description:** Etching time duration

#### **Prediction Columns**
These columns contain ML model predictions and their associated uncertainties:

- **`Pred_avg_etch_rate`** (float)
  - **Description:** Predicted average etch rate from ML models
  - **Code Mapping:** `EXCEL_PRED_RATE_COL`

- **`Pred_Range`** (float)
  - **Description:** Predicted range from ML models
  - **Code Mapping:** `EXCEL_PRED_RANGE_COL`

- **`Etch_rate_uncertainty`** (float)
  - **Description:** Uncertainty in etch rate prediction (from GPR models)
  - **Code Mapping:** `EXCEL_RATE_UNCERTAINTY_COL`

- **`Range_uncertainty`** (float)
  - **Description:** Uncertainty in range prediction (from GPR models)
  - **Code Mapping:** `EXCEL_RANGE_UNCERTAINTY_COL`

#### **Status & Tracking Columns**
These columns manage the workflow and experimental status:

- **`Status`** (string)
  - **Values:** "pending", "completed"
  - **Description:** Overall recipe status
  - **Workflow:** "pending" for new recipes, "completed" when experimental results are available
  - **Code Mapping:** `EXCEL_STATUS_COL`

- **`Date_Completed`** (datetime)
  - **Description:** Date when experiments were completed
  - **Workflow:** Empty for new recipes, filled when experiments complete
  - **Code Mapping:** `EXCEL_DATE_COL`

- **`Lotname`** (string)
  - **Description:** Experimental lot identifier
  - **Workflow:** Empty for new recipes, filled when experiments complete
  - **Code Mapping:** `EXCEL_LOT_COL`

- **`idrun`** (int)
  - **Description:** Experimental run identifier
  - **Workflow:** Empty for new recipes, filled when experiments complete
  - **Note:** Generated by user, not by the system

- **`Iteration_num`** (int) **[NEW COLUMN]**
  - **Description:** Iteration number for tracking (1, 2, 3, 4, 5, etc.)
  - **Workflow:** Each iteration contains exactly 3 recipes
  - **Purpose:** Enables iteration-based analysis and tracking

#### **Workflow Management Columns**
These columns manage the experimental workflow and approval process:

- **`Ingestion_status`** (string)
  - **Values:** "waiting", "approved", "not_approved", "recipe_rejected"
  - **Description:** Workflow status for data ingestion
  - **Workflow Logic:**
    - **"waiting":** Default for new recipes, waiting for manual approval
    - **"approved":** Code has green light to ingest data for this run
    - **"not_approved":** No corresponding actual data in full_dataset.csv, use comment instead
    - **"recipe_rejected":** Marked for deletion, will be removed before adding new recipes
  - **Code Mapping:** `EXCEL_INGEST_COL`

- **`Comment`** (string)
  - **Description:** Experimental notes and comments
  - **Workflow:** Empty for new recipes, used for experimental notes
  - **Special Use:** When `Ingestion_status = "not_approved"`, contains explanation instead of actual results

### **Data Flow and Workflow**

#### **New Recipe Addition Process**
When the system proposes new recipes for iteration N:

1. **Recipe Generation:**
   - 3 new recipes generated using EI + batch optimization
   - Feature values selected from GPR model predictions
   - Uncertainty values calculated from GPR models

2. **Excel Entry Creation:**
   ```python
   new_recipe = {
       # Feature columns (mapped to Excel column names)
       'O2_flow': recipe[FEATURES.index('Etch_AvgO2Flow')],
       'cf4_flow': recipe[FEATURES.index('Etch_Avgcf4Flow')],
       'Rf1_Pow': recipe[FEATURES.index('Etch_Avg_Rf1_Pow')],
       'Rf2_Pow': recipe[FEATURES.index('Etch_Avg_Rf2_Pow')],
       'Pressure': recipe[FEATURES.index('Etch_AvgPres')],
       
       # Fixed parameters
       'Chamber_temp': 50,
       'Electrode_temp': 15,
       'Etch_time': 5,
       
       # Prediction columns
       'Pred_avg_etch_rate': rate,
       'Pred_Range': range_nm / 5.0,  # Convert back to original units
       'Etch_rate_uncertainty': rate_unc,
       'Range_uncertainty': range_unc / 5.0,  # Convert back to original units
       
       # Status and tracking columns
       'Status': 'pending',
       'Date_Completed': '',  # Empty for new recipes
       'Lotname': '',  # Empty for new recipes
       'idrun': '',  # Empty for new recipes
       'Iteration_num': iteration_num,  # NEW: Iteration tracking
       
       # Workflow columns
       'Ingestion_status': 'waiting',  # Default for new recipes
       'Comment': ''  # Empty for new recipes
   }
   ```

3. **Recipe Rejection Handling:**
   - System checks for `Ingestion_status = "recipe_rejected"`
   - Deletes all rejected recipes before adding new ones
   - Prevents duplicate recipe accumulation

#### **Experimental Completion Process**
When experiments are completed:

1. **Data Ingestion:**
   - Actual results added to `full_dataset.csv`
   - `Status` updated to "completed"
   - `Date_Completed` filled with completion date
   - `Lotname` and `idrun` filled with experimental identifiers

2. **Status Updates:**
   - `Ingestion_status` updated to "approved" if data ingestion successful
   - `Ingestion_status` updated to "not_approved" if issues encountered
   - `Comment` field used for explanations when `Ingestion_status = "not_approved"`

### **Column Mapping Constants**
The system uses these constants to map between Excel columns and internal code:

```python
EXCEL_STATUS_COL = "Status"                    # completed/pending
EXCEL_DATE_COL = "Date_Completed"             # Completion date
EXCEL_LOT_COL = "Lotname"                     # Lot identifier
EXCEL_INGEST_COL = "Ingestion_status"         # approved/waiting/not_approved/recipe_rejected
EXCEL_PRED_RATE_COL = "Pred_avg_etch_rate"    # Predicted etch rate
EXCEL_PRED_RANGE_COL = "Pred_Range"           # Predicted range
EXCEL_RATE_UNCERTAINTY_COL = "Etch_rate_uncertainty"  # Rate uncertainty
EXCEL_RANGE_UNCERTAINTY_COL = "Range_uncertainty"     # Range uncertainty
```

### **Data Validation Rules**

1. **Feature Constraints:**
   - All feature values must be within `FEATURE_RANGES`
   - `Rf2_Pow > Rf1_Pow` constraint enforced
   - Feature values must be numeric

2. **Status Validation:**
   - `Status` must be "pending" or "completed"
   - `Ingestion_status` must be one of the four valid values
   - `Iteration_num` must be positive integer

3. **Workflow Validation:**
   - New recipes must have `Status = "pending"`
   - New recipes must have empty `Date_Completed`, `Lotname`, `idrun`
   - New recipes must have `Ingestion_status = "waiting"`

### **Iteration Tracking Logic**

The `Iteration_num` column enables precise iteration tracking:

- **Iteration 1:** Recipes 1-3 (first experimental batch)
- **Iteration 2:** Recipes 4-6 (second experimental batch)
- **Iteration 3:** Recipes 7-9 (third experimental batch)
- **Iteration 4:** Recipes 10-12 (fourth experimental batch)
- **Iteration 5:** Recipes 13-15 (fifth experimental batch - newly proposed)

Each iteration contains exactly 3 recipes, enabling:
- Iteration-based analysis
- Progress tracking
- Model performance evaluation per iteration
- Workflow management

### **Integration with ML Models**

The Excel file integrates with the ML model system:

1. **Model Selection by Iteration:**
   - Iterations 1-2: Random Forest for both rate and range
   - Iterations 3-4: Extra Trees for rate, Random Forest for range
   - Iteration 5+: GPR with hyperparameter optimization for both

2. **Uncertainty Integration:**
   - GPR models provide uncertainty estimates
   - Uncertainty values stored in `Etch_rate_uncertainty` and `Range_uncertainty`
   - Used for error bars in plots and decision making

3. **Prediction Updates:**
   - Predictions updated when new models are trained
   - Uncertainty values recalculated with each iteration
   - Historical predictions preserved for analysis

### **File Management**

- **Primary File:** SharePoint Excel file (production)
- **Local Copy:** `data/test_pareto_recipes.xlsx` (development/testing)
- **Backup:** Snapshot files in `dt-cache/snapshots/`
- **Version Control:** Hash-based change tracking

### **Error Handling**

1. **Timezone Issues:**
   - Excel writing handles timezone-aware datetime objects
   - Automatic conversion to timezone-unaware format

2. **Missing Data:**
   - Graceful handling of empty fields
   - Default values for new recipes
   - Validation before Excel writing

3. **Duplicate Prevention:**
   - Checks for existing recipes with same predicted values
   - Prevents duplicate addition
   - Handles recipe rejection workflow

---

*This documentation is maintained as part of the Pareto Optimization System. For questions or issues, refer to the troubleshooting section or contact the development team.*
