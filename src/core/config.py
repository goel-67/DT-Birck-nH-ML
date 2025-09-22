"""
Configuration module for Pareto optimization system.
Extracts all configuration constants and settings from pareto.py.
"""

import os
import datetime as dt
from dotenv import load_dotenv

# Load environment variables
# Try to load from multiple possible locations
load_dotenv()  # Load from .env in root directory if it exists
load_dotenv("config/pareto_config.env")  # Load from config directory

# Environment-specific configuration
ENV = os.getenv("ENVIRONMENT", "development")
if ENV == "production":
    load_dotenv("config/production.env")
else:
    load_dotenv("config/development.env")

# ==================== VERSION ====================
CODE_VERSION = "v1.3.0"

# ==================== PATHS & DIRECTORIES ====================
DATASET_CSV = os.getenv("DT_DATASET_CSV", "data/full_dataset.csv")
ROOT_DIR = os.getenv("DT_CACHE_ROOT", "dt-cache")
CACHE_DIRECTORY = ROOT_DIR  # Alias for cache validation
ROLLING_DIR = os.path.join(ROOT_DIR, "rolling")
SNAPSHOTS_DIR = os.path.join(ROOT_DIR, "snapshots")
MANIFESTS_DIR = os.path.join(ROOT_DIR, "manifests")
ITERATIONS_DIR = os.path.join(ROOT_DIR, "iterations")
TODAY = dt.datetime.now().strftime("%Y-%m-%d")
SNAP_DIR = os.path.join(SNAPSHOTS_DIR, TODAY)
PLOTS_DIR = os.path.join(SNAP_DIR, "plots")

# ==================== SAMPLING PARAMETERS ====================
K = int(os.getenv("DT_TOP_K", "3"))
SAMPLING_METHOD = os.getenv("DT_SAMPLING", "sobol")
SAMPLING_SEED = int(os.getenv("DT_SEED", "42"))
ROUND_SOBOL_TO_POW2 = os.getenv("DT_SOBOL_POW2", "true").lower() in ("1", "true", "yes")
N_SAMPLES = int(os.getenv("DT_NSAMPLES", "200000"))

# ==================== PARETO OPTIMIZATION PARAMETERS ====================
ALPHA = float(os.getenv("DT_ALPHA", "1.0"))  # Weight for exploitation
BETA = float(os.getenv("DT_BETA", "0.25"))   # Weight for exploration
GAMMA = float(os.getenv("DT_GAMMA", "0.4"))  # Weight for diversity

# Thresholds for Pareto front improvement
RATE_IMPROVEMENT_THRESHOLD = float(os.getenv("DT_RATE_THRESHOLD", "0.05"))
RANGE_IMPROVEMENT_THRESHOLD = float(os.getenv("DT_RANGE_THRESHOLD", "0.05"))

# Model choice for first two iterations
MODEL_CHOICE_FIRST_TWO_ITERATIONS = os.getenv("DT_MODEL_CHOICE", "rf_both").strip().lower()

# GPR hyperparameter optimization toggle
GPR_HYPERPARAMETER_OPTIMIZATION = os.getenv("DT_GPR_HYPEROPT", "false").lower() in ("1", "true", "yes")

# Target rate constraints
TARGET_RATE_MIN = float(os.getenv("DT_RATE_MIN", "35"))
TARGET_RATE_MAX = float(os.getenv("DT_RATE_MAX", "110"))

# Iteration configuration
POINTS_PER_ITERATION = int(os.getenv("DT_POINTS_PER_ITERATION", "3"))

# Training data cutoff method
TRAINING_CUTOFF_METHOD = os.getenv("DT_TRAINING_CUTOFF_METHOD", "first_recipe_current")
# Options: "first_recipe_current" or "last_recipe_previous"

# Ingestion mode
INGEST_MODE = os.getenv("DT_INGEST_MODE", "auto").strip().lower()

# Cache management toggle
CACHE_MANAGEMENT_MODE = os.getenv("CACHE_MANAGEMENT_MODE", "full_rebuild").lower()
# Options: "incremental" or "full_rebuild" (default)
# Set CACHE_MANAGEMENT_MODE=incremental to build upon existing cache

# ==================== FEATURE RANGES ====================
FEATURE_RANGES = {
    "Etch_AvgO2Flow": (10.0, 90.0),
    "Etch_Avgcf4Flow": (10.0, 90.0),
    "Etch_Avg_Rf1_Pow": (0.0, 100.0),
    "Etch_Avg_Rf2_Pow": (50.0, 700.0),
    "Etch_AvgPres": (1.0, 100.0)
}
FEATURES = list(FEATURE_RANGES.keys())

# ==================== EXCEL COLUMN MAPPINGS ====================
EXCEL_STATUS_COL = os.getenv("DT_EXCEL_STATUS_COL", "Status")
EXCEL_DATE_COL = os.getenv("DT_EXCEL_DATE_COL", "Date_Completed")
EXCEL_LOT_COL = os.getenv("DT_EXCEL_LOT_COL", "Lotname")
EXCEL_INGEST_COL = os.getenv("DT_EXCEL_INGEST_COL", "Ingestion_status")
EXCEL_PRED_RATE_COL = os.getenv("DT_EXCEL_PRED_RATE_COL", "Pred_avg_etch_rate")
EXCEL_PRED_RANGE_COL = os.getenv("DT_EXCEL_PRED_RANGE_COL", "Pred_Range")
EXCEL_RATE_UNCERTAINTY_COL = os.getenv("DT_EXCEL_RATE_UNCERTAINTY_COL", "Etch_rate_uncertainty")
EXCEL_RANGE_UNCERTAINTY_COL = os.getenv("DT_EXCEL_RANGE_UNCERTAINTY_COL", "Range_uncertainty")

# ==================== PLOTTING CONSTANTS ====================
HIGHLIGHT_COLORS = ["purple", "orangered", "green"]
CIRCLED_NUMBERS = ["①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨", "⑩", "⑪", "⑫", "⑬", "⑭", "⑮", "⑯", "⑰", "⑱", "⑲", "⑳"]
SUBSCRIPT_NUMBERS = ["₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉"]

# ==================== HIGHLIGHT PRESETS ====================
HIGHLIGHT_LOTS = [
    "07_30_2025[_367W]",
    "08_05_2025[_384W]",
    "08_05_2025[_594W]"
]

# ==================== SHAREPOINT / GRAPH CONFIGURATION ====================
GRAPH_CLIENT_ID = os.getenv("GRAPH_CLIENT_ID", "").strip()
GRAPH_CLIENT_SECRET = os.getenv("GRAPH_CLIENT_SECRET", "").strip()
GRAPH_TENANT_ID = os.getenv("GRAPH_TENANT_ID", "").strip()
GRAPH_TENANT_NAME = os.getenv("GRAPH_TENANT_NAME", "purdue0").strip()
GRAPH_SITE_NAME = os.getenv("GRAPH_SITE_NAME", "Birck-nanoHUB-DT").strip()
RECIPES_FILE_PATH = os.getenv("RECIPES_FILE_PATH", "Experimental Data/Pareto Recipes.xlsx").strip()
LOCAL_RECIPES_XLSX = os.getenv("DT_RECIPES_XLSX", "").strip()

# ==================== OPTIONAL DEPENDENCIES ====================
# Check for optional dependencies
try:
    from scipy.stats import qmc
    _HAS_QMC = True
except Exception:
    _HAS_QMC = False

try:
    import requests, msal
    _HAS_MSAL = True
except Exception:
    _HAS_MSAL = False

# ==================== CACHE MANAGEMENT ====================
# Cache behavior control
CACHE_FRESH_START = os.getenv("CACHE_FRESH_START", "false").lower() == "true"  # Force recreate all cache
CACHE_INCREMENTAL = os.getenv("CACHE_INCREMENTAL", "true").lower() == "true"   # Use existing cache, add new data
CACHE_BACKUP_BEFORE_OVERWRITE = os.getenv("CACHE_BACKUP_BEFORE_OVERWRITE", "true").lower() == "true"  # Backup before fresh start

# Cache validation settings
CACHE_VALIDATE_ON_START = os.getenv("CACHE_VALIDATE_ON_START", "true").lower() == "true"  # Validate cache integrity
CACHE_AUTO_REPAIR = os.getenv("CACHE_AUTO_REPAIR", "false").lower() == "true"  # Auto-repair corrupted cache
CACHE_REGENERATE_PLOTS = os.getenv("CACHE_REGENERATE_PLOTS", "true").lower() == "true"  # Generate plots during cache regeneration

# Database management
DATABASE_FRESH_START = os.getenv("DATABASE_FRESH_START", "false").lower() == "true"  # Force recreate database
DATABASE_INCREMENTAL = os.getenv("DATABASE_INCREMENTAL", "true").lower() == "true"   # Use existing database, add new data
