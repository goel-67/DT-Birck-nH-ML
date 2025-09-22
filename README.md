# Pareto Optimization System

A comprehensive system for Pareto optimization of semiconductor etching processes using machine learning and iterative training.

## Project Structure

```
Pareto_backup/
├── src/                          # Source code
│   ├── core/                     # Core system components
│   │   ├── main.py              # Main entry point
│   │   └── config.py            # Configuration and settings
│   ├── data/                     # Data management
│   │   ├── data_manager.py      # Data loading and processing
│   │   ├── excel_manager.py     # Excel file handling
│   │   └── dataset_new.py       # Database connectivity
│   ├── ml/                       # Machine learning
│   │   ├── ml_models.py         # ML models and training
│   │   └── sampler.py           # Candidate sampling
│   ├── optimization/             # Optimization algorithms
│   │   └── pareto_optimizer.py  # Pareto front optimization
│   ├── visualization/            # Plotting and visualization
│   │   └── plotter.py           # All plotting functionality
│   ├── cli/                      # Command-line interface
│   │   ├── cli.py               # CLI interface
│   │   └── iteration_manager.py # Iteration management
│   └── utils/                    # Utility functions
├── config/                        # Configuration files
│   ├── requirements.txt          # Python dependencies
│   └── pareto_config.env         # Environment variables
├── data/                          # Data files
│   ├── full_dataset.csv          # Main dataset
│   ├── loocv_summary_results.csv # Cross-validation results
│   ├── rf_predictions_idrun_4822.csv # RF predictions
│   └── test_pareto_recipes.xlsx  # Test recipes
├── tests/                         # Test files
│   ├── test_modular.py           # Modular component tests
│   ├── test_iterative_training.py # Iterative training tests
│   └── test_comprehensive_plots.py # Plotting tests
├── docs/                          # Documentation
│   ├── ITERATIVE_TRAINING_FIXES.md # Training fixes
│   └── RF_FILTER_CHANGES.md      # RF filter changes
├── scripts/                       # Utility scripts
│   └── create_test_recipes.py     # Test recipe creation
├── notebooks/                     # Jupyter notebooks
│   ├── test_models.ipynb         # Model testing notebook
│   └── test.ipynb                # Additional tests
├── fimap/                         # Process data files
│   └── [107 .fimap files]        # Semiconductor process data
├── dt-cache/                      # Cache and output
│   ├── iterations/               # Iteration-specific data
│   ├── manifests/                # System manifests
│   ├── rolling/                  # Rolling data
│   └── snapshots/                # Date-based snapshots
├── pareto.py                      # Legacy monolithic implementation
├── pareto_new.py                  # Alternative implementation
└── README.md                      # This file
```

## Key Components

### Core System (`src/core/`)
- **main.py**: Main entry point that orchestrates the entire Pareto optimization system
- **config.py**: Centralized configuration with environment variables and system settings

### Data Management (`src/data/`)
- **data_manager.py**: Handles data loading, processing, validation, and Excel integration
- **excel_manager.py**: Excel file handling and recipe management
- **dataset_new.py**: Database connectivity for data extraction from PostgreSQL

### Machine Learning (`src/ml/`)
- **ml_models.py**: Random Forest and Extra Trees models for etch rate and range prediction
- **sampler.py**: Sobol sequence and random sampling for candidate generation

### Optimization (`src/optimization/`)
- **pareto_optimizer.py**: Pareto front calculations, scoring, and proposal generation

### Visualization (`src/visualization/`)
- **plotter.py**: Comprehensive plotting including Pareto fronts, parity plots, and metrics

### CLI Interface (`src/cli/`)
- **cli.py**: Command-line interface for system interaction
- **iteration_manager.py**: Iteration data management and storage

## Installation

1. Install Python dependencies:
```bash
pip install -r config/requirements.txt
```

2. Set up environment variables:
```bash
# Copy and modify the environment file
cp config/pareto_config.env .env
```

## Usage

### Main Execution
```bash
python src/core/main.py
```

### CLI Interface
```bash
python src/cli/cli.py --help
```

## Data Files

- **full_dataset.csv**: Main semiconductor etching process dataset
- **fimap/**: Directory containing 107 .fimap files with process data
- **test_pareto_recipes.xlsx**: Test Excel file for recipe management

## Configuration

- **pareto_config.env**: Environment variables for system configuration
- **requirements.txt**: Python package dependencies

## Legacy Files

- **pareto.py**: Original monolithic implementation (255KB, 5445 lines)
- **pareto_new.py**: Alternative implementation (255KB, 5439 lines)

## Cache Structure

The `dt-cache/` directory contains:
- **iterations/**: Iteration-specific data and plots
- **manifests/**: System manifests and metadata
- **rolling/**: Rolling data and predictions
- **snapshots/**: Date-based snapshots with plots

## Testing

Run tests from the `tests/` directory:
```bash
python tests/test_modular.py
python tests/test_iterative_training.py
python tests/test_comprehensive_plots.py
```

## Documentation

See the `docs/` directory for detailed documentation on:
- Iterative training fixes
- Random Forest filter changes
- System architecture and design decisions
