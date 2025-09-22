# Iterative Training System Fixes

## Overview
This document summarizes the fixes implemented to make the training system in `pareto.py` truly iterative, as requested by the user.

## What Was Fixed

### 1. **Training Data Progression Logic**
- **Before**: The system was training on all data at once, regardless of iteration
- **After**: Each iteration now trains on data up to the completion date of its first row in the Excel file

### 2. **New Functions Added**

#### `_get_iteration_training_cutoff_date(recipes_df, iteration_num)`
- Calculates the training cutoff date for each iteration
- For iteration N, returns the completion date of the first row of iteration N
- Handles Excel date format conversion (MM/DD/YYYY to datetime)

#### `_get_training_data_for_iteration(df, recipes_df, iteration_num)`
- Returns training data for a specific iteration
- Filters `full_dataset.csv` to include only data with `run_date` < cutoff date
- Ensures experimental points from the current iteration are NOT included in training

#### `_get_completed_experimental_data_for_iteration(df, recipes_df, iteration_num)`
- Retrieves completed experimental data for a specific iteration
- Finds lots in `full_dataset.csv` that match completed recipes from Excel

#### `_get_training_data_for_main_proposals(df, recipes_df)`
- Determines training data for the main recipe proposals
- Uses data that would be available for the next iteration

### 3. **Updated Functions**

#### `_calculate_uncertainties_for_iteration()`
- Now uses proper iterative training data instead of hardcoded idruns
- Supports any number of iterations (not just 1 and 2)

#### `_run_loocv_iteration_specific()`
- Now uses proper iterative training data
- Removed redundant cumulative data progression logic
- Cleaner model selection logging

#### `_propose_next_iteration_recipes()`
- Now uses iterative training data for model retraining
- Pareto front calculation uses training data for consistency
- Better error handling and logging

#### `_generate_fresh_proposals_for_iteration()`
- Updated to support iterative training when recipes_df is available
- Falls back to full dataset when no recipes available

### 4. **Main Function Updates**
- Main model training now uses iterative training data
- Uncertainty calculation supports all iterations
- Better integration with the iteration system

## How It Works Now

### **Iteration 1**
- Training data: All data with `run_date` < completion date of first row in Excel
- Experimental points: First 3 recipes (highlighted but not trained on)

### **Iteration 2**
- Training data: All data with `run_date` < completion date of first row of iteration 2
- This includes historical data + completed experimental points from iteration 1

### **Iteration 3**
- Training data: All data with `run_date` < completion date of first row of iteration 3
- This includes historical data + completed experimental points from iterations 1 and 2

### **And so on...**

## Key Benefits

1. **Proper Data Isolation**: Each iteration trains on data that was actually available at that time
2. **Progressive Learning**: Models improve as more experimental data becomes available
3. **Consistent Training**: All functions (LOOCV, uncertainty, proposals) use the same training data
4. **Date-Based Logic**: Training cutoff is based on actual completion dates from Excel
5. **Scalable**: Supports any number of iterations, not just hardcoded limits

## Testing

A test script `test_iterative_training.py` has been created to verify:
- Cutoff date calculation
- Training data retrieval
- Experimental data retrieval
- Main proposals training data

## Usage

The system now automatically:
1. Reads completion dates from the Excel file
2. Calculates appropriate training data for each iteration
3. Uses iterative training for all model training operations
4. Maintains consistency across LOOCV, uncertainty calculation, and recipe generation

## Notes

- Excel dates are expected in MM/DD/YYYY format
- `full_dataset.csv` dates are expected in YYYY-MM-DD format
- The system automatically converts between formats as needed
- Training data includes all data up to (but not including) the cutoff date
- Experimental points are matched by lotname between Excel and full_dataset.csv
