#!/usr/bin/env python3
"""
Test script to verify the iterative training system in pareto.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Import the functions from pareto.py
from pareto_new import (
    _get_iteration_training_cutoff_date,
    _get_training_data_for_iteration,
    _get_completed_experimental_data_for_iteration,
    _get_training_data_for_main_proposals
)

def create_test_data():
    """Create test data to simulate the real scenario"""
    
    # Create test full_dataset.csv
    np.random.seed(42)
    n_points = 100
    
    # Generate dates from 2024-01-01 to 2025-01-01
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_points)]
    
    # Create test data
    test_data = {
        'LOTNAME': [f'TEST_LOT_{i:03d}' for i in range(n_points)],
        'FIMAP_FILE': [f'test_fimap_{i:03d}.fimap' for i in range(n_points)],
        'run_date': dates,
        'AvgEtchRate': np.random.uniform(30, 120, n_points),
        'RangeEtchRate': np.random.uniform(0.5, 3.0, n_points),
        'Etch_AvgO2Flow': np.random.uniform(10, 90, n_points),
        'Etch_Avgcf4Flow': np.random.uniform(10, 90, n_points),
        'Etch_Avg_Rf1_Pow': np.random.uniform(0, 100, n_points),
        'Etch_Avg_Rf2_Pow': np.random.uniform(50, 700, n_points),
        'Etch_AvgPres': np.random.uniform(1, 100, n_points)
    }
    
    df = pd.DataFrame(test_data)
    df.to_csv('test_full_dataset.csv', index=False)
    print(f"Created test full_dataset.csv with {len(df)} points")
    
    # Create test Excel recipes data
    recipes_data = {
        'O2_flow': [45.0, 50.0, 55.0, 60.0, 65.0, 70.0],
        'cf4_flow': [30.0, 35.0, 40.0, 45.0, 50.0, 55.0],
        'Rf1_Pow': [20.0, 25.0, 30.0, 35.0, 40.0, 45.0],
        'Rf2_Pow': [200.0, 250.0, 300.0, 350.0, 400.0, 450.0],
        'Pressure': [10.0, 15.0, 20.0, 25.0, 30.0, 35.0],
        'Pred_avg_etch_rate': [80.0, 85.0, 90.0, 95.0, 100.0, 105.0],
        'Pred_Range': [1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
        'Status': ['completed', 'completed', 'completed', 'pending', 'pending', 'pending'],
        'Date_Completed': ['7/22/2025', '7/23/2025', '7/24/2025', '', '', ''],
        'Lotname': ['TEST_LOT_101', 'TEST_LOT_102', 'TEST_LOT_103', '', '', ''],
        'Ingestion_status': ['approved', 'approved', 'approved', 'waiting', 'waiting', 'waiting']
    }
    
    recipes_df = pd.DataFrame(recipes_data)
    print(f"Created test recipes data with {len(recipes_df)} rows")
    print("Iteration 1: rows 0-2 (completed)")
    print("Iteration 2: rows 3-5 (pending)")
    
    return df, recipes_df

def test_iterative_training():
    """Test the iterative training functions"""
    
    print("\n" + "="*60)
    print("TESTING ITERATIVE TRAINING SYSTEM")
    print("="*60)
    
    # Create test data
    df, recipes_df = create_test_data()
    
    # Test 1: Get cutoff dates for different iterations
    print("\n1. Testing cutoff date calculation:")
    for iteration in [1, 2, 3]:
        cutoff_date = _get_iteration_training_cutoff_date(recipes_df, iteration)
        if cutoff_date:
            print(f"   Iteration {iteration}: cutoff = {cutoff_date.strftime('%Y-%m-%d')}")
        else:
            print(f"   Iteration {iteration}: no cutoff date available")
    
    # Test 2: Get training data for different iterations
    print("\n2. Testing training data retrieval:")
    for iteration in [1, 2, 3]:
        training_data = _get_training_data_for_iteration(df, recipes_df, iteration)
        if not training_data.empty:
            print(f"   Iteration {iteration}: {len(training_data)} training points")
            print(f"   Date range: {training_data['run_date'].min().strftime('%Y-%m-%d')} to {training_data['run_date'].max().strftime('%Y-%m-%d')}")
        else:
            print(f"   Iteration {iteration}: no training data available")
    
    # Test 3: Get experimental data for iterations
    print("\n3. Testing experimental data retrieval:")
    for iteration in [1, 2]:
        exp_data = _get_completed_experimental_data_for_iteration(df, recipes_df, iteration)
        if not exp_data.empty:
            print(f"   Iteration {iteration}: {len(exp_data)} experimental points")
            print(f"   Lots: {', '.join(exp_data['LOTNAME'].tolist())}")
        else:
            print(f"   Iteration {iteration}: no experimental data available")
    
    # Test 4: Get training data for main proposals
    print("\n4. Testing main proposals training data:")
    main_training = _get_training_data_for_main_proposals(df, recipes_df)
    if not main_training.empty:
        print(f"   Main proposals: {len(main_training)} training points")
        print(f"   Date range: {main_training['run_date'].min().strftime('%Y-%m-%d')} to {main_training['run_date'].max().strftime('%Y-%m-%d')}")
    else:
        print("   Main proposals: no training data available")
    
    # Clean up test files
    if os.path.exists('test_full_dataset.csv'):
        os.remove('test_full_dataset.csv')
        print("\nCleaned up test files")

if __name__ == "__main__":
    test_iterative_training()
