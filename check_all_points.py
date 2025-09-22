#!/usr/bin/env python3
import sys
sys.path.append('.')

import pandas as pd

def check_all_points():
    print("=== CHECKING DATA POINTS BY DATE ===")
    
    # Load dataset
    dataset_df = pd.read_csv('full_dataset.csv')
    dataset_df['run_date'] = pd.to_datetime(dataset_df['run_date'])
    
    # Sort by date to see the progression
    dataset_sorted = dataset_df.sort_values('run_date')
    print('Last 10 data points by date:')
    for idx, row in dataset_sorted.tail(10).iterrows():
        print(f'  {row["LOTNAME"]} - {row["run_date"]}')
    
    print('\n=== CHECKING FOR ANY OTHER POINTS ON 2025-09-09 ===')
    points_909 = dataset_df[dataset_df['run_date'].dt.date == pd.to_datetime('2025-09-09').date()]
    print(f'Total points on 2025-09-09: {len(points_909)}')
    for idx, row in points_909.iterrows():
        print(f'  {row["LOTNAME"]} - {row["run_date"]}')
    
    print('\n=== CHECKING FOR POINTS AFTER 2025-09-09 ===')
    points_after = dataset_df[dataset_df['run_date'] > pd.to_datetime('2025-09-09')]
    print(f'Points after 2025-09-09: {len(points_after)}')
    for idx, row in points_after.iterrows():
        print(f'  {row["LOTNAME"]} - {row["run_date"]}')
    
    print('\n=== CHECKING TOTAL DATASET SIZE ===')
    print(f'Total dataset: {len(dataset_df)} points')
    
    print('\n=== EXPECTED TRAINING COUNTS ===')
    expected_counts = {1: 92, 2: 96, 3: 100, 4: 103, 5: 107, 6: 112}
    for iter_num, expected in expected_counts.items():
        print(f'Iteration {iter_num}: {expected} points')

if __name__ == "__main__":
    check_all_points()
