#!/usr/bin/env python3
import sys
sys.path.append('.')

import pandas as pd
from src.data.data_manager import DataManager

def debug_date_cutoff():
    print("=== DEBUGGING DATE CUTOFF LOGIC ===")
    
    dm = DataManager()
    recipes_df = dm.read_recipes_excel()
    dataset_df = pd.read_csv('full_dataset.csv')
    dataset_df['run_date'] = pd.to_datetime(dataset_df['run_date'])
    
    cutoff_date = dm.get_iteration_training_cutoff_date(recipes_df, 5)
    print(f'Cutoff date: {cutoff_date}')
    
    # Check the 400W point specifically
    point_400w = dataset_df[dataset_df['LOTNAME'] == 'etch09/09/2025_400W']
    if not point_400w.empty:
        row = point_400w.iloc[0]
        print(f'etch09/09/2025_400W run_date: {row["run_date"]}')
        print(f'Is run_date < cutoff_date? {row["run_date"] < cutoff_date}')
        print(f'Is run_date == cutoff_date? {row["run_date"] == cutoff_date}')
    
    # Check all 4 points on 2025-09-09
    print('\n=== ALL 4 POINTS ON 2025-09-09 ===')
    cutoff_date_only = cutoff_date.date()
    points_909 = dataset_df[dataset_df['run_date'].dt.date == cutoff_date_only]
    for idx, row in points_909.iterrows():
        print(f'{row["LOTNAME"]}: run_date = {row["run_date"]}, < cutoff? {row["run_date"] < cutoff_date}')
    
    # Check iteration 5 lotnames
    print('\n=== ITERATION 5 LOTNAMES ===')
    iter5_lotnames = dm._get_iteration_lotnames(recipes_df, 5)
    print(f'Iteration 5 lotnames: {iter5_lotnames}')
    
    # Check which points are excluded by each filter
    print('\n=== FILTER ANALYSIS ===')
    exclude_iteration_mask = ~dataset_df["LOTNAME"].isin(iter5_lotnames)
    date_mask = dataset_df["run_date"] < cutoff_date
    combined_mask = exclude_iteration_mask & date_mask
    
    print(f'Points excluded by iteration filter: {sum(~exclude_iteration_mask)}')
    print(f'Points excluded by date filter: {sum(~date_mask)}')
    print(f'Points included by combined filter: {sum(combined_mask)}')
    
    # Show which specific points are excluded by each filter
    print('\n=== POINTS EXCLUDED BY ITERATION FILTER ===')
    excluded_by_iter = dataset_df[~exclude_iteration_mask]
    for idx, row in excluded_by_iter.iterrows():
        print(f'  {row["LOTNAME"]}')
    
    print('\n=== POINTS EXCLUDED BY DATE FILTER ===')
    excluded_by_date = dataset_df[~date_mask]
    for idx, row in excluded_by_date.iterrows():
        print(f'  {row["LOTNAME"]} - {row["run_date"]}')

if __name__ == "__main__":
    debug_date_cutoff()
