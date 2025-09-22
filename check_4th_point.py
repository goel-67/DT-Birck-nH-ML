#!/usr/bin/env python3
import sys
sys.path.append('.')

import pandas as pd
from src.core.config import DATASET_CSV

def check_4th_point():
    print("=== ALL 4 DATA POINTS ON 2025-09-09 ===")
    
    dataset_df = pd.read_csv(DATASET_CSV)
    dataset_df['run_date'] = pd.to_datetime(dataset_df['run_date'])
    
    cutoff_date = pd.to_datetime('2025-09-09')
    matching_data = dataset_df[dataset_df['run_date'].dt.date == cutoff_date.date()]
    
    for idx, row in matching_data.iterrows():
        print(f"Row {idx}: {row['LOTNAME']} - {row['run_date']}")
    
    print("\n=== ITERATION 5 RECIPES FROM EXCEL ===")
    iter5_lotnames = ['etch09/09/2025_641W', 'etch09/09/2025_108W', 'etch09/09/2025_82W']
    print(f"Iteration 5 lot names: {iter5_lotnames}")
    
    print("\n=== CHECKING WHICH POINT IS NOT AN ITERATION 5 RECIPE ===")
    for idx, row in matching_data.iterrows():
        if row['LOTNAME'] not in iter5_lotnames:
            print(f"NOT an iteration 5 recipe: Row {idx}: {row['LOTNAME']} - {row['run_date']}")
        else:
            print(f"IS an iteration 5 recipe: Row {idx}: {row['LOTNAME']} - {row['run_date']}")

if __name__ == "__main__":
    check_4th_point()
