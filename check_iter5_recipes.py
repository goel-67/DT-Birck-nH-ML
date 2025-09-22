#!/usr/bin/env python3
import sys
sys.path.append('.')

from src.data.data_manager import DataManager
import pandas as pd

def check_iter5_recipes():
    print("=== ITERATION 5 RECIPES ===")
    
    dm = DataManager()
    recipes_df = dm.read_recipes_excel()
    
    iter5_recipes = recipes_df[recipes_df['Iteration_num'] == 5]
    print("Iteration 5 recipes:")
    for idx, row in iter5_recipes.iterrows():
        print(f"  {row['Lotname']} - {row['Date_Completed']}")
    
    print("\n=== CHECKING IF ANY RECIPE IS NOT ON 2025-09-09 ===")
    for idx, row in iter5_recipes.iterrows():
        if '2025-09-09' not in str(row['Date_Completed']):
            print(f"  Found recipe not on 2025-09-09: {row['Lotname']} - {row['Date_Completed']}")
    
    print("\n=== CHECKING DATASET FOR ITERATION 5 RECIPES ===")
    from src.core.config import DATASET_CSV
    dataset_df = pd.read_csv(DATASET_CSV)
    
    # Check if any of the iteration 5 recipes are in the dataset with different dates
    iter5_lotnames = iter5_recipes['Lotname'].tolist()
    print(f"Iteration 5 lot names: {iter5_lotnames}")
    
    for lotname in iter5_lotnames:
        matching_rows = dataset_df[dataset_df['LOTNAME'] == lotname]
        if not matching_rows.empty:
            for idx, row in matching_rows.iterrows():
                print(f"  Dataset: {row['LOTNAME']} - {row['run_date']}")
        else:
            print(f"  Not found in dataset: {lotname}")

if __name__ == "__main__":
    check_iter5_recipes()
