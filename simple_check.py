#!/usr/bin/env python3
import sys
sys.path.append('.')

from src.data.data_manager import DataManager

def simple_check():
    print("=== CHECKING ALL RECIPES ===")
    
    dm = DataManager()
    recipes_df = dm.read_recipes_excel()
    
    print("All recipes in Excel file:")
    for idx, row in recipes_df.iterrows():
        lotname = row['Lotname']
        iteration = row['Iteration_num']
        date = row['Date_Completed']
        print(f"  {lotname} - Iteration {iteration} - {date}")
    
    print(f"\nTotal recipes: {len(recipes_df)}")

if __name__ == "__main__":
    simple_check()
