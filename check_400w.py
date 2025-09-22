#!/usr/bin/env python3
import sys
sys.path.append('.')

from src.data.data_manager import DataManager

def check_400w():
    print("=== CHECKING etch09/09/2025_400W ===")
    
    dm = DataManager()
    recipes_df = dm.read_recipes_excel()
    
    matching_recipe = recipes_df[recipes_df['Lotname'] == 'etch09/09/2025_400W']
    if not matching_recipe.empty:
        for idx, row in matching_recipe.iterrows():
            print(f"Found: {row['Lotname']} - Iteration {row['Iteration_num']} - {row['Date_Completed']}")
    else:
        print("Not found in recipes Excel file")
        
    print("\n=== ALL RECIPES ON 2025-09-09 ===")
    all_recipes_909 = recipes_df[recipes_df['Date_Completed'].str.contains('2025-09-09', na=False)]
    for idx, row in all_recipes_909.iterrows():
        print(f"{row['Lotname']} - Iteration {row['Iteration_num']} - {row['Date_Completed']}")

if __name__ == "__main__":
    check_400w()
