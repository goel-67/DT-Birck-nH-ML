#!/usr/bin/env python3
import sys
sys.path.append('.')

from src.data.data_manager import DataManager

def check_september_recipes():
    print("=== CHECKING etch09/02/2025_400W ===")
    
    dm = DataManager()
    recipes_df = dm.read_recipes_excel()
    
    matching_recipe = recipes_df[recipes_df['Lotname'] == 'etch09/02/2025_400W']
    if not matching_recipe.empty:
        for idx, row in matching_recipe.iterrows():
            print(f'Found: {row["Lotname"]} - Iteration {row["Iteration_num"]} - {row["Date_Completed"]}')
    else:
        print('Not found in recipes Excel file')
    
    print('\n=== CHECKING ALL RECIPES AROUND SEPTEMBER ===')
    september_recipes = recipes_df[recipes_df['Date_Completed'].str.contains('2025-09', na=False)]
    for idx, row in september_recipes.iterrows():
        print(f'{row["Lotname"]} - Iteration {row["Iteration_num"]} - {row["Date_Completed"]}')
    
    print('\n=== CHECKING ALL RECIPES AROUND SEPTEMBER 2 ===')
    sept2_recipes = recipes_df[recipes_df['Date_Completed'].str.contains('2025-09-02', na=False)]
    for idx, row in sept2_recipes.iterrows():
        print(f'{row["Lotname"]} - Iteration {row["Iteration_num"]} - {row["Date_Completed"]}')

if __name__ == "__main__":
    check_september_recipes()
