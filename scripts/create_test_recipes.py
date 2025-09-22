import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Create sample recipe data
np.random.seed(42)

# Sample feature values within the defined ranges
feature_ranges = {
    "Etch_AvgO2Flow": (10.0, 90.0),
    "Etch_Avgcf4Flow": (10.0, 90.0),
    "Etch_Avg_Rf1_Pow": (0.0, 100.0),
    "Etch_Avg_Rf2_Pow": (50.0, 700.0),
    "Etch_AvgPres": (1.0, 100.0)
}

# Real lot names from the dataset
real_lot_names = [
    "etch8/22/2024", "etch8/27/2024", "etch8/30/2024",  # Iteration 1
    "etch9/3/2024", "etch9/6/2024", "etch9/12/2024",    # Iteration 2  
    "etch9/13/2024", "etch9/18/2024", "etch9/20/2024"   # Iteration 3
]

# Create recipes for iteration 1 (completed)
iteration_1_recipes = []
for i in range(3):
    recipe = {}
    for feature, (min_val, max_val) in feature_ranges.items():
        recipe[feature] = np.random.uniform(min_val, max_val)
    
    # Add metadata
    recipe['Lotname'] = real_lot_names[i]
    recipe['Status'] = 'completed'
    recipe['Date_Completed'] = datetime.now() - timedelta(days=30-i)
    recipe['Ingestion_status'] = 'approved'
    recipe['Pred_avg_etch_rate'] = np.random.uniform(40, 100)
    recipe['Pred_Range'] = np.random.uniform(2, 6)
    recipe['Etch_rate_uncertainty'] = np.random.uniform(5, 15)
    recipe['Range_uncertainty'] = np.random.uniform(0.3, 0.8)
    
    iteration_1_recipes.append(recipe)

# Create recipes for iteration 2 (proposed)
iteration_2_recipes = []
for i in range(3):
    recipe = {}
    for feature, (min_val, max_val) in feature_ranges.items():
        recipe[feature] = np.random.uniform(min_val, max_val)
    
    # Add metadata
    recipe['Lotname'] = real_lot_names[i+3]
    recipe['Status'] = 'proposed'
    recipe['Date_Completed'] = None
    recipe['Ingestion_status'] = 'pending'
    recipe['Pred_avg_etch_rate'] = np.random.uniform(40, 100)
    recipe['Pred_Range'] = np.random.uniform(2, 6)
    recipe['Etch_rate_uncertainty'] = np.random.uniform(5, 15)
    recipe['Range_uncertainty'] = np.random.uniform(0.3, 0.8)
    
    iteration_2_recipes.append(recipe)

# Create recipes for iteration 3 (proposed)
iteration_3_recipes = []
for i in range(3):
    recipe = {}
    for feature, (min_val, max_val) in feature_ranges.items():
        recipe[feature] = np.random.uniform(min_val, max_val)
    
    # Add metadata
    recipe['Lotname'] = real_lot_names[i+6]
    recipe['Status'] = 'proposed'
    recipe['Date_Completed'] = None
    recipe['Ingestion_status'] = 'pending'
    recipe['Pred_avg_etch_rate'] = np.random.uniform(40, 100)
    recipe['Pred_Range'] = np.random.uniform(2, 6)
    recipe['Etch_rate_uncertainty'] = np.random.uniform(5, 15)
    recipe['Range_uncertainty'] = np.random.uniform(0.3, 0.8)
    
    iteration_3_recipes.append(recipe)

# Combine all recipes
all_recipes = iteration_1_recipes + iteration_2_recipes + iteration_3_recipes

# Create DataFrame
df = pd.DataFrame(all_recipes)

# Reorder columns to match expected format
column_order = [
    'Lotname', 'Status', 'Date_Completed', 'Ingestion_status',
    'Pred_avg_etch_rate', 'Pred_Range', 'Etch_rate_uncertainty', 'Range_uncertainty'
] + list(feature_ranges.keys())

df = df[column_order]

# Save to Excel
df.to_excel('data/test_pareto_recipes.xlsx', index=False)
print("Created test_pareto_recipes.xlsx with sample recipe data")
print(f"Total recipes: {len(df)}")
print(f"Iteration 1 (completed): {len(iteration_1_recipes)}")
print(f"Iteration 2 (proposed): {len(iteration_2_recipes)}")
print(f"Iteration 3 (proposed): {len(iteration_3_recipes)}")
