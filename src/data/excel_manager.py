import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
import os
from ..core.config import *

class ExcelManager:
    def __init__(self, file_path: str = None):
        self.file_path = file_path or LOCAL_RECIPES_XLSX
        self.recipes = None
        
    def read_recipes(self) -> Optional[pd.DataFrame]:
        """Read recipes from Excel file"""
        try:
            if os.path.exists(self.file_path):
                self.recipes = pd.read_excel(self.file_path)
                return self.recipes
            else:
                print(f"[excel_manager] Excel file not found: {self.file_path}")
                return None
        except Exception as e:
            print(f"[excel_manager] Error reading Excel file: {e}")
            return None
    
    def get_iteration_status(self) -> Dict[int, Dict[str, Any]]:
        """Analyze Excel file to determine iteration status"""
        if self.recipes is None:
            return {}
            
        iteration_status = {}
        
        # Normalize status column
        if EXCEL_STATUS_COL in self.recipes.columns:
            self.recipes["Status_norm"] = self.recipes[EXCEL_STATUS_COL].astype(str).str.strip().str.lower()
        
        # Group by iteration number (extract from recipe names)
        for idx, row in self.recipes.iterrows():
            recipe_name = str(row.get("Recipe_Name", ""))
            
            # Extract iteration number from recipe name
            iteration_num = None
            if "iteration" in recipe_name.lower():
                try:
                    # Look for iteration number in recipe name
                    parts = recipe_name.lower().split("iteration")
                    if len(parts) > 1:
                        iteration_part = parts[1].strip()
                        iteration_num = int(iteration_part.split()[0])
                except:
                    pass
            
            # CRITICAL FIX: Use Iteration_num column if it exists, otherwise infer from row order
            if iteration_num is None:
                # Check if Iteration_num column exists in the row data
                if 'Iteration_num' in row and pd.notna(row['Iteration_num']):
                    iteration_num = int(row['Iteration_num'])
                else:
                    # Fallback: infer from row order (only for legacy data without Iteration_num column)
                    iteration_num = (idx // DT_POINTS_PER_ITERATION) + 1
            
            if iteration_num not in iteration_status:
                iteration_status[iteration_num] = {
                    "start_idx": idx,
                    "end_idx": idx,
                    "completed_count": 0,
                    "pending_count": 0,
                    "is_completed": False,
                    "recipes": []
                }
            
            # Update iteration info
            iteration_status[iteration_num]["end_idx"] = idx
            
            # Check status
            status = row.get("Status_norm", "pending")
            if status == "completed":
                iteration_status[iteration_num]["completed_count"] += 1
            else:
                iteration_status[iteration_num]["pending_count"] += 1
            
            # Add recipe to iteration - we'll collect all recipes for this iteration
            # and then store them as a DataFrame slice like pareto.py does
            pass  # We'll handle this after the loop
        
        # Now collect all recipes for each iteration as DataFrame slices (like pareto.py does)
        for iter_num in iteration_status:
            start_idx = iteration_status[iter_num]["start_idx"]
            end_idx = iteration_status[iter_num]["end_idx"]
            # Store the DataFrame slice like pareto.py does
            iteration_status[iter_num]["recipes"] = self.recipes.iloc[start_idx:end_idx]
        
        # Determine if iterations are completed
        for iter_num, status in iteration_status.items():
            status["is_completed"] = status["completed_count"] >= POINTS_PER_ITERATION
        
        return iteration_status
    
    def get_proposed_recipes_for_iteration(self, iteration_num: int) -> List[Dict[str, Any]]:
        """Get proposed recipes for a specific iteration"""
        if self.recipes is None:
            return []
            
        iteration_status = self.get_iteration_status()
        
        if iteration_num not in iteration_status:
            return []
            
        # Convert DataFrame slice to list of dictionaries (like pareto.py does)
        recipes_df = iteration_status[iteration_num]["recipes"]
        proposed_recipes = []
        
        for i, (_, recipe) in enumerate(recipes_df.iterrows()):
            recipe_data = {
                "O2_flow": recipe.get("O2_flow", 0.0),
                "cf4_flow": recipe.get("cf4_flow", 0.0),
                "Rf1_Pow": recipe.get("Rf1_Pow", 0.0),
                "Rf2_Pow": recipe.get("Rf2_Pow", 0.0),
                "Pressure": recipe.get("Pressure", 0.0),
                "predicted_rate": recipe.get(EXCEL_PRED_RATE_COL, 0.0),
                "predicted_range": recipe.get(EXCEL_PRED_RANGE_COL, 0.0),
                "rate_uncertainty": recipe.get(EXCEL_RATE_UNCERTAINTY_COL, 0.0),
                "range_uncertainty": recipe.get(EXCEL_RANGE_UNCERTAINTY_COL, 0.0),
                "status": recipe.get(EXCEL_STATUS_COL, ""),
                "lotname": recipe.get(EXCEL_LOT_COL, ""),
                "date_completed": recipe.get(EXCEL_DATE_COL, "")
            }
            proposed_recipes.append(recipe_data)
        
        return proposed_recipes
    
    def update_ingestion_status(self, lot_names: List[str]):
        """Update ingestion status for completed lots"""
        if self.recipes is None:
            return
            
        # This would require write access to the Excel file
        # For now, just print what would be updated
        print(f"[excel_manager] Would update ingestion status for lots: {lot_names}")
    
    def write_proposals(self, proposals: pd.DataFrame):
        """Write new proposals to Excel file"""
        # This would require write access to the Excel file
        # For now, just print what would be written
        print(f"[excel_manager] Would write {len(proposals)} proposals to Excel")
    
    def get_excel_path(self) -> str:
        """Get the Excel file path"""
        return self.file_path
    
    def write_recipes_to_excel(self, recipes_df: pd.DataFrame, excel_path: str) -> bool:
        """Write recipes DataFrame to Excel file"""
        try:
            print(f"[excel_manager] Writing {len(recipes_df)} recipes to Excel file: {excel_path}")
            
            # Create a copy to avoid modifying the original
            df_to_write = recipes_df.copy()
            
            # Fix timezone issues for datetime columns
            for col in df_to_write.columns:
                if df_to_write[col].dtype == 'datetime64[ns, UTC]':
                    # Convert timezone-aware datetime to timezone-unaware
                    df_to_write[col] = df_to_write[col].dt.tz_localize(None)
                elif df_to_write[col].dtype == 'object':
                    # Check if column contains datetime objects
                    if df_to_write[col].notna().any():
                        sample_val = df_to_write[col].dropna().iloc[0] if df_to_write[col].notna().any() else None
                        if isinstance(sample_val, pd.Timestamp) and sample_val.tz is not None:
                            # Convert timezone-aware timestamps to timezone-unaware
                            df_to_write[col] = df_to_write[col].apply(
                                lambda x: x.tz_localize(None) if isinstance(x, pd.Timestamp) and x.tz is not None else x
                            )
            
            # Write to Excel file
            df_to_write.to_excel(excel_path, index=False)
            
            # Update our internal recipes
            self.recipes = recipes_df
            
            print(f"[excel_manager] Successfully wrote recipes to Excel file")
            return True
            
        except Exception as e:
            print(f"[excel_manager] Error writing recipes to Excel file: {e}")
            return False
    
    def calculate_uncertainties_for_iteration(self, iteration_num: int, recipes_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate uncertainties for a specific iteration"""
        recipes_with_uncertainties = recipes_df.copy()
        
        # Get proposed recipes for this iteration
        proposed_recipes = self.get_proposed_recipes_for_iteration(iteration_num)
        
        if not proposed_recipes:
            return recipes_with_uncertainties
        
        # Calculate uncertainties based on iteration number
        for recipe in proposed_recipes:
            idx = recipe["idx"]
            
            # Simple uncertainty model: decreases with iteration number
            base_uncertainty_rate = 0.1  # 10% base uncertainty
            base_uncertainty_range = 0.15  # 15% base uncertainty
            
            # Uncertainty decreases with each iteration
            iteration_factor = max(0.1, 1.0 - (iteration_num - 1) * 0.2)
            
            rate_uncertainty = base_uncertainty_rate * iteration_factor
            range_uncertainty = base_uncertainty_range * iteration_factor
            
            # Update the recipes DataFrame
            if idx < len(recipes_with_uncertainties):
                recipes_with_uncertainties.loc[idx, EXCEL_RATE_UNCERTAINTY_COL] = rate_uncertainty
                recipes_with_uncertainties.loc[idx, EXCEL_RANGE_UNCERTAINTY_COL] = range_uncertainty
        
        return recipes_with_uncertainties

def _read_recipes_excel():
    """Read recipes Excel file - compatibility function for existing code"""
    excel_manager = ExcelManager()
    return excel_manager.read_recipes()

def _get_excel_iteration_status(recipes: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
    """Get iteration status from Excel - compatibility function for existing code"""
    excel_manager = ExcelManager()
    excel_manager.recipes = recipes
    return excel_manager.get_iteration_status()

def _get_proposed_recipes_for_iteration(recipes: pd.DataFrame, iteration_num: int) -> List[Dict[str, Any]]:
    """Get proposed recipes for iteration - compatibility function for existing code"""
    excel_manager = ExcelManager()
    excel_manager.recipes = recipes
    return excel_manager.get_proposed_recipes_for_iteration(iteration_num)

def _calculate_uncertainties_for_iteration(iteration_num: int, recipes_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate uncertainties for iteration - compatibility function for existing code"""
    excel_manager = ExcelManager()
    return excel_manager.calculate_uncertainties_for_iteration(iteration_num, recipes_df)

def _update_ingestion_status(recipes: pd.DataFrame, new_completed_lots: List[str]):
    """Update ingestion status - compatibility function for existing code"""
    excel_manager = ExcelManager()
    excel_manager.recipes = recipes
    excel_manager.update_ingestion_status(new_completed_lots)
