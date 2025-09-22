import json
import os
import shutil
from typing import Dict, List, Any, Optional
from ..core.config import *

class IterationManager:
    def __init__(self):
        self.iterations_dir = ITERATIONS_DIR
        
    def get_iteration_data(self, iteration_num: int) -> Dict[str, Any]:
        """Get data for a specific iteration"""
        iter_dir = os.path.join(self.iterations_dir, f"iteration_{iteration_num}")
        if not os.path.exists(iter_dir):
            return {}
            
        # Load iteration data
        data = {}
        
        # Load summary data
        summary_path = os.path.join(iter_dir, "summary_data.csv")
        if os.path.exists(summary_path):
            import pandas as pd
            data["summary_data"] = pd.read_csv(summary_path)
        
        # Load Pareto front
        front_path = os.path.join(iter_dir, "pareto_front.csv")
        if os.path.exists(front_path):
            import pandas as pd
            data["pareto_front"] = pd.read_csv(front_path)
        
        # Load selected points
        selected_path = os.path.join(iter_dir, "selected_points.csv")
        if os.path.exists(selected_path):
            import pandas as pd
            data["selected_points"] = pd.read_csv(selected_path)
        
        # Load iteration status
        status_path = os.path.join(iter_dir, "iteration_status.json")
        if os.path.exists(status_path):
            with open(status_path, 'r') as f:
                data["iteration_status"] = json.load(f)
        
        # Load highlight lots
        highlight_path = os.path.join(iter_dir, "highlight_lots.txt")
        if os.path.exists(highlight_path):
            with open(highlight_path, 'r') as f:
                data["highlight_lots"] = [line.strip() for line in f.readlines()]
        
        return data
    
    def save_iteration_data(self, iteration_num: int, data: Dict[str, Any]):
        """Save data for a specific iteration"""
        iter_dir = os.path.join(self.iterations_dir, f"iteration_{iteration_num}")
        os.makedirs(iter_dir, exist_ok=True)
        
        # Save summary data
        if "summary_data" in data:
            summary_path = os.path.join(iter_dir, "summary_data.csv")
            data["summary_data"].to_csv(summary_path, index=False)
        
        # Save Pareto front
        if "pareto_front" in data:
            front_path = os.path.join(iter_dir, "pareto_front.csv")
            data["pareto_front"].to_csv(front_path, index=False)
        
        # Save selected points
        if "selected_points" in data:
            selected_path = os.path.join(iter_dir, "selected_points.csv")
            data["selected_points"].to_csv(selected_path, index=False)
        
        # Save iteration status
        if "iteration_status" in data:
            status_path = os.path.join(iter_dir, "iteration_status.json")
            with open(status_path, 'w') as f:
                json.dump(data["iteration_status"], f, indent=2)
        
        # Save highlight lots
        if "highlight_lots" in data:
            highlight_path = os.path.join(iter_dir, "highlight_lots.txt")
            with open(highlight_path, 'w') as f:
                for lot in data["highlight_lots"]:
                    f.write(f"{lot}\n")
    
    def list_iterations(self) -> List[int]:
        """List all available iterations"""
        if not os.path.exists(self.iterations_dir):
            return []
            
        iterations = []
        for item in os.listdir(self.iterations_dir):
            if item.startswith("iteration_"):
                try:
                    iter_num = int(item.split("_")[1])
                    iterations.append(iter_num)
                except ValueError:
                    continue
        return sorted(iterations)
    
    def clear_iterations(self):
        """Clear all iteration data"""
        if os.path.exists(self.iterations_dir):
            shutil.rmtree(self.iterations_dir)
            os.makedirs(self.iterations_dir)
            print("[iteration_manager] All iteration data cleared")
    
    def get_iteration_plots(self, iteration_num: int) -> List[str]:
        """Get list of plot files for a specific iteration"""
        iter_plots_dir = os.path.join(self.iterations_dir, f"iteration_{iteration_num}", "plots")
        if not os.path.exists(iter_plots_dir):
            return []
            
        plots = []
        for file in os.listdir(iter_plots_dir):
            if file.endswith('.png'):
                plots.append(os.path.join(iter_plots_dir, file))
        
        return sorted(plots)
    
    def regenerate_iteration(self, iteration_num: int, df_summary, current_front, 
                           selected_points, recipes_df, iteration_status):
        """Regenerate a specific iteration"""
        from plotter import _create_comprehensive_iteration_plots
        
        print(f"[iteration_manager] Regenerating iteration {iteration_num}")
        
        # Create comprehensive plots
        result = _create_comprehensive_iteration_plots(iteration_num, df_summary, current_front, 
                                                    selected_points, recipes_df, iteration_status)
        
        if result:
            print(f"[iteration_manager] Successfully regenerated iteration {iteration_num}")
            return True
        else:
            print(f"[iteration_manager] Failed to regenerate iteration {iteration_num}")
            return False
    
    def regenerate_all_iterations(self, df_summary, current_front, recipes_df):
        """Regenerate all iterations"""
        print("[iteration_manager] Regenerating all iterations")
        
        # Get iteration status
        from excel_manager import _get_excel_iteration_status
        iteration_status = _get_excel_iteration_status(recipes_df)
        
        if not iteration_status:
            print("[iteration_manager] No iterations found to regenerate")
            return False
        
        success_count = 0
        for iteration_num in iteration_status.keys():
            if self.regenerate_iteration(iteration_num, df_summary, current_front, 
                                       [], recipes_df, iteration_status):
                success_count += 1
        
        print(f"[iteration_manager] Successfully regenerated {success_count}/{len(iteration_status)} iterations")
        return success_count == len(iteration_status)

def list_iterations():
    """List all available iterations - compatibility function for existing code"""
    iteration_manager = IterationManager()
    iterations = iteration_manager.list_iterations()
    
    if not iterations:
        print("No iterations found")
        return
    
    print("Available iterations:")
    for iter_num in iterations:
        print(f"  Iteration {iter_num}")
        
        # Get plot count
        plots = iteration_manager.get_iteration_plots(iter_num)
        print(f"    Plots: {len(plots)}")
        
        # Get data info
        data = iteration_manager.get_iteration_data(iter_num)
        if data:
            print(f"    Data files: {list(data.keys())}")

def clear_iteration_cache():
    """Clear all iteration cache data - compatibility function for existing code"""
    iteration_manager = IterationManager()
    iteration_manager.clear_iterations()

def regenerate_all_iterations():
    """Regenerate all iterations - compatibility function for existing code"""
    # This would need to be called from main with proper data
    print("[regenerate] This function needs to be called from main with proper data context")
    return False

def view_iteration(iteration_num: int):
    """View a specific iteration - compatibility function for existing code"""
    iteration_manager = IterationManager()
    
    # Check if iteration exists
    if iteration_num not in iteration_manager.list_iterations():
        print(f"Iteration {iteration_num} not found")
        return
    
    # Get iteration data
    data = iteration_manager.get_iteration_data(iteration_num)
    
    print(f"\n=== Iteration {iteration_num} ===")
    
    # Display summary
    if "summary_data" in data:
        print(f"Summary data: {len(data['summary_data'])} records")
    
    if "pareto_front" in data:
        print(f"Pareto front: {len(data['pareto_front'])} points")
    
    if "selected_points" in data:
        print(f"Selected points: {len(data['selected_points'])} recipes")
    
    if "highlight_lots" in data:
        print(f"Highlight lots: {data['highlight_lots']}")
    
    # List plots
    plots = iteration_manager.get_iteration_plots(iteration_num)
    if plots:
        print(f"\nPlots available:")
        for plot in plots:
            plot_name = os.path.basename(plot)
            print(f"  {plot_name}")
    else:
        print("\nNo plots found for this iteration")
