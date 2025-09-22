#!/usr/bin/env python3
"""
Test script for the new comprehensive iteration plotting system in pareto.py
"""

import os
import sys
import pandas as pd
import numpy as np

# Add the current directory to Python path to import pareto functions
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_comprehensive_plotting():
    """Test the comprehensive plotting system"""
    print("Testing comprehensive iteration plotting system...")
    
    try:
        # Import the comprehensive plotting functions
        from pareto_new import (
            _create_comprehensive_iteration_plots,
            _plot_front_with_proposed_recipes,
            _plot_parity_with_horizontal_lines,
            _plot_front_with_predicted_and_actual,
            _plot_parity_with_actual_points,
            _get_loocv_data_for_iteration,
            _get_completed_recipes_up_to_iteration,
            _get_highlight_lots_for_iteration
        )
        
        print("✓ Successfully imported comprehensive plotting functions")
        
        # Test data creation
        test_df_summary = pd.DataFrame({
            "LOTNAME": [f"test_lot_{i}" for i in range(10)],
            "FIMAP_FILE": [f"test_fimap_{i}" for i in range(10)],
            "AvgEtchRate": np.random.uniform(50, 150, 10),
            "Range_nm": np.random.uniform(2, 15, 10)
        })
        
        test_current_front = pd.DataFrame({
            "AvgEtchRate": [50, 75, 100, 125, 150],
            "Range_nm": [2, 3, 4, 6, 8]
        })
        
        test_selected_points = [(60, 2.5), (80, 3.0), (110, 4.5)]
        test_selected_uncertainties = [(5.0, 0.5), (6.0, 0.6), (7.0, 0.7)]
        
        # Mock recipes dataframe
        test_recipes_df = pd.DataFrame({
            "O2_flow": [30, 40, 50],
            "cf4_flow": [20, 25, 30],
            "Rf1_Pow": [50, 60, 70],
            "Rf2_Pow": [200, 250, 300],
            "Pressure": [10, 15, 20],
            "predicted_rate": [60, 80, 110],
            "predicted_range": [2.5, 3.0, 4.5],
            "status": ["pending", "pending", "pending"],
            "lotname": ["", "", ""],
            "Pred_etch_rate_uncertainty": [5.0, 6.0, 7.0],
            "Pred_range_uncertainty": [0.5, 0.6, 0.7]
        })
        
        test_iteration_status = {
            1: {
                "start_idx": 0,
                "end_idx": 3,
                "completed_count": 0,
                "pending_count": 3,
                "is_completed": False,
                "recipes": test_recipes_df
            }
        }
        
        print("✓ Successfully created test data")
        
        # Test individual plotting functions
        print("\nTesting individual plotting functions...")
        
        # Test 1: Pareto front with proposed recipes
        try:
            test_plots_dir = "test_comprehensive_plots"
            os.makedirs(test_plots_dir, exist_ok=True)
            
            plot1 = _plot_front_with_proposed_recipes(
                test_df_summary, test_current_front, test_selected_points, 
                test_selected_uncertainties, 1, test_plots_dir
            )
            print(f"✓ Plot 1 created: {plot1}")
            
            # Test 2 & 3: Parity plots with horizontal lines
            plots_23 = _plot_parity_with_horizontal_lines(
                1, test_recipes_df, test_selected_points, 
                test_selected_uncertainties, test_plots_dir
            )
            print(f"✓ Plots 2 & 3 created: {plots_23}")
            
            # Test 4: Pareto front with predicted and actual
            plot4 = _plot_front_with_predicted_and_actual(
                test_df_summary, test_current_front, test_selected_points,
                test_selected_uncertainties, [], test_recipes_df, 1, test_plots_dir
            )
            print(f"✓ Plot 4 created: {plot4}")
            
            # Test 5 & 6: Parity plots with actual points
            plots_56 = _plot_parity_with_actual_points(
                1, test_recipes_df, test_selected_points,
                test_selected_uncertainties, [], test_plots_dir
            )
            print(f"✓ Plots 5 & 6 created: {plots_56}")
            
            print("\n✓ All individual plotting functions working correctly!")
            
        except Exception as e:
            print(f"✗ Error testing individual plotting functions: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Test helper functions
        print("\nTesting helper functions...")
        
        try:
            highlight_lots = _get_highlight_lots_for_iteration(test_recipes_df, 1)
            print(f"✓ Highlight lots function: {highlight_lots}")
            
            completed_recipes = _get_completed_recipes_up_to_iteration(test_recipes_df, 0)
            print(f"✓ Completed recipes function: {len(completed_recipes)} recipes")
            
            print("✓ All helper functions working correctly!")
            
        except Exception as e:
            print(f"✗ Error testing helper functions: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Clean up test files
        import shutil
        if os.path.exists(test_plots_dir):
            shutil.rmtree(test_plots_dir)
            print("✓ Test files cleaned up")
        
        print("\n🎉 All tests passed! The comprehensive plotting system is working correctly.")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Make sure pareto.py is in the same directory and all dependencies are installed.")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_comprehensive_plotting()
    sys.exit(0 if success else 1)
