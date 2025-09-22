"""
Main execution module for Pareto optimization system.
Orchestrates all components and replicates the main functionality from pareto.py
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple

from ..core.config import (
    CODE_VERSION, ROOT_DIR, ROLLING_DIR, SNAPSHOTS_DIR, MANIFESTS_DIR,
    ITERATIONS_DIR, SNAP_DIR, PLOTS_DIR, TODAY, FEATURES, FEATURE_RANGES,
    TARGET_RATE_MIN, TARGET_RATE_MAX, K, SAMPLING_METHOD, N_SAMPLES,
    SAMPLING_SEED, ALPHA, BETA, GAMMA, POINTS_PER_ITERATION, INGEST_MODE,
    DATABASE_FRESH_START, CACHE_MANAGEMENT_MODE
)

from ..data.data_manager import DataManager
from ..data.cache_manager import CacheManager
from ..ml.ml_models import MLModels, SamplingEngine
from ..optimization.pareto_optimizer import ParetoOptimizer
from ..visualization.plotter import Plotter


class ParetoSystem:
    """Main Pareto optimization system that orchestrates all components."""
    
    def __init__(self):
        # Initialize cache manager first
        self.cache_manager = CacheManager()
        
        # Prepare cache based on configuration
        if not self.cache_manager.prepare_cache():
            print("[main] ⚠️ Cache preparation failed, continuing anyway...")
        
        # Initialize other components
        self.data_manager = DataManager()
        self.ml_models = MLModels()
        self.sampling_engine = SamplingEngine()
        self.pareto_optimizer = ParetoOptimizer()
        self.plotter = Plotter()
        
        # Initialize database with fresh start option
        self.data_manager.database = self.data_manager.database.__class__(fresh_start=DATABASE_FRESH_START)
        
        # Ensure directories exist
        self.data_manager._ensure_dirs()
    
    def clear_all_cache(self):
        """Clear all cache files and rebuild from scratch"""
        print(f"[cache] Clearing all cache files...")
        import shutil
        import os
        
        # Clear cache directories
        cache_dirs = [ITERATIONS_DIR, ROLLING_DIR, SNAPSHOTS_DIR, MANIFESTS_DIR]
        for cache_dir in cache_dirs:
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
                print(f"[cache] Cleared {cache_dir}")
        
        # Clear database
        db_path = "pareto_cache.db"
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
                print(f"[cache] Cleared {db_path}")
            except PermissionError:
                print(f"[cache] Warning: Could not delete {db_path} (file in use), will be recreated")
        
        # Recreate directories
        self.data_manager._ensure_dirs()
        print(f"[cache] Cache cleared and directories recreated")
    
    def _cache_has_data(self) -> bool:
        """Check if cache has actual data (not just empty directories)"""
        import os
        
        # First check if database exists and has data
        db_path = os.path.join(os.getcwd(), "pareto_cache.db")
        if os.path.exists(db_path):
            try:
                import sqlite3
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM iterations")
                iteration_count = cursor.fetchone()[0]
                conn.close()
                
                if iteration_count > 0:
                    print(f"[cache] Database has {iteration_count} iterations - cache has data")
                    return True
            except Exception as e:
                print(f"[cache] Error checking database: {e}")
        
        # Fallback: Check if cache directory has plots
        if not os.path.exists(ITERATIONS_DIR):
            return False
        
        iteration_dirs = [d for d in os.listdir(ITERATIONS_DIR) if d.startswith('iteration_')]
        if not iteration_dirs:
            return False
        
        # Check if any iteration has plots
        for iteration_dir in iteration_dirs:
            plots_dir = os.path.join(ITERATIONS_DIR, iteration_dir, "plots")
            if os.path.exists(plots_dir):
                plot_files = [f for f in os.listdir(plots_dir) if f.endswith('.png')]
                if plot_files:
                    print(f"[cache] Found {len(plot_files)} plot files - cache has data")
                    return True
        
        print(f"[cache] No cache data found (no database iterations or plot files)")
        return False
    
    def _cache_directory_empty_but_database_has_data(self) -> bool:
        """Check if cache directory is empty but database has data (need to regenerate cache)"""
        import os
        
        # Check if database has data
        db_path = os.path.join(os.getcwd(), "pareto_cache.db")
        if not os.path.exists(db_path):
            return False
        
        try:
            import sqlite3
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM iterations")
            iteration_count = cursor.fetchone()[0]
            conn.close()
            
            if iteration_count == 0:
                return False
        except Exception:
            return False
        
        # Check if cache directory is empty (no plot files)
        if not os.path.exists(ITERATIONS_DIR):
            return True
        
        iteration_dirs = [d for d in os.listdir(ITERATIONS_DIR) if d.startswith('iteration_')]
        if not iteration_dirs:
            return True
        
        # Check if any iteration has plots
        for iteration_dir in iteration_dirs:
            plots_dir = os.path.join(ITERATIONS_DIR, iteration_dir, "plots")
            if os.path.exists(plots_dir):
                plot_files = [f for f in os.listdir(plots_dir) if f.endswith('.png')]
                if plot_files:
                    return False  # Cache has plots, not empty
        
        # Database has data but cache directory is empty
        return True
    
    def check_iteration_completion_status(self) -> bool:
        """Check if the previous iteration is complete before proceeding with new processing"""
        print(f"[incremental] Checking iteration completion status...")
        
        # Read recipes Excel
        recipes = self.data_manager.read_recipes_excel()
        if recipes is None:
            print(f"[incremental] No recipes found - proceeding with first iteration")
            return True
        
        # CRITICAL: Delete recipe_rejected rows FIRST
        if 'Ingestion_status' in recipes.columns:
            rejected_mask = recipes['Ingestion_status'] == 'recipe_rejected'
            rejected_count = rejected_mask.sum()
            if rejected_count > 0:
                print(f"[incremental] Deleting {rejected_count} recipe_rejected rows before checking status")
                recipes = recipes[~rejected_mask].copy()
                # Write back to SharePoint immediately
                success = self.data_manager.write_recipes_to_sharepoint(recipes)
                if success:
                    print(f"[incremental] Successfully deleted {rejected_count} recipe_rejected rows from SharePoint")
                else:
                    print(f"[incremental] Failed to delete recipe_rejected rows from SharePoint")
        
        # Get iteration status (after deletion)
        iteration_status = self.data_manager.get_excel_iteration_status(recipes)
        if not iteration_status:
            print(f"[incremental] No iteration status found - proceeding with first iteration")
            return True
        
        # Find the latest iteration
        latest_iteration = max(iteration_status.keys())
        latest_status = iteration_status[latest_iteration]
        
        print(f"[incremental] Latest iteration: {latest_iteration}")
        print(f"[incremental] Status: {latest_status}")
        
        # Check if latest iteration is complete
        if latest_status.get("status") == "completed":
            print(f"[incremental] Iteration {latest_iteration} is complete - proceeding with next iteration")
            return True
        else:
            print(f"[incremental] Iteration {latest_iteration} is not complete - no new processing needed")
            return False
    
    def run_main_optimization(self, met_snap: bool = True) -> Dict[str, Any]:
        """Run the main Pareto optimization process with incremental architecture"""
        print(f"[main] Starting Pareto optimization system v{CODE_VERSION}")
        print(f"[main] Cache management mode: {CACHE_MANAGEMENT_MODE}")
        
        # Check cache management mode
        if CACHE_MANAGEMENT_MODE == "full_rebuild":
            print(f"[main] FULL REBUILD MODE: Clearing all cache and rebuilding from scratch")
            self.clear_all_cache()
            # In full rebuild mode, skip incremental checks and proceed with full processing
            print(f"[main] FULL REBUILD MODE: Skipping incremental checks, proceeding with full processing")
        else:
            # Check if cache exists and has data
            if not self._cache_has_data():
                print(f"[main] No cache data found - switching to full rebuild mode")
                print(f"[main] FULL REBUILD MODE: Clearing all cache and rebuilding from scratch")
                self.clear_all_cache()
                print(f"[main] FULL REBUILD MODE: Skipping incremental checks, proceeding with full processing")
            else:
                # Cache regeneration is handled by cache_manager._trigger_cache_regeneration()
                # No need to duplicate it here
                # Only check iteration completion in incremental mode when cache has data
                print(f"[main] Cache has data - checking iteration completion status...")
                if not self.check_iteration_completion_status():
                    print(f"[main] Previous iteration not complete - no new processing needed")
                    return {"status": "no_processing_needed", "reason": "previous_iteration_not_complete"}
                else:
                    print(f"[main] Iteration completion check passed - proceeding with incremental processing")
        
        # Load dataset
        print("[main] Loading dataset...")
        df = self.data_manager.load_dataset()
        print(f"[main] Loaded dataset with {len(df)} rows")
        
        # Filter summary data
        df_summary = df[["LOTNAME", "FIMAP_FILE", "AvgEtchRate", "RangeEtchRate"]].copy()
        df_summary = df_summary[df_summary["AvgEtchRate"] <= 250].copy()
        df_summary["Range_nm"] = df_summary["RangeEtchRate"] * 5.0
        
        # Read recipes Excel
        print("[main] Reading recipes Excel...")
        recipes = self.data_manager.read_recipes_excel()
        
        # Get new completed lots
        new_completed_lots = self._get_new_completed_lots(recipes)
        if new_completed_lots:
            print(f"[main] Found {len(new_completed_lots)} newly completed lots: {new_completed_lots}")
        
        # Get highlight lots
        highlight_lots_auto = self._get_auto_highlight_lots(recipes)
        print(f"[main] Auto highlight lots: {highlight_lots_auto}")
        
        # Calculate Pareto front
        print("[main] Calculating Pareto front...")
        df_complete = df[["LOTNAME", "FIMAP_FILE", "AvgEtchRate", "RangeEtchRate"]].copy()
        df_complete["Range_nm"] = df_complete["RangeEtchRate"] * 5.0
        current_front = self.pareto_optimizer.pareto_front(df_complete)
        pareto_pts = current_front[["AvgEtchRate", "Range_nm"]].values
        
        # Update Pareto history
        front_ver = self._update_pareto_history(df_complete)
        
        # Get training data for proposals
        print("[main] Preparing training data for proposals...")
        training_df = self._get_training_data_for_main_proposals(df, recipes)
        
        X = training_df[FEATURES].astype(float).values
        y_rate = training_df["AvgEtchRate"].values
        y_range = training_df["RangeEtchRate"].values
        
        # Train models
        print("[main] Training models...")
        iteration_status = self.data_manager.get_excel_iteration_status(recipes) if recipes is not None else {}
        
        # Run LOOCV for each iteration with proper model selection and training data
        print("[main] Running LOOCV for each iteration with iteration-specific model selection...")
        
        if iteration_status:
            # Run LOOCV for each completed iteration
            completed_iterations = [iter_num for iter_num, status in iteration_status.items() 
                                  if status.get('is_completed', False)]
            
            # Also include incomplete iterations 5+ to test GPR models
            all_iterations = list(iteration_status.keys())
            incomplete_iterations_5plus = [iter_num for iter_num in all_iterations 
                                         if iter_num >= 5 and iter_num not in completed_iterations]
            
            # Combine completed iterations with incomplete iterations 5+ for LOOCV testing
            iterations_to_test = completed_iterations + incomplete_iterations_5plus
            
            if iterations_to_test:
                print(f"[main] Running LOOCV for iterations: {iterations_to_test}")
                print(f"[main] Completed iterations: {completed_iterations}")
                if incomplete_iterations_5plus:
                    print(f"[main] Incomplete iterations 5+ (for GPR testing): {incomplete_iterations_5plus}")
                
                # Run LOOCV for the latest completed iteration (for main plots)
                if completed_iterations:
                    latest_iteration = max(completed_iterations)
                    loocv_training_df = self._get_training_data_for_iteration(df, recipes, latest_iteration)
                    print(f"[main] Main LOOCV using {len(loocv_training_df)} training points for iteration {latest_iteration}")
                    
                    loocv_results = self.ml_models.run_loocv_iteration_specific(loocv_training_df, latest_iteration, recipes)
                    loocv_path = self.data_manager.loocv_path()
                    loocv_results.to_csv(loocv_path, index=False)
                    
                    # Extract model types from the LOOCV results
                    rate_model_type = loocv_results['rate_model_type'].iloc[0] if 'rate_model_type' in loocv_results.columns else "Unknown"
                    range_model_type = loocv_results['range_model_type'].iloc[0] if 'range_model_type' in loocv_results.columns else "Unknown"
                    print(f"[main] Main LOOCV completed with {rate_model_type} for rate, {range_model_type} for range")
                
                # Run LOOCV for each iteration separately and save results
                self._run_iteration_specific_loocv(df, recipes, iterations_to_test)
            else:
                # No completed iterations, use iteration 1
                print("[main] No completed iterations, using iteration 1 for LOOCV")
                loocv_training_df = self._get_training_data_for_iteration(df, recipes, 1)
                loocv_results = self.ml_models.run_loocv_iteration_specific(loocv_training_df, 1, recipes)
                loocv_path = self.data_manager.loocv_path()
                loocv_results.to_csv(loocv_path, index=False)
        else:
            # No iteration info, use iteration 1
            print("[main] No iteration info, using iteration 1 for LOOCV")
            loocv_training_df = self._get_training_data_for_iteration(df, recipes, 1)
            loocv_results = self.ml_models.run_loocv_iteration_specific(loocv_training_df, 1, recipes)
            loocv_path = self.data_manager.loocv_path()
            loocv_results.to_csv(loocv_path, index=False)
        
        # Create parity plots
        print("[main] Creating parity plots...")
        if new_completed_lots:
            self.plotter.plot_parity_from_loocv_with_highlights(loocv_path, new_completed_lots, recipes)
        elif recipes is not None:
            hl_union = list(set(highlight_lots_auto))
            if hl_union:
                self.plotter.plot_parity_from_loocv_with_highlights(loocv_path, hl_union, recipes)
            else:
                self.plotter.plot_parity_from_loocv(loocv_path)
        else:
            self.plotter.plot_parity_from_loocv(loocv_path)
        
        if iteration_status:
            # Find the highest completed iteration for training
            completed_iterations = [iter_num for iter_num, status in iteration_status.items() 
                                  if status.get('is_completed', False)]
            
            if completed_iterations:
                max_completed_iteration = max(completed_iterations)
                next_iteration = max(iteration_status.keys()) + 1
                
                # Train models with appropriate selection based on the completed iteration
                rate_model, range_model, rate_params, range_params = self.ml_models.train_models_for_iteration(
                    X, y_rate, y_range, max_completed_iteration
                )
                print(f"[main] Trained models using iteration {max_completed_iteration} data for proposing iteration {next_iteration}: {self.ml_models.rate_model_type} for rate, {self.ml_models.range_model_type} for range")
            else:
                # Fallback to iteration 1 models if no completed iterations
                rate_model, range_model, rate_params, range_params = self.ml_models.train_models_for_iteration(
                    X, y_rate, y_range, 1
                )
                print(f"[main] No completed iterations, trained models for iteration 1: {self.ml_models.rate_model_type} for rate, {self.ml_models.range_model_type} for range")
        else:
            # Fallback to iteration 1 models if no iteration info
            rate_model, range_model, rate_params, range_params = self.ml_models.train_models_for_iteration(
                X, y_rate, y_range, 1
            )
            print("[main] No iteration info, using iteration 1 models")
        
        # Generate candidates
        print("[main] Generating candidates...")
        Xcand, lower, upper = self.sampling_engine.generate_candidates()
        Xcand = self.sampling_engine.quantize(Xcand, FEATURES)
        
        # Predict outcomes
        mu_r, sd_r = self.ml_models.pred_stats(rate_model, Xcand)
        mu_g, sd_g = self.ml_models.pred_stats(range_model, Xcand)
        mu_g_nm = mu_g * 5.0
        sd_g_nm = sd_g * 5.0
        
        # Filter candidates
        Xb, mur, mug, mask, rf_mask = self.sampling_engine.filter_candidates(Xcand, mu_r, mu_g_nm)
        
        # Build proposals
        print("[main] Building proposals...")
        xlsx_path, sel_rows, new_recs, rates, ranges_nm = self.pareto_optimizer.build_and_write_proposals(
            Xb, mur, mug, sd_r[mask & rf_mask], sd_g_nm[mask & rf_mask], pareto_pts
        )
        
        # Create front plot
        print("[main] Creating front plot...")
        hl_union = list(set(highlight_lots_auto))
        front_plot = self.plotter.plot_front(
            df_summary, current_front, list(zip(rates, ranges_nm)), hl_union,
            new_completed_lots, recipes
        )
        
        # Calculate cache key for metrics
        dataset_h = self.data_manager.dataset_hash(df)
        features_h = self.data_manager.features_hash()
        model_h = self.data_manager.model_config_hash()
        code_h = self.data_manager.code_hash()
        cache_key_str = self.data_manager.cache_key(dataset_h, features_h, model_h, code_h)
        
        # Create metrics plots
        met_plots = None
        if met_snap:
            # Create metrics_over_time.csv if it doesn't exist
            metrics_path = os.path.join(ROLLING_DIR, "metrics_over_time.csv")
            if not os.path.exists(metrics_path):
                self._create_metrics_over_time(df, cache_key_str)
            if os.path.exists(metrics_path):
                met_plots = self.plotter.plot_metrics_over_time(metrics_path)
        
        # Process iterations
        if recipes is not None:
            print("[main] Processing iterations...")
            self._process_iterations(df, df_summary, current_front, rates, ranges_nm, recipes, iteration_status)
        
        # Update ingestion status
        if new_completed_lots and recipes is not None:
            print(f"[main] Updating ingestion status for {len(new_completed_lots)} newly processed lots")
            self._update_ingestion_status(recipes, new_completed_lots)
        
        # Create manifest
        manifest = self._create_manifest(
            df, front_ver, xlsx_path, front_plot, met_plots, new_completed_lots
        )
        
        print("OK")
        print(json.dumps(manifest, indent=2))
        
        return manifest
    
    def _get_new_completed_lots(self, recipes_df: Optional[pd.DataFrame]) -> List[str]:
        """Get newly completed lots"""
        if recipes_df is None:
            return []
        
        # For testing purposes, return all completed lots
        # This will force the system to process iterations
        completed_lots = []
        if "Status_norm" in recipes_df.columns:
            completed_mask = recipes_df["Status_norm"] == "completed"
            # Handle case sensitivity for lotname column
            lotname_col = "Lotname" if "Lotname" in recipes_df.columns else "LOTNAME"
            if lotname_col in recipes_df.columns:
                completed_lots = recipes_df.loc[completed_mask, lotname_col].tolist()
            else:
                completed_lots = []
        
        return completed_lots
    
    def _get_auto_highlight_lots(self, recipes_df: Optional[pd.DataFrame]) -> List[str]:
        """Get automatically highlighted lots"""
        if recipes_df is None:
            return []
        
        # Get completed lots from recipes
        completed_lots = []
        if "Status_norm" in recipes_df.columns:
            completed_mask = recipes_df["Status_norm"] == "completed"
            # Handle case sensitivity for lotname column
            lotname_col = "Lotname" if "Lotname" in recipes_df.columns else "LOTNAME"
            if lotname_col in recipes_df.columns:
                completed_lots = recipes_df.loc[completed_mask, lotname_col].tolist()
            else:
                completed_lots = []
        
        return completed_lots
    
    def _get_loocv_iteration(self, iteration_status: Dict[int, Dict[str, Any]]) -> int:
        """Get the iteration number to use for LOOCV model selection"""
        if iteration_status:
            completed_iterations = [iter_num for iter_num, status in iteration_status.items() 
                                  if status.get('is_completed', False)]
            if completed_iterations:
                # Use the highest completed iteration for LOOCV model selection
                return max(completed_iterations)
            else:
                # No completed iterations, use iteration 1
                return 1
        else:
            # No iteration info, use iteration 1
            return 1
    
    def _get_training_data_for_iteration(self, df: pd.DataFrame, recipes_df: pd.DataFrame, iteration_num: int) -> pd.DataFrame:
        """Get training data for a specific iteration"""
        return self.data_manager.get_training_data_for_iteration(df, recipes_df, iteration_num)
    
    def _run_iteration_specific_loocv(self, df: pd.DataFrame, recipes: pd.DataFrame, completed_iterations: List[int]):
        """Run LOOCV for each iteration separately and save results"""
        print(f"[loocv] Running iteration-specific LOOCV for {len(completed_iterations)} iterations...")
        
        for iteration_num in completed_iterations:
            print(f"[loocv] Processing iteration {iteration_num}...")
            
            # Get iteration-specific training data
            training_df = self._get_training_data_for_iteration(df, recipes, iteration_num)
            print(f"[loocv] Iteration {iteration_num}: Using {len(training_df)} training points")
            
            # Run LOOCV for this iteration
            loocv_results = self.ml_models.run_loocv_iteration_specific(training_df, iteration_num, recipes)
            
            # Save iteration-specific LOOCV results
            loocv_path = os.path.join(ITERATIONS_DIR, f"iteration_{iteration_num}", "loocv_predictions.csv")
            os.makedirs(os.path.dirname(loocv_path), exist_ok=True)
            loocv_results.to_csv(loocv_path, index=False)
            
            # Calculate and print metrics for this iteration
            self._print_iteration_metrics(loocv_results, iteration_num)
            
            print(f"[loocv] Iteration {iteration_num} LOOCV completed and saved to {loocv_path}")
    
    def _print_iteration_metrics(self, loocv_results: pd.DataFrame, iteration_num: int):
        """Print R² and RMSE metrics for an iteration"""
        try:
            from sklearn.metrics import r2_score, mean_squared_error
            
            # Calculate metrics for rate
            rate_r2 = r2_score(loocv_results['loo_true_rate'], loocv_results['loo_pred_rate'])
            rate_rmse = np.sqrt(mean_squared_error(loocv_results['loo_true_rate'], loocv_results['loo_pred_rate']))
            
            # Calculate metrics for range
            range_r2 = r2_score(loocv_results['loo_true_range'], loocv_results['loo_pred_range'])
            range_rmse = np.sqrt(mean_squared_error(loocv_results['loo_true_range'], loocv_results['loo_pred_range']))
            
            print(f"[metrics] Iteration {iteration_num}: Rate R²={rate_r2:.4f}, Rate RMSE={rate_rmse:.4f}, Range R²={range_r2:.4f}, Range RMSE={range_rmse:.4f}")
            
        except Exception as e:
            print(f"[metrics] Error calculating metrics for iteration {iteration_num}: {e}")
    
    def _update_pareto_history(self, df_complete: pd.DataFrame) -> int:
        """Update Pareto front history"""
        # This would implement the Pareto front history tracking
        # For now, return a simple version number
        return 1
    
    def _get_training_data_for_main_proposals(self, df: pd.DataFrame, 
                                            recipes_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Get training data for main proposals"""
        if recipes_df is None:
            return df.copy()
        
        # Use the most recent COMPLETED iteration's training data
        iteration_status = self.data_manager.get_excel_iteration_status(recipes_df)
        if iteration_status:
            # Find the highest completed iteration
            completed_iterations = [iter_num for iter_num, status in iteration_status.items() 
                                  if status.get('is_completed', False)]
            
            if completed_iterations:
                max_completed_iteration = max(completed_iterations)
                print(f"[main] Using training data from last completed iteration {max_completed_iteration}")
                return self.data_manager.get_training_data_for_iteration(df, recipes_df, max_completed_iteration)
            else:
                print("[main] No completed iterations found, using full dataset")
                return df.copy()
        
        return df.copy()
    
    def _process_iterations(self, df: pd.DataFrame, df_summary: pd.DataFrame, 
                           current_front: pd.DataFrame, rates: np.ndarray, ranges_nm: np.ndarray,
                           recipes_df: pd.DataFrame, iteration_status: Dict[int, Dict[str, Any]]):
        """Process all iterations with comprehensive functionality matching pareto.py"""
        print(f"[iteration] Processing {len(iteration_status)} iterations...")
        
        # First, analyze the Excel file to see what iterations already have proposed recipes
        print(f"[iteration] Excel analysis: {len(iteration_status)} iterations with proposed recipes")
        for iter_num, status in iteration_status.items():
            print(f"[iteration] Iteration {iter_num}: {status['completed_count']}/{POINTS_PER_ITERATION} completed")
        
        # Prepare selected points with feature values for potential recipe proposals
        print("[iteration] Preparing selected points with feature values for recipe proposals...")
        selected_points_with_features = []
        for i, (rate, range_nm) in enumerate(zip(rates, ranges_nm)):
            # Get the corresponding feature values from the selected recipes
            features = {
                "O2_flow": 0.0,  # Would need to extract from new_recs
                "cf4_flow": 0.0,
                "Rf1_Pow": 0.0,
                "Rf2_Pow": 0.0,
                "Pressure": 0.0,
                "rate": rate,
                "range_nm": range_nm
            }
            selected_points_with_features.append(features)
            print(f"[iteration] Recipe {i+1}: O2={features['O2_flow']:.1f}, cf4={features['cf4_flow']:.1f}, "
                  f"Rf1={features['Rf1_Pow']:.1f}, Rf2={features['Rf2_Pow']:.1f}, "
                  f"P={features['Pressure']:.1f} → Rate={features['rate']:.1f}, Range={features['range_nm']:.1f}")
        
        # Process each iteration based on Excel data
        max_iteration_in_excel = max(iteration_status.keys()) if iteration_status else -1
        
        # Determine which iterations to process based on completion status
        iterations_to_process = []
        
        # Always process completed iterations
        completed_iterations = [iter_num for iter_num, status in iteration_status.items() 
                              if status.get('is_completed', False)]
        iterations_to_process.extend(completed_iterations)
        
        # Process current iteration if it has any recipes (even if not completed)
        # This allows plotting the first 3 plots that don't require completed runs
        current_iteration = max_iteration_in_excel
        if current_iteration in iteration_status:
            # Check if this iteration has any recipes (completed or pending)
            total_recipes = iteration_status[current_iteration].get('completed_count', 0) + iteration_status[current_iteration].get('pending_count', 0)
            if total_recipes > 0 and current_iteration not in iterations_to_process:
                iterations_to_process.append(current_iteration)
        
        # Only add next iteration if current iteration is completed
        if current_iteration in iteration_status and iteration_status[current_iteration].get('is_completed', False):
            next_iteration = current_iteration + 1
            iterations_to_process.append(next_iteration)
        
        print(f"[iteration] Will process iterations: {iterations_to_process}")
        
        # Calculate uncertainties for iterations that will be processed
        iteration_params = {}  # Store parameters for each iteration
        if recipes_df is not None:
            print(f"[uncertainty] Calculating uncertainties for {len(iterations_to_process)} iterations...")
            recipes_with_uncertainties = recipes_df.copy()
            
            # Calculate uncertainties for each iteration that will be processed
            for iter_num in iterations_to_process:
                recipes_with_uncertainties, rate_params, range_params = self._calculate_uncertainties_for_iteration(iter_num, recipes_with_uncertainties)
                iteration_params[iter_num] = {'rate_params': rate_params, 'range_params': range_params}
            
            # Save the updated recipes with uncertainties back to the Excel file
            print("[uncertainty] Uncertainties calculated and added to recipes DataFrame")
        
        for iteration_num in iterations_to_process:
            print(f"[iteration] Processing iteration {iteration_num}...")
            
            # Ensure iteration directory exists
            self.cache_manager.ensure_iteration_directory(iteration_num)
            
            # Recalculate Pareto front for this iteration to ensure it includes the most up-to-date data
            print(f"[iteration] Recalculating Pareto front for iteration {iteration_num}...")
            df_complete_iteration = df[["LOTNAME", "FIMAP_FILE", "AvgEtchRate", "RangeEtchRate"]].copy()
            df_complete_iteration["Range_nm"] = df_complete_iteration["RangeEtchRate"] * 5.0
            current_front_iteration = self.pareto_optimizer.pareto_front(df_complete_iteration)
            print(f"[iteration] Pareto front for iteration {iteration_num} has {len(current_front_iteration)} points")
            
            # Create comprehensive plots for this iteration using the new 8-plot system
            # Get the proposed recipes for this specific iteration from Excel
            proposed_recipes_for_iteration = self.data_manager.get_proposed_recipes_for_iteration(recipes_with_uncertainties, iteration_num)
            
            if proposed_recipes_for_iteration:
                # Convert to the format expected by plotting functions
                selected_points_for_plot = [(r["predicted_rate"], r["predicted_range"]) for r in proposed_recipes_for_iteration]
            else:
                # Fallback to generated Pareto points if no Excel recipes exist
                selected_points_for_plot = list(zip(rates, ranges_nm))
            
            front_plot = self._create_comprehensive_iteration_plots(iteration_num, df_summary, current_front_iteration, 
                                                             selected_points_for_plot, 
                                                             recipes_with_uncertainties, iteration_status, df, iteration_params)
            
            if front_plot is None:
                print(f"[iteration] Skipping iteration {iteration_num} - no data available")
                continue
            
            # Don't propose new recipes here - only process existing iterations
            
            print(f"[iteration] Completed iteration {iteration_num}")
    
        # Only propose new recipes if the last iteration is completed and there are no pending iterations
        last_iteration = max_iteration_in_excel
        if last_iteration in iteration_status and iteration_status[last_iteration]["is_completed"]:
            next_iteration = last_iteration + 1
            print(f"[iteration] Last iteration {last_iteration} is completed. Proposing recipes for iteration {next_iteration}...")
            
            # Ensure iteration directory exists
            self.cache_manager.ensure_iteration_directory(next_iteration)
            
            # Generate new recipes for this iteration
            success, rates, ranges_nm = self._propose_next_iteration_recipes(recipes_df, next_iteration, selected_points_with_features)
            if success:
                print(f"[iteration] Successfully proposed recipes for iteration {next_iteration}")
            else:
                print(f"[iteration] Failed to propose recipes for iteration {next_iteration}")
        else:
            print(f"[iteration] Last iteration {last_iteration} is not completed yet. No new recipes to propose.")
    
    def _calculate_uncertainties_for_iteration(self, iteration_num: int, recipes_df: pd.DataFrame) -> Tuple[pd.DataFrame, dict, dict]:
        """Calculate uncertainties for recipes in a specific iteration"""
        print(f"[uncertainty] Calculating uncertainties for iteration {iteration_num}")
        
        # Load the full dataset
        try:
            df = self.data_manager.load_dataset()
            print(f"[uncertainty] Loaded dataset with {len(df)} rows")
        except Exception as e:
            print(f"[uncertainty] Error loading dataset: {e}")
            return recipes_df
        
        # Get the proper training data for this iteration
        training_data = self.data_manager.get_training_data_for_iteration(df, recipes_df, iteration_num)
        
        if training_data.empty:
            print(f"[uncertainty] Iteration {iteration_num}: No training data available")
            return recipes_df
        
        # Prepare features and targets
        X = training_data[FEATURES].values
        y_rate = training_data['AvgEtchRate'].values
        y_range = training_data['RangeEtchRate'].values
        
        # Train models with appropriate selection based on iteration
        print(f"[uncertainty] Training models for iteration {iteration_num}...")
        rate_model, range_model, rate_params, range_params = self.ml_models.train_models_for_iteration(
            X, y_rate, y_range, iteration_num
        )
        print(f"[uncertainty] Iteration {iteration_num}: Using {self.ml_models.rate_model_type} for rate, {self.ml_models.range_model_type} for range")
        
        # Get the first 3 recipes from this iteration for uncertainty calculation
        iteration_recipes = recipes_df.head(3)
        
        if len(iteration_recipes) == 0:
            print(f"[uncertainty] No recipes found for iteration {iteration_num}")
            return recipes_df
        
        # Extract features for these recipes
        recipe_features = []
        for _, recipe in iteration_recipes.iterrows():
            # Extract feature values from the recipe
            feature_values = []
            for feature in FEATURES:
                if feature in recipe.index:
                    feature_values.append(recipe[feature])
                else:
                    # Use default values if features not found
                    feature_values.append(np.mean(training_data[feature]))
            recipe_features.append(feature_values)
        
        recipe_features = np.array(recipe_features)
        
        # Predict uncertainties
        rate_mean, rate_std = self.ml_models.pred_stats(rate_model, recipe_features)
        range_mean, range_std = self.ml_models.pred_stats(range_model, recipe_features)
        
        # Convert range uncertainty to thickness range (multiply by 5)
        range_std_thickness = range_std * 5.0
        
        print(f"[uncertainty] Predicted uncertainties for {len(iteration_recipes)} recipes:")
        for i, (_, recipe) in enumerate(iteration_recipes.iterrows()):
            print(f"[uncertainty] Recipe {i+1}: Rate ±{rate_std[i]:.2f}, Range ±{range_std_thickness[i]:.2f}")
        
        # Add uncertainty columns to recipes DataFrame
        recipes_df_copy = recipes_df.copy()
        
        # Add uncertainty columns if they don't exist
        if 'Pred_etch_rate_uncertainty' not in recipes_df_copy.columns:
            recipes_df_copy['Pred_etch_rate_uncertainty'] = np.nan
        if 'Pred_range_uncertainty' not in recipes_df_copy.columns:
            recipes_df_copy['Pred_range_uncertainty'] = np.nan
        
        # Update uncertainties for the first 3 recipes
        for i in range(min(3, len(recipes_df_copy))):
            recipes_df_copy.loc[i, 'Pred_etch_rate_uncertainty'] = rate_std[i]
            recipes_df_copy.loc[i, 'Pred_range_uncertainty'] = range_std_thickness[i]
        
        print(f"[uncertainty] Added uncertainty columns to recipes DataFrame")
        return recipes_df_copy, rate_params, range_params
    
    def _create_comprehensive_iteration_plots(self, iteration_num: int, df_summary: pd.DataFrame, 
                                            current_front_iteration: pd.DataFrame, selected_points: List[Tuple[float, float]], 
                                            recipes_df: pd.DataFrame, iteration_status: Dict[int, Dict[str, Any]], 
                                            original_df: pd.DataFrame, iteration_params: Dict[int, Dict[str, Any]]) -> Optional[str]:
        """Create all 9 comprehensive plots for a specific iteration"""
        print(f"[comprehensive_plots] Creating all 9 plots for iteration {iteration_num}")
        
        # Create iteration directory
        iter_dir = os.path.join(ITERATIONS_DIR, f"iteration_{iteration_num}")
        iter_plots_dir = os.path.join(iter_dir, "plots")
        os.makedirs(iter_dir, exist_ok=True)
        os.makedirs(iter_plots_dir, exist_ok=True)
        
        # Get proposed recipes for this iteration
        proposed_recipes = self.data_manager.get_proposed_recipes_for_iteration(recipes_df, iteration_num)
        
        # Get completed recipes from previous iterations for training progression
        completed_recipes = self.data_manager.get_completed_recipes_up_to_iteration(recipes_df, iteration_num - 1)
        
        # Get highlight lots for this iteration (completed recipes)
        highlight_lots = self.data_manager.get_highlight_lots_for_iteration(recipes_df, iteration_num)
        
        # Get training data count for this iteration (for accurate legend)
        training_data = self.data_manager.get_training_data_for_iteration(original_df, recipes_df, iteration_num)
        training_count = len(training_data)
        print(f"[comprehensive_plots] Iteration {iteration_num}: Training data count = {training_count}")
        
        # Extract uncertainties for the proposed recipes
        selected_uncertainties = []
        if proposed_recipes:
            for i, r in enumerate(proposed_recipes):
                # Get uncertainties from the recipe data
                rate_uncertainty = r.get("rate_uncertainty", 0.0)
                range_uncertainty = r.get("range_uncertainty", 0.0)
                
                # Fallback to old column names if new ones don't exist
                if rate_uncertainty == 0.0:
                    rate_uncertainty = r.get("Pred_etch_rate_uncertainty", 0.0)
                if range_uncertainty == 0.0:
                    range_uncertainty = r.get("Pred_range_uncertainty", 0.0)
                    
                selected_uncertainties.append((rate_uncertainty, range_uncertainty))
        
        # Use the selected_points passed from main function (these are the Excel recipes)
        selected_points_for_plot = selected_points
        
        # Plot 1: Pareto front with 3 proposed recipes (⓿₁ symbols with uncertainty bars)
        print(f"[comprehensive_plots] Creating Plot 1: Pareto front with proposed recipes")
        front_plot_1 = self.plotter.plot_front_with_proposed_recipes(df_summary, current_front_iteration, 
                                                                   selected_points_for_plot, selected_uncertainties, 
                                                                   highlight_lots, iteration_num, iter_plots_dir, training_count)
        
        # Plot 2 & 3: Parity plots with horizontal lines for proposed points
        print(f"[comprehensive_plots] Creating Plots 2 & 3: Parity plots with horizontal lines")
        parity_plots_23 = self.plotter.plot_parity_with_horizontal_lines(iteration_num, recipes_df, 
                                                                        selected_points_for_plot, selected_uncertainties,
                                                                        highlight_lots, iter_plots_dir)
        
        # Plot 4: Pareto front with both predicted (faded) and actual (full opacity) points
        print(f"[comprehensive_plots] Creating Plot 4: Pareto front with predicted and actual")
        front_plot_4 = self.plotter.plot_front_with_predicted_and_actual(df_summary, current_front_iteration, 
                                                                         selected_points_for_plot, selected_uncertainties,
                                                                         highlight_lots, recipes_df, iteration_num, 
                                                                         iter_plots_dir, training_count)
        
        # Plot 5 & 6: Parity plots with actual points instead of horizontal lines
        print(f"[comprehensive_plots] Creating Plots 5 & 6: Parity plots with actual points")
        parity_plots_56 = self.plotter.plot_parity_with_actual_points(iteration_num, recipes_df, 
                                                                      selected_points_for_plot, selected_uncertainties,
                                                                      highlight_lots, iter_plots_dir, original_df)
        
        # Plot 7 & 8: Metrics plots
        print(f"[comprehensive_plots] Creating Plots 7 & 8: Metrics plots")
        metrics_plots = self.plotter.plot_metrics_for_iteration(iteration_num, iter_plots_dir)
        
        # Plot 9: DEBUG - Pareto front with hypothetical completion
        print(f"[comprehensive_plots] Creating Plot 9: Debug Pareto front with hypothetical completion")
        debug_plot_9 = self.plotter.plot_debug_pareto_with_hypothetical_completion(df_summary, current_front_iteration, 
                                                                                  selected_points_for_plot, 
                                                                                  iteration_num, iter_plots_dir)
        
        # Save iteration data
        self._save_iteration_data_with_excel_info(iteration_num, df_summary, current_front_iteration,
                                                selected_points_for_plot, iteration_status, recipes_df, original_df, iteration_params)
        
        print(f"[comprehensive_plots] Successfully created all 9 plots for iteration {iteration_num}")
        return front_plot_1
    
    def _save_iteration_data_with_excel_info(self, iteration_num: int, df_summary: pd.DataFrame, 
                                           current_front_iteration: pd.DataFrame, selected_points: List[Tuple[float, float]], 
                                           iteration_status: Dict[int, Dict[str, Any]], recipes_df: pd.DataFrame, 
                                           original_df: pd.DataFrame, iteration_params: Dict[int, Dict[str, Any]]):
        """Save iteration data with Excel information"""
        iter_dir = os.path.join(ITERATIONS_DIR, f"iteration_{iteration_num}")
        
        # Save iteration status
        status_data = {
            "iteration": int(iteration_num),  # Convert to Python int
            "status": str(iteration_status.get(iteration_num, {})),
            "timestamp": datetime.now().isoformat()
        }
        status_path = os.path.join(iter_dir, "iteration_status.json")
        with open(status_path, 'w') as f:
            json.dump(status_data, f, indent=2)
        
        # Save selected points
        selected_df = pd.DataFrame(selected_points, columns=["rate", "range_nm"])
        selected_path = os.path.join(iter_dir, "selected_points.csv")
        selected_df.to_csv(selected_path, index=False)
        
        # Save Pareto front
        front_path = os.path.join(iter_dir, "pareto_front.csv")
        current_front_iteration.to_csv(front_path, index=False)
        
        # Save summary data
        summary_path = os.path.join(iter_dir, "summary_data.csv")
        df_summary.to_csv(summary_path, index=False)
        
        # Save highlight lots
        highlight_lots = self.data_manager.get_highlight_lots_for_iteration(recipes_df, iteration_num)
        highlight_path = os.path.join(iter_dir, "highlight_lots.txt")
        with open(highlight_path, 'w') as f:
            f.write('\n'.join(highlight_lots))
        
        # Save training data debug info
        # Get the correct model types for this iteration
        if iteration_num <= 2:
            rate_model_type = "RandomForest"
            range_model_type = "RandomForest"
        elif iteration_num <= 4:
            rate_model_type = "ExtraTrees"
            range_model_type = "RandomForest"
        else:
            rate_model_type = "GPR"
            range_model_type = "GPR"
        
        # Get the parameters from the uncertainty calculation for this specific iteration
        if iteration_num in iteration_params:
            actual_rate_params = iteration_params[iteration_num]['rate_params']
            actual_range_params = iteration_params[iteration_num]['range_params']
        else:
            # Fallback to ML models object if parameters not available
            actual_rate_params = getattr(self.ml_models, 'rate_params', {})
            actual_range_params = getattr(self.ml_models, 'range_params', {})
        
        debug_info = self.data_manager.get_training_data_debug_info(
            original_df, recipes_df, iteration_num,
            rate_model_type, range_model_type,
            actual_rate_params, actual_range_params
        )
        debug_path = os.path.join(iter_dir, "training_data_debug.txt")
        with open(debug_path, 'w') as f:
            f.write(f"First lot: {debug_info.get('first_lot', 'N/A')}\n")
            f.write(f"Last lot: {debug_info.get('last_lot', 'N/A')}\n")
            f.write(f"Training cutoff date: {debug_info.get('training_cutoff_date', 'N/A')}\n")
            f.write(f"Total training points: {debug_info.get('total_training_points', 'N/A')}\n")
            if 'rate_model' in debug_info:
                f.write(f"Rate model: {debug_info['rate_model']}\n")
            if 'range_model' in debug_info:
                f.write(f"Range model: {debug_info['range_model']}\n")
            if 'rate_params' in debug_info:
                f.write(f"Rate params: {debug_info['rate_params']}\n")
            if 'range_params' in debug_info:
                f.write(f"Range params: {debug_info['range_params']}\n")
        
        # Get LOOCV data for this iteration
        loocv_data = self._get_loocv_data_for_iteration(iteration_num)
        
        # Get uncertainties for this iteration
        uncertainties = self._get_uncertainties_for_iteration(iteration_num)
        
        # Get metrics for this iteration
        metrics = self._get_metrics_for_iteration(iteration_num)
        
        # Save to SQLite database with complete data
        self.data_manager.save_iteration_data_to_database(
            iteration_num, debug_info, highlight_lots, current_front_iteration, selected_points, 
            original_df, loocv_data, uncertainties, metrics
        )
        
        print(f"[iteration] Saved data for iteration {iteration_num}")
    
    def _propose_next_iteration_recipes(self, recipes_df: pd.DataFrame, iteration_num: int,
                                       selected_points_with_features: List[Dict[str, Any]]) -> Tuple[bool, np.ndarray, np.ndarray]:
        """Propose recipes for the specified iteration"""
        print(f"[proposals] Generating new recipes for iteration {iteration_num}")
        
        try:
            # Load dataset for training
            df = self.data_manager.load_dataset()
            
            # Get training data from the last completed iteration (not the current iteration)
            iteration_status = self.data_manager.get_excel_iteration_status(recipes_df)
            if iteration_status:
                completed_iterations = [iter_num for iter_num, status in iteration_status.items() 
                                      if status.get('is_completed', False)]
                
                if completed_iterations:
                    last_completed_iteration = max(completed_iterations)
                    print(f"[proposals] Using training data from last completed iteration {last_completed_iteration} for proposing iteration {iteration_num}")
                    training_df = self.data_manager.get_training_data_for_iteration(df, recipes_df, last_completed_iteration)
                else:
                    print(f"[proposals] No completed iterations found, using full dataset")
                    training_df = df.copy()
            else:
                print(f"[proposals] No iteration status available, using full dataset")
                training_df = df.copy()
            
            if training_df.empty:
                print(f"[proposals] Error: No training data available for iteration {iteration_num}")
                return False
            
            # Prepare features and targets
            X = training_df[FEATURES].astype(float).values
            y_rate = training_df["AvgEtchRate"].values
            y_range = training_df["RangeEtchRate"].values
            
            # Train models using the last completed iteration number
            if completed_iterations:
                training_iteration = max(completed_iterations)
            else:
                training_iteration = 1
                
            rate_model, range_model, rate_params, range_params = self.ml_models.train_models_for_iteration(
                X, y_rate, y_range, training_iteration
            )
            print(f"[proposals] Trained models using iteration {training_iteration} data for proposing iteration {iteration_num}: {self.ml_models.rate_model_type} for rate, {self.ml_models.range_model_type} for range")
            
            # Generate candidates
            Xcand, lower, upper = self.sampling_engine.generate_candidates()
            Xcand = self.sampling_engine.quantize(Xcand, FEATURES)
            
            # Predict outcomes
            mu_r, sd_r = self.ml_models.pred_stats(rate_model, Xcand)
            mu_g, sd_g = self.ml_models.pred_stats(range_model, Xcand)
            mu_g_nm = mu_g * 5.0
            sd_g_nm = sd_g * 5.0
            
            # Filter candidates
            Xb, mur, mug, mask, rf_mask = self.sampling_engine.filter_candidates(Xcand, mu_r, mu_g_nm)
            
            # Calculate current Pareto front
            df_complete = df[["LOTNAME", "FIMAP_FILE", "AvgEtchRate", "RangeEtchRate"]].copy()
            df_complete["Range_nm"] = df_complete["RangeEtchRate"] * 5.0
            current_front = self.pareto_optimizer.pareto_front(df_complete)
            pareto_pts = current_front[["AvgEtchRate", "Range_nm"]].values
            
            # Build proposals using Expected Improvement batch optimization
            xlsx_path, sel_rows, new_recs, rates, ranges_nm = self.pareto_optimizer.build_and_write_proposals_ei(
                Xb, mur, mug, sd_r[mask & rf_mask], sd_g_nm[mask & rf_mask], pareto_pts, self.ml_models
            )
            
            print(f"[proposals] Generated {len(new_recs)} new recipes for iteration {iteration_num}")
            print(f"[proposals] Predicted rates: {rates}")
            print(f"[proposals] Predicted ranges: {ranges_nm}")
            
            # Add these recipes to the SharePoint Excel file
            # The EI method returns the uncertainties directly, so we need to extract them from the selection details
            if len(new_recs) > 0 and len(sel_rows) > 0:
                # Extract uncertainties from the selection details
                rate_uncertainties = np.array([detail['rate_uncertainty'] for detail in sel_rows])
                range_uncertainties = np.array([detail['range_uncertainty'] for detail in sel_rows])
            else:
                rate_uncertainties = np.array([])
                range_uncertainties = np.array([])
            
            # Try to add recipes to SharePoint Excel (but don't fail the entire process if this fails)
            success = self._add_recipes_to_sharepoint_excel(recipes_df, iteration_num, new_recs, rates, ranges_nm, rate_uncertainties, range_uncertainties)
            
            if success:
                print(f"[proposals] Successfully added {len(new_recs)} recipes to SharePoint Excel for iteration {iteration_num}")
            else:
                print(f"[proposals] Failed to add recipes to SharePoint Excel for iteration {iteration_num}")
                print(f"[proposals] Continuing with plot generation despite SharePoint failure...")
            
            # Always return True since we successfully generated the recipes
            # SharePoint writing failure shouldn't prevent plot generation
            return True, rates, ranges_nm
            
        except Exception as e:
            print(f"[proposals] Error generating recipes for iteration {iteration_num}: {e}")
            return False, np.array([]), np.array([])
    
    def _add_recipes_to_sharepoint_excel(self, recipes_df: pd.DataFrame, iteration_num: int, 
                                        new_recs: List[Dict], rates: np.ndarray, ranges_nm: np.ndarray,
                                        rate_uncertainties: np.ndarray, range_uncertainties: np.ndarray) -> bool:
        """Add new recipes to the SharePoint Excel file"""
        try:
            print(f"[DEBUG] _add_recipes_to_sharepoint_excel called with iteration_num={iteration_num}")
            print(f"[proposals] Adding {len(new_recs)} recipes to SharePoint Excel for iteration {iteration_num}")
            
            # CRITICAL: Delete recipe_rejected rows before adding new recipes
            if 'Ingestion_status' in recipes_df.columns:
                rejected_mask = recipes_df['Ingestion_status'] == 'recipe_rejected'
                rejected_count = rejected_mask.sum()
                if rejected_count > 0:
                    print(f"[proposals] Deleting {rejected_count} recipe_rejected rows before adding new recipes")
                    recipes_df = recipes_df[~rejected_mask].copy()
                else:
                    print(f"[proposals] No recipe_rejected rows found")
            
            # Create new recipe entries with correct Excel structure
            new_recipes = []
            for i, (recipe, rate, range_nm, rate_unc, range_unc) in enumerate(zip(new_recs, rates, ranges_nm, rate_uncertainties, range_uncertainties)):
                new_recipe = {
                    # Feature columns (mapped to Excel column names)
                    'O2_flow': recipe[FEATURES.index('Etch_AvgO2Flow')],
                    'cf4_flow': recipe[FEATURES.index('Etch_Avgcf4Flow')],
                    'Rf1_Pow': recipe[FEATURES.index('Etch_Avg_Rf1_Pow')],
                    'Rf2_Pow': recipe[FEATURES.index('Etch_Avg_Rf2_Pow')],
                    'Pressure': recipe[FEATURES.index('Etch_AvgPres')],
                    
                    # Fixed parameters
                    'Chamber_temp': 50,
                    'Electrode_temp': 15,
                    'Etch_time': 5,
                    
                    # Prediction columns
                    'Pred_avg_etch_rate': rate,
                    'Pred_Range': range_nm / 5.0,  # Convert back to original units
                    'Etch_rate_uncertainty': rate_unc,
                    'Range_uncertainty': range_unc / 5.0,  # Convert back to original units
                    
                    # Status and tracking columns
                    'Status': 'pending',
                    'Date_Completed': '',  # Empty for new recipes
                    'Lotname': '',  # Empty for new recipes
                    'idrun': '',  # Empty for new recipes
                    'Iteration_num': iteration_num,  # NEW: Iteration tracking
                    
                    # Workflow columns
                    'Ingestion_status': 'waiting',  # Default for new recipes
                    'Comment': ''  # Empty for new recipes
                }
                print(f"[DEBUG] Created recipe {i+1} with Iteration_num={new_recipe['Iteration_num']}")
                new_recipes.append(new_recipe)
            
            # Add new recipes to the DataFrame
            new_recipes_df = pd.DataFrame(new_recipes)
            updated_recipes_df = pd.concat([recipes_df, new_recipes_df], ignore_index=True)
            
            # Write back to SharePoint Excel file
            success = self.data_manager.write_recipes_to_sharepoint(updated_recipes_df)
            if success:
                print(f"[proposals] Successfully wrote {len(new_recipes)} recipes to SharePoint Excel")
                return True
            else:
                print(f"[proposals] Failed to write recipes to SharePoint Excel")
                return False
                
        except Exception as e:
            print(f"[proposals] Error adding recipes to SharePoint Excel: {e}")
            return False
    
    def _update_ingestion_status(self, recipes_df: pd.DataFrame, new_completed_lots: List[str]):
        """Update ingestion status for processed lots"""
        # This would update the ingestion status in the recipes DataFrame
        pass
    
    def _create_manifest(self, df: pd.DataFrame, front_ver: int, xlsx_path: str, 
                        front_plot: str, met_plots: Optional[Tuple[str, str]], 
                        new_completed_lots: List[str]) -> Dict[str, Any]:
        """Create manifest with all results"""
        # Calculate hashes
        dataset_h = self.data_manager.dataset_hash(df)
        features_h = self.data_manager.features_hash()
        model_h = self.data_manager.model_config_hash()
        code_h = self.data_manager.code_hash()
        cache_key_str = self.data_manager.cache_key(dataset_h, features_h, model_h, code_h)
        
        manifest = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "code_version": CODE_VERSION,
            "ingest_mode": INGEST_MODE,
            "new_completed_lots": list(new_completed_lots),
            "hashes": {
                "dataset": dataset_h, 
                "features": features_h, 
                "model": model_h, 
                "code": code_h, 
                "cache_key": cache_key_str
            },
            "rolling": {
                "predictions_by_date": os.path.relpath(os.path.join(ROLLING_DIR, "predictions_by_date.csv"), ROOT_DIR),
                "loocv_predictions": os.path.relpath(self.data_manager.loocv_path(), ROOT_DIR),
                "pareto_front_history": os.path.relpath(self.data_manager.pareto_history_path(), ROOT_DIR),
                "metrics_over_time": os.path.relpath(os.path.join(ROLLING_DIR, "metrics_over_time.csv"), ROOT_DIR),
            },
            "snapshot": {
                "date": TODAY,
                "dir": os.path.relpath(SNAP_DIR, ROOT_DIR),
                "proposals_xlsx": os.path.relpath(xlsx_path, ROOT_DIR),
                "plots": {
                    "front": os.path.relpath(front_plot, ROOT_DIR),
                    "parity_rate": os.path.relpath(os.path.join(PLOTS_DIR, "parity_rate.png"), ROOT_DIR),
                    "parity_range": os.path.relpath(os.path.join(PLOTS_DIR, "parity_range.png"), ROOT_DIR),
                    "metrics_rmse": os.path.relpath(os.path.join(PLOTS_DIR, "metrics_rmse.png"), ROOT_DIR) if met_plots else None,
                    "metrics_coverage": os.path.relpath(os.path.join(PLOTS_DIR, "metrics_coverage.png"), ROOT_DIR) if met_plots else None,
                }
            },
            "front_version": front_ver
        }
        
        self.data_manager._manifest_write(manifest)
        
        # Save system metadata to database
        self.data_manager.database.save_system_metadata("last_run", datetime.now().isoformat())
        self.data_manager.database.save_system_metadata("code_version", CODE_VERSION)
        self.data_manager.database.save_system_metadata("total_iterations", str(len(new_completed_lots) // 3))
        
        return manifest
    
    def _get_loocv_data_for_iteration(self, iteration_num: int) -> pd.DataFrame:
        """Get LOOCV data for a specific iteration"""
        try:
            # Load LOOCV predictions from file cache
            loocv_path = os.path.join(ROOT_DIR, "rolling", "loocv_predictions.csv")
            if os.path.exists(loocv_path):
                loocv_df = pd.read_csv(loocv_path)
                return loocv_df
            else:
                print(f"[main] LOOCV data not found at {loocv_path}")
                return pd.DataFrame()
        except Exception as e:
            print(f"[main] Error loading LOOCV data: {e}")
            return pd.DataFrame()
    
    def _get_uncertainties_for_iteration(self, iteration_num: int) -> Dict[str, List[float]]:
        """Get uncertainties for a specific iteration"""
        try:
            # For now, return placeholder uncertainties
            # In a real implementation, this would come from the uncertainty calculation
            uncertainties = {
                'rate': [29.98, 29.98, 29.98],  # Rate uncertainties for 3 recipes
                'range': [4.47, 4.47, 4.47]     # Range uncertainties for 3 recipes
            }
            return uncertainties
        except Exception as e:
            print(f"[main] Error getting uncertainties: {e}")
            return {'rate': [], 'range': []}
    
    def _get_metrics_for_iteration(self, iteration_num: int) -> Dict[str, float]:
        """Get metrics for a specific iteration"""
        try:
            # For now, return placeholder metrics
            # In a real implementation, this would come from model evaluation
            metrics = {
                'rmse': {'rate': 25.0, 'range': 3.5},
                'r2': {'rate': 0.85, 'range': 0.78},
                'mae': {'rate': 20.0, 'range': 2.8}
            }
            return metrics
        except Exception as e:
            print(f"[main] Error getting metrics: {e}")
            return {}
    
    def _create_metrics_over_time(self, df: pd.DataFrame, cache_key_str: str):
        """Create metrics_over_time.csv file for plotting"""
        print("[metrics] Creating metrics_over_time.csv for plotting")
        
        # Create a simple metrics file with basic structure
        # This is a simplified version - in production, this would use backtesting data
        metrics_data = []
        
        # Get unique dates from the dataset
        df_copy = df.copy()
        df_copy["run_date"] = pd.to_datetime(df_copy["run_date"])
        unique_dates = sorted(df_copy["run_date"].dropna().unique())
        
        for i, date in enumerate(unique_dates):
            # Calculate metrics for data up to this date
            data_up_to_date = df_copy[df_copy["run_date"] <= date]
            
            if len(data_up_to_date) > 0:
                # Simple metrics calculation
                rmse_rate = np.sqrt(np.mean((data_up_to_date["AvgEtchRate"] - data_up_to_date["AvgEtchRate"].mean())**2))
                rmse_range = np.sqrt(np.mean((data_up_to_date["RangeEtchRate"] - data_up_to_date["RangeEtchRate"].mean())**2))
                coverage_rate = 0.8  # Placeholder
                coverage_range = 0.8  # Placeholder
                
                metrics_data.append({
                    "train_end_date": date.strftime("%Y-%m-%d"),
                    "rmse_rate": rmse_rate,
                    "rmse_range": rmse_range,
                    "coverage_rate_1s": coverage_rate,
                    "coverage_range_1s": coverage_range,
                    "n_points_up_to_date": len(data_up_to_date)
                })
        
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            metrics_path = os.path.join(ROLLING_DIR, "metrics_over_time.csv")
            metrics_df.to_csv(metrics_path, index=False)
            print(f"[metrics] Created metrics file with {len(metrics_data)} data points")
        else:
            print("[metrics] No metrics data created")


def main():
    """Main entry point with command-line argument support"""
    parser = argparse.ArgumentParser(description='Pareto Optimization System')
    parser.add_argument('--gpr-hyperopt', 
                       choices=['true', 'false', 'yes', 'no', '1', '0'],
                       default=None,
                       help='Enable/disable GPR hyperparameter optimization (default: disabled)')
    parser.add_argument('--fast-mode', 
                       action='store_true',
                       help='Enable fast mode (disables GPR hyperparameter optimization)')
    
    args = parser.parse_args()
    
    # Override GPR hyperparameter optimization setting if specified
    if args.gpr_hyperopt is not None:
        enable_hyperopt = args.gpr_hyperopt.lower() in ['true', 'yes', '1']
        os.environ['DT_GPR_HYPEROPT'] = str(enable_hyperopt).lower()
        print(f"[main] GPR hyperparameter optimization {'enabled' if enable_hyperopt else 'disabled'} via command line")
    elif args.fast_mode:
        os.environ['DT_GPR_HYPEROPT'] = 'false'
        print("[main] Fast mode enabled - GPR hyperparameter optimization disabled")
    
    system = ParetoSystem()
    return system.run_main_optimization()


if __name__ == "__main__":
    main()


