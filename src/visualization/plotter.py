"""
Plotting module for Pareto optimization system.
Handles all visualization functionality including Pareto fronts, parity plots, and metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from sklearn.metrics import r2_score, mean_squared_error
from typing import List, Tuple, Optional, Dict, Any
import os

from ..core.config import (
    HIGHLIGHT_COLORS, CIRCLED_NUMBERS, SUBSCRIPT_NUMBERS,
    EXCEL_PRED_RATE_COL, EXCEL_PRED_RANGE_COL, EXCEL_LOT_COL,
    EXCEL_STATUS_COL, EXCEL_DATE_COL, EXCEL_RATE_UNCERTAINTY_COL, EXCEL_RANGE_UNCERTAINTY_COL,
    PLOTS_DIR, ROLLING_DIR, ITERATIONS_DIR, POINTS_PER_ITERATION
)


class Plotter:
    """Handles all plotting functionality for the Pareto optimization system."""
    
    def __init__(self):
        # Set matplotlib backend to avoid display issues
        plt.switch_backend('Agg')
    
    def plot_parity_from_loocv(self, loocv_csv: str) -> Tuple[str, str]:
        """Create parity plots from LOOCV data"""
        df = pd.read_csv(loocv_csv)
        df["Date"] = pd.to_datetime(df["Date"])
        
        # Rate plot
        y = df["loo_true_rate"].values
        yp = df["loo_pred_rate"].values
        ys = df["loo_std_rate"].values
        r2r = r2_score(y, yp) if len(np.unique(y)) > 1 else np.nan
        rmse_r = np.sqrt(mean_squared_error(y, yp)) if len(np.unique(y)) > 1 else np.nan
        
        plt.figure(figsize=(7, 6))
        plt.errorbar(y, yp, yerr=ys, fmt='o', alpha=0.6, ecolor='gray', capsize=2, markersize=4)
        mn = min(y.min(), yp.min())
        mx = max(y.max(), yp.max())
        plt.plot([mn, mx], [mn, mx], 'k--', lw=1)
        plt.xlabel("Actual AvgEtchRate")
        plt.ylabel("Predicted AvgEtchRate")
        plt.title(f"LOOCV Parity — AvgEtchRate | R² = {r2r:.3f}, RMSE = {rmse_r:.3f}")
        plt.grid(True)
        plt.tight_layout()
        p1 = os.path.join(PLOTS_DIR, "parity_rate.png")
        plt.savefig(p1, dpi=160)
        plt.close()

        # Range plot
        yg = df["loo_true_range"].values
        ypg = df["loo_pred_range"].values
        ysg = df["loo_std_range"].values
        r2g = r2_score(yg, ypg) if len(np.unique(yg)) > 1 else np.nan
        rmse_g = np.sqrt(mean_squared_error(yg, ypg)) if len(np.unique(yg)) > 1 else np.nan
        
        plt.figure(figsize=(7, 6))
        plt.errorbar(yg, ypg, yerr=ysg, fmt='o', alpha=0.6, ecolor='gray', capsize=2, markersize=4)
        mn = min(yg.min(), ypg.min())
        mx = max(yg.max(), ypg.max())
        plt.plot([mn, mx], [mn, mx], 'k--', lw=1)
        plt.xlabel("Actual RangeEtchRate")
        plt.ylabel("Predicted RangeEtchRate")
        plt.title(f"LOOCV Parity — RangeEtchRate | R² = {r2g:.3f}, RMSE = {rmse_g:.3f}")
        plt.grid(True)
        plt.tight_layout()
        p2 = os.path.join(PLOTS_DIR, "parity_range.png")
        plt.savefig(p2, dpi=160)
        plt.close()
        
        return p1, p2
    
    def plot_parity_from_loocv_with_highlights(self, loocv_csv: str, highlight_lots: List[str], 
                                             recipes_df: Optional[pd.DataFrame] = None, 
                                             iteration_num: Optional[int] = None, 
                                             plots_dir: Optional[str] = None) -> Tuple[str, str]:
        """Enhanced LOOCV parity plots with highlighted lots"""
        df = pd.read_csv(loocv_csv)
        df["Date"] = pd.to_datetime(df["Date"])
        
        # Rate plot
        y = df["loo_true_rate"].values
        yp = df["loo_pred_rate"].values
        ys = df["loo_std_rate"].values
        r2r = r2_score(y, yp) if len(np.unique(y)) > 1 else np.nan
        
        plt.figure(figsize=(10, 8))
        
        # Plot all points (non-highlighted)
        non_highlight_mask = ~df["LOTNAME"].isin(highlight_lots)
        if non_highlight_mask.any():
            y_nh = y[non_highlight_mask]
            yp_nh = yp[non_highlight_mask]
            ys_nh = ys[non_highlight_mask]
            plt.errorbar(y_nh, yp_nh, yerr=ys_nh, fmt='o', alpha=0.6, ecolor='gray', 
                        capsize=4, markersize=8, label="Historical (LOOCV)", linewidth=2)
        
        # Highlight points using Excel predictions
        if highlight_lots and recipes_df is not None:
            for i, lot in enumerate(highlight_lots):
                if lot in df["LOTNAME"].values and lot in recipes_df[EXCEL_LOT_COL].values:
                    lot_mask = df["LOTNAME"] == lot
                    actual_rate = df.loc[lot_mask, "loo_true_rate"].iloc[0]
                    
                    excel_mask = recipes_df[EXCEL_LOT_COL] == lot
                    if excel_mask.any() and EXCEL_PRED_RATE_COL in recipes_df.columns:
                        pred_rate = recipes_df.loc[excel_mask, EXCEL_PRED_RATE_COL].iloc[0]
                        
                        color = HIGHLIGHT_COLORS[i % len(HIGHLIGHT_COLORS)]
                        plt.scatter(actual_rate, pred_rate, s=120, c=color, edgecolors='k', 
                                  linewidth=1.5, zorder=5, label=f"{lot} (Excel)")
        
        # Set plot limits and ideal line
        all_rates = np.concatenate([y, [pred_rate] if 'pred_rate' in locals() else []])
        mn = min(all_rates.min(), yp.min()) if len(all_rates) > 0 else yp.min()
        mx = max(all_rates.max(), yp.max()) if len(all_rates) > 0 else yp.max()
        plt.plot([mn, mx], [mn, mx], 'k--', lw=2, alpha=0.7)
        
        plt.xlabel("Actual AvgEtchRate (nm/min)", fontsize=16, weight='bold')
        plt.ylabel("Predicted AvgEtchRate (nm/min)", fontsize=16, weight='bold')
        
        rmse_r = np.sqrt(mean_squared_error(y, yp)) if len(np.unique(y)) > 1 else np.nan
        
        if iteration_num is not None:
            plt.title(f"Parity — AvgEtchRate | R² = {r2r:.3f}, RMSE = {rmse_r:.3f} - Iteration {iteration_num}", 
                     fontsize=18, weight='bold', pad=20)
        else:
            plt.title(f"Parity — AvgEtchRate | R² = {r2r:.3f}, RMSE = {rmse_r:.3f}", 
                     fontsize=18, weight='bold', pad=20)
        
        try:
            plt.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=14)
        except:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
        
        plt.grid(True, alpha=0.4, linewidth=1.5)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        
        target_plots_dir = plots_dir if plots_dir else PLOTS_DIR
        p1 = os.path.join(target_plots_dir, "parity_rate.png")
        plt.savefig(p1, dpi=160)
        plt.close()

        # Thickness Range plot
        yg = df["loo_true_range"].values * 5.0
        ypg = df["loo_pred_range"].values * 5.0
        ysg = df["loo_std_range"].values * 5.0
        r2g = r2_score(yg, ypg) if len(np.unique(yg)) > 1 else np.nan
        
        plt.figure(figsize=(10, 8))
        
        if non_highlight_mask.any():
            yg_nh = yg[non_highlight_mask]
            ypg_nh = ypg[non_highlight_mask]
            ysg_nh = ysg[non_highlight_mask]
            plt.errorbar(yg_nh, ypg_nh, yerr=ysg_nh, fmt='o', alpha=0.6, ecolor='gray', 
                        capsize=4, markersize=8, label="Historical (LOOCV)", linewidth=2)
        
        if highlight_lots and recipes_df is not None:
            for i, lot in enumerate(highlight_lots):
                if lot in df["LOTNAME"].values and lot in recipes_df[EXCEL_LOT_COL].values:
                    lot_mask = df["LOTNAME"] == lot
                    actual_range = df.loc[lot_mask, "loo_true_range"].iloc[0] * 5.0
                    
                    excel_mask = recipes_df[EXCEL_LOT_COL] == lot
                    if excel_mask.any() and EXCEL_PRED_RANGE_COL in recipes_df.columns:
                        pred_range = recipes_df.loc[excel_mask, EXCEL_PRED_RANGE_COL].iloc[0]
                        
                        color = HIGHLIGHT_COLORS[i % len(HIGHLIGHT_COLORS)]
                        plt.scatter(actual_range, pred_range, s=120, c=color, edgecolors='k', 
                                  linewidth=1.5, zorder=5, label=f"{lot} (Excel)")
        
        mn = min(yg.min(), ypg.min())
        mx = max(yg.max(), ypg.max())
        plt.plot([mn, mx], [mn, mx], 'k--', lw=2)
        
        plt.xlabel("Actual Thickness Range (nm)", fontsize=16, weight='bold')
        plt.ylabel("Predicted Thickness Range (nm)", fontsize=16, weight='bold')
        
        rmse_g = np.sqrt(mean_squared_error(yg, ypg)) if len(np.unique(yg)) > 1 else np.nan
        
        if iteration_num is not None:
            plt.title(f"Parity — Thickness Range | R² = {r2g:.3f}, RMSE = {rmse_g:.3f} - Iteration {iteration_num}", 
                     fontsize=18, weight='bold', pad=20)
        else:
            plt.title(f"Parity — Thickness Range | R² = {r2g:.3f}, RMSE = {rmse_g:.3f}", 
                     fontsize=18, weight='bold', pad=20)
        
        try:
            plt.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=14)
        except:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
        
        plt.grid(True, alpha=0.4, linewidth=1.5)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        
        p2 = os.path.join(target_plots_dir, "parity_range.png")
        plt.savefig(p2, dpi=160)
        plt.close()
        
        return p1, p2
    
    def plot_front(self, summary_df: pd.DataFrame, pareto_df: pd.DataFrame, 
                  selected_points: List[Tuple[float, float]], highlight_lots: List[str], 
                  new_completed_lots: Optional[List[str]] = None, recipes_df: Optional[pd.DataFrame] = None,
                  iteration_num: Optional[int] = None, plots_dir: Optional[str] = None,
                  selected_uncertainties: Optional[List[Tuple[float, float]]] = None) -> str:
        """Create Pareto front plot with selected points and highlights"""
        plt.figure(figsize=(12, 10))
        
        # Plot historical data
        plt.scatter(summary_df["AvgEtchRate"], summary_df["Range_nm"], s=80, edgecolor='k', alpha=0.6, label="Historical")
        
        # Plot Pareto front
        plt.plot(pareto_df["AvgEtchRate"], pareto_df["Range_nm"], 'r--', lw=2.5, label="Pareto front")
        plt.scatter(pareto_df["AvgEtchRate"], pareto_df["Range_nm"], s=100, facecolors='none', edgecolors='r', linewidth=1.5)
        
        # Plot new predicted recipes with symbols
        for i, (r, rng) in enumerate(selected_points):
            point_num = i + 1
            iter_num = iteration_num if iteration_num is not None else 1
            
            # Use Unicode symbols
            if point_num == 1:
                symbol = f"0{iter_num}"
            elif point_num == 2:
                symbol = f"1{iter_num}"
            elif point_num == 3:
                symbol = f"2{iter_num}"
            else:
                symbol = f"3{iter_num}"
            
            plt.scatter(r, rng, marker='o', s=300, c='gold', edgecolors='k', lw=2, 
                       label="Proposed Recipes" if i == 0 else "", zorder=10)
            
            plt.text(r, rng, symbol, color='black', fontsize=16, weight='bold', 
                    ha='center', va='center', zorder=11)
            
            # Add uncertainty bars if provided
            if selected_uncertainties and i < len(selected_uncertainties):
                rate_uncertainty, range_uncertainty = selected_uncertainties[i]
                
                plt.errorbar(r, rng, yerr=range_uncertainty, fmt='none', color='gold', alpha=0.7, 
                            capsize=5, capthick=2, elinewidth=2)
                plt.errorbar(r, rng, xerr=rate_uncertainty, fmt='none', color='gold', alpha=0.7, 
                            capsize=5, capthick=2, elinewidth=2)
        
        # Add prediction legend entry
        if highlight_lots:
            plt.scatter([], [], marker='x', s=150, c='gray', linewidth=2, label="Predicted Outcomes")
        
        # Plot new completed runs
        if new_completed_lots and recipes_df is not None:
            colors = ['#FF8C00', '#800080', '#228B22']
            
            for i, lot in enumerate(new_completed_lots):
                color = colors[i % len(colors)]
                has_actual_data = lot in summary_df["LOTNAME"].values
                
                recipe_info = None
                if lot in recipes_df[EXCEL_LOT_COL].values:
                    recipe_row = recipes_df[recipes_df[EXCEL_LOT_COL] == lot]
                    pred_rate = recipe_row[EXCEL_PRED_RATE_COL].iloc[0] if EXCEL_PRED_RATE_COL in recipe_row.columns else None
                    pred_range = recipe_row[EXCEL_PRED_RANGE_COL].iloc[0] if EXCEL_PRED_RANGE_COL in recipe_row.columns else None
                    
                    recipe_info = {
                        "pred_rate": pred_rate,
                        "pred_range": pred_range
                    }
                
                if has_actual_data and recipe_info and recipe_info["pred_rate"] is not None:
                    row = summary_df[summary_df["LOTNAME"] == lot]
                    r = row["AvgEtchRate"].iloc[0]
                    rng = row["Range_nm"].iloc[0]
                    
                    plt.scatter(r, rng, marker='o', s=80, c=color, edgecolors='k', lw=0.8, label=f"{lot} (Actual)")
                    plt.plot([recipe_info["pred_rate"], r], [recipe_info["pred_range"], rng], '--', color=color, alpha=0.7, lw=1)
                    plt.scatter(recipe_info["pred_rate"], recipe_info["pred_range"], marker='x', s=150, c=color, linewidth=2, alpha=0.8)
        
        # Plot highlighted lots
        for i, lot in enumerate(highlight_lots):
            color = HIGHLIGHT_COLORS[i % len(HIGHLIGHT_COLORS)]
            has_actual_data = lot in summary_df["LOTNAME"].values
            
            recipe_info = None
            if recipes_df is not None and lot in recipes_df[EXCEL_LOT_COL].values:
                recipe_row = recipes_df[recipes_df[EXCEL_LOT_COL] == lot]
                pred_rate = recipe_row[EXCEL_PRED_RATE_COL].iloc[0] if EXCEL_PRED_RATE_COL in recipe_row.columns else None
                pred_range = recipe_row[EXCEL_PRED_RANGE_COL].iloc[0] if EXCEL_PRED_RANGE_COL in recipe_row.columns else None
                
                recipe_info = {
                    "pred_rate": pred_rate,
                    "pred_range": pred_range
                }
            
            if has_actual_data and recipe_info and recipe_info["pred_rate"] is not None:
                row = summary_df[summary_df["LOTNAME"] == lot]
                r = row["AvgEtchRate"].iloc[0]
                rng = row["Range_nm"].iloc[0]
                
                plt.scatter(r, rng, marker='o', s=80, c=color, edgecolors='k', linewidth=0.8, label=f"{lot} (Actual)")
                plt.plot([recipe_info["pred_rate"], r], [recipe_info["pred_range"], rng], '--', color=color, alpha=0.7, lw=1)
                plt.scatter(recipe_info["pred_rate"], recipe_info["pred_range"], marker='x', s=100, c=color, linewidth=1, alpha=0.8)
        
        plt.xlabel("Average Etch Rate (nm/min)", fontsize=22, weight='bold')
        plt.ylabel("Thickness Range (nm)", fontsize=22, weight='bold')
        
        if iteration_num is not None:
            plt.title(f"Pareto + Selected — Top 3 (SOBOL sampling) - Iteration {iteration_num}", 
                     fontsize=24, weight='bold', pad=20)
        else:
            plt.title(f"Pareto + Selected — Top 3 (SOBOL sampling)", 
                     fontsize=24, weight='bold', pad=20)
        
        plt.legend(loc="upper left", fontsize=20)
        plt.grid(True, alpha=0.4, linewidth=1.5)
        plt.ylim(0, 25)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        
        target_plots_dir = plots_dir if plots_dir else PLOTS_DIR
        p = os.path.join(target_plots_dir, "front.png")
        plt.savefig(p, dpi=160)
        plt.close()
        
        return p
    
    def plot_metrics_over_time(self, csv_path: str, iteration_num: Optional[int] = None, 
                              plots_dir: Optional[str] = None) -> Optional[Tuple[str, str]]:
        """Create metrics plots over time"""
        df = pd.read_csv(csv_path)
        if df is None or len(df) == 0:
            print(f"[plots] No metrics data found at {csv_path}")
            return None
        
        df["train_end_date"] = pd.to_datetime(df["train_end_date"])
        df = df.sort_values("train_end_date")
        
        if len(df) < 2:
            print(f"[plots] Only {len(df)} data point(s) available, skipping metrics plots")
            return None
        
        print(f"[plots] Creating metrics plots with {len(df)} data points")
        
        target_plots_dir = plots_dir if plots_dir else PLOTS_DIR
        
        # RMSE plot with dual Y-axes
        fig, ax1 = plt.subplots(figsize=(14, 9))
        ax2 = ax1.twinx()
        
        # Calculate confidence intervals
        n_points = df["n_points_up_to_date"].values
        rmse_rate_se = df["rmse_rate"].values / np.sqrt(n_points)
        rmse_range_se = df["rmse_range"].values / np.sqrt(n_points)
        
        # Plot RMSE Rate on left Y-axis
        line1 = ax1.errorbar(df["train_end_date"], df["rmse_rate"], yerr=rmse_rate_se, 
                            fmt='o-', color='#1f77b4', label="RMSE Rate", linewidth=3, 
                            markersize=10, capsize=6, alpha=0.8)
        ax1.set_xlabel("Train end date", fontsize=16, weight='bold')
        ax1.set_ylabel("RMSE Rate", color='#1f77b4', fontsize=16, weight='bold')
        ax1.tick_params(axis='y', labelcolor='#1f77b4', labelsize=14)
        ax1.tick_params(axis='x', labelsize=14)
        
        # Plot RMSE Range on right Y-axis
        line2 = ax2.errorbar(df["train_end_date"], df["rmse_range"], yerr=rmse_range_se, 
                            fmt='s-', color='#ff7f0e', label="RMSE Range", linewidth=3, 
                            markersize=10, capsize=6, alpha=0.8)
        ax2.set_ylabel("RMSE Range", color='#ff7f0e', fontsize=16, weight='bold')
        ax2.tick_params(axis='y', labelcolor='#ff7f0e', labelsize=14)
        
        # Set Y-axis limits
        ax1.set_ylim(0, max(df["rmse_rate"].max() * 1.1, 20))
        ax2.set_ylim(0, max(df["rmse_range"].max() * 1.2, 2))
        
        # Highlight latest iteration
        if len(df) > 0:
            latest_date = df["train_end_date"].iloc[-1]
            latest_rate = df["rmse_rate"].iloc[-1]
            latest_range = df["rmse_range"].iloc[-1]
            
            ax1.scatter(latest_date, latest_rate, s=300, c='red', edgecolors='k', zorder=5, label="Latest")
            ax1.text(latest_date, latest_rate + max(rmse_rate_se[-1], 1.0), 
                    f"Latest (Iteration {iteration_num})" if iteration_num is not None else "Latest", 
                    color='red', fontsize=14, weight='bold', ha='center')
            
            ax2.scatter(latest_date, latest_range, s=300, c='red', edgecolors='k', zorder=5)
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=14)
        
        if iteration_num == 1:
            title = f"RMSE over time - Baseline Model Performance ({len(df)} data points)"
        elif iteration_num is not None:
            title = f"RMSE over time - Iteration {iteration_num} ({len(df)} data points)"
        else:
            title = f"RMSE over time ({len(df)} data points)"
        
        plt.title(title, fontsize=18, weight='bold', pad=20)
        ax1.grid(True, alpha=0.4, linewidth=1.5)
        plt.tight_layout()
        
        p1 = os.path.join(target_plots_dir, "metrics_rmse.png")
        plt.savefig(p1, dpi=160)
        plt.close()

        # Coverage plot
        plt.figure(figsize=(14, 8))
        
        coverage_rate_se = np.sqrt(df["coverage_rate_1s"].values * (1 - df["coverage_rate_1s"].values) / n_points)
        coverage_range_se = np.sqrt(df["coverage_range_1s"].values * (1 - df["coverage_range_1s"].values) / n_points)
        
        plt.errorbar(df["train_end_date"], df["coverage_rate_1s"], yerr=coverage_rate_se,
                    fmt='o-', label="Rate within 1σ", linewidth=3, markersize=10, capsize=6)
        plt.errorbar(df["train_end_date"], df["coverage_range_1s"], yerr=coverage_range_se,
                    fmt='s-', label="Range within 1σ", linewidth=3, markersize=10, capsize=6)
        
        if len(df) > 0:
            latest_coverage_rate = df["coverage_rate_1s"].iloc[-1]
            latest_coverage_range = df["coverage_range_1s"].iloc[-1]
            plt.scatter(latest_date, latest_coverage_rate, s=300, c='red', edgecolors='k', zorder=5)
            plt.scatter(latest_date, latest_coverage_range, s=300, c='red', edgecolors='k', zorder=5)
            plt.text(latest_date, latest_coverage_rate + max(coverage_rate_se[-1], 0.02), 
                    f"Latest (Iteration {iteration_num})" if iteration_num is not None else "Latest", 
                    color='red', fontsize=14, weight='bold', ha='center')
        
        plt.ylim(0, 1.05)
        plt.xlabel("Train end date", fontsize=16, weight='bold')
        plt.ylabel("Coverage", fontsize=16, weight='bold')
        
        if iteration_num == 1:
            title = f"Uncertainty Coverage over time - Baseline Model Performance ({len(df)} data points)"
        elif iteration_num is not None:
            title = f"Uncertainty Coverage over time - Iteration {iteration_num} ({len(df)} data points)"
        else:
            title = f"Uncertainty Coverage over time ({len(df)} data points)"
        
        plt.title(title, fontsize=18, weight='bold', pad=20)
        plt.legend(fontsize=14)
        plt.grid(True, alpha=0.4, linewidth=1.5)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        
        p2 = os.path.join(target_plots_dir, "metrics_coverage.png")
        plt.savefig(p2, dpi=160)
        plt.close()
        
        return p1, p2
    
    def get_symbol_for_point(self, point_num: int, iteration_num: int) -> str:
        """Generate proper Unicode symbol for a point in an iteration"""
        iter_idx = iteration_num - 1
        point_idx = point_num - 1
        
        if iter_idx < 0 or iter_idx >= len(CIRCLED_NUMBERS):
            return f"?{iteration_num}"
        if point_idx < 0 or point_idx >= len(SUBSCRIPT_NUMBERS):
            return f"{CIRCLED_NUMBERS[iter_idx]}?"
        
        return CIRCLED_NUMBERS[iter_idx] + SUBSCRIPT_NUMBERS[point_idx]

    def plot_metrics_over_time(self, metrics_csv: str) -> Tuple[str, str]:
        """Create metrics plots over time"""
        if not os.path.exists(metrics_csv):
            return None, None
            
        df = pd.read_csv(metrics_csv)
        
        # RMSE plot
        plt.figure(figsize=(10, 6))
        plt.plot(df['train_end_date'], df['rmse_rate'], 'o-', label='Rate RMSE', alpha=0.7)
        plt.plot(df['train_end_date'], df['rmse_range'], 's-', label='Range RMSE', alpha=0.7)
        plt.xlabel('Date')
        plt.ylabel('RMSE')
        plt.title('Model Performance Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        rmse_path = os.path.join(PLOTS_DIR, "metrics_rmse.png")
        plt.savefig(rmse_path, dpi=160)
        plt.close()
        
        # Coverage plot
        plt.figure(figsize=(10, 6))
        plt.plot(df['train_end_date'], df['coverage_rate_1s'], 'o-', label='Rate Coverage', alpha=0.7)
        plt.plot(df['train_end_date'], df['coverage_range_1s'], 's-', label='Range Coverage', alpha=0.7)
        plt.xlabel('Date')
        plt.ylabel('Coverage')
        plt.title('Model Coverage Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        coverage_path = os.path.join(PLOTS_DIR, "metrics_coverage.png")
        plt.savefig(coverage_path, dpi=160)
        plt.close()
        
        return rmse_path, coverage_path
    
    def _get_highlighted_lots_actual_data(self, df_summary: pd.DataFrame, highlight_lots: List[str]) -> List[float]:
        """Get actual range values for highlighted lots to determine y-axis limits"""
        actual_ranges = []
        for lot in highlight_lots:
            if lot in df_summary["LOTNAME"].values:
                row = df_summary[df_summary["LOTNAME"] == lot]
                actual_range = row["Range_nm"].iloc[0]
                if not np.isnan(actual_range) and not np.isinf(actual_range):
                    actual_ranges.append(actual_range)
        return actual_ranges
    
    def _get_training_points_count_for_iteration(self, iteration_num: int, training_count: int = None) -> int:
        """Get the actual number of training points for a specific iteration"""
        if training_count is not None:
            print(f"[plotter] Iteration {iteration_num}: Using provided training count = {training_count}")
            return training_count
        
        # No hardcoded fallback - this should not happen in normal operation
        print(f"[plotter] ERROR: No training count provided for iteration {iteration_num} - this should not happen!")
        return 112  # Default fallback only for error cases
    
    def _calculate_ylim_for_iteration(self, df_summary: pd.DataFrame, highlight_lots: List[str], default_ylim: float = 18.0) -> float:
        """Calculate y-axis limit based on actual data of highlighted lots"""
        actual_ranges = self._get_highlighted_lots_actual_data(df_summary, highlight_lots)
        
        if not actual_ranges:
            return default_ylim
        
        max_range = max(actual_ranges)
        
        # If max range is above default, add some padding
        if max_range > default_ylim:
            return max_range + 1.0  # Add 1 nm padding
        else:
            return default_ylim
    
    def plot_front_with_proposed_recipes(self, df_summary: pd.DataFrame, current_front: pd.DataFrame, 
                                       selected_points: List[Tuple[float, float]], selected_uncertainties: List[Tuple[float, float]], 
                                       highlight_lots: List[str], iteration_num: int, plots_dir: str, training_count: int = None) -> str:
        """Plot 1: Pareto front with 3 proposed recipes using ⓿₁ symbols with uncertainty bars - EXACT COPY FROM PARETO.PY"""
        plt.figure(figsize=(12, 10))
        
        # Plot historical data (no label needed)
        plt.scatter(df_summary["AvgEtchRate"], df_summary["Range_nm"], s=80, edgecolor='k', 
                   alpha=0.6)
        
        # Plot Pareto front
        plt.plot(current_front["AvgEtchRate"], current_front["Range_nm"], 'r--', lw=2.5, 
                 label="Pareto front")
        plt.scatter(current_front["AvgEtchRate"], current_front["Range_nm"], s=100, 
                   facecolors='none', edgecolors='r', linewidth=1.5)
        
        # Plot proposed recipes with ⓿₁ symbols
        if selected_points and len(selected_points) >= 3:
            # Use consistent colors from existing system
            highlight_colors = HIGHLIGHT_COLORS
            
            for i, (rate, range_nm) in enumerate(selected_points[:3]):
                # Check if coordinates are valid numbers (not nan or inf)
                if (isinstance(rate, (int, float)) and isinstance(range_nm, (int, float)) and
                    not np.isnan(rate) and not np.isnan(range_nm) and
                    not np.isinf(rate) and not np.isinf(range_nm)):
                    
                    color = highlight_colors[i % len(highlight_colors)]
                    
                    # Create the symbol: outer number = point number, inner number = iteration number
                    point_num = i + 1
                    symbol = self.get_symbol_for_point(point_num, iteration_num)
                    
                    # Add the symbol text directly on the data point for better visibility
                    plt.text(rate, range_nm, symbol, color=color, fontsize=32, weight='bold', 
                            ha='center', va='center', zorder=11,
                            path_effects=[pe.withStroke(linewidth=3, foreground="white")])
                    
                    # Add uncertainty bars if provided
                    if selected_uncertainties and i < len(selected_uncertainties):
                        rate_uncertainty, range_uncertainty = selected_uncertainties[i]
                        
                        # Check if uncertainties are valid numbers (not nan or inf)
                        if (isinstance(rate_uncertainty, (int, float)) and 
                            isinstance(range_uncertainty, (int, float)) and
                            not np.isnan(rate_uncertainty) and not np.isnan(range_uncertainty) and
                            not np.isinf(rate_uncertainty) and not np.isinf(range_uncertainty) and
                            rate_uncertainty > 0 and range_uncertainty > 0):
                            
                            # Vertical uncertainty bar for thickness range (y-axis) - more translucent
                            plt.errorbar(rate, range_nm, yerr=range_uncertainty, fmt='none', 
                                       color=color, alpha=0.3, capsize=8, capthick=3, elinewidth=3)
                            
                            # Horizontal uncertainty bar for etch rate (x-axis) - more translucent
                            plt.errorbar(rate, range_nm, xerr=rate_uncertainty, fmt='none', 
                                       color=color, alpha=0.3, capsize=8, capthick=3, elinewidth=3)
        
        # Set labels and title
        plt.xlabel("Average Etch Rate (nm/min)", fontsize=22, weight='bold')
        plt.ylabel("Thickness Range (nm)", fontsize=22, weight='bold')
        plt.title(f"Pareto Front with Proposed Recipes - Iteration {iteration_num}", 
                 fontsize=24, weight='bold', pad=20)
        
        # Track proposed points for dynamic axis adjustment
        proposed_points_rates = []
        proposed_points_ranges = []
        
        # Collect proposed points coordinates for axis adjustment
        if selected_points and len(selected_points) >= 3:
            for i, (rate, range_nm) in enumerate(selected_points[:3]):
                if (isinstance(rate, (int, float)) and isinstance(range_nm, (int, float)) and
                    not np.isnan(rate) and not np.isnan(range_nm) and
                    not np.isinf(rate) and not np.isinf(range_nm)):
                    proposed_points_rates.append(rate)
                    proposed_points_ranges.append(range_nm)
        
        # Calculate y-axis limit based on actual data
        ylim = self._calculate_ylim_for_iteration(df_summary, highlight_lots)
        
        # Dynamically adjust axis limits to include proposed points if they go outside bounds
        if proposed_points_rates and proposed_points_ranges:
            # Check if proposed points exceed the hard-set y-limit of 18
            y_exceeds_limit = any(y > 18 for y in proposed_points_ranges)
            
            # Also check if points go outside the natural data range
            max_data_x = df_summary["AvgEtchRate"].max()
            max_data_y = df_summary["Range_nm"].max()
            x_exceeds_data = any(x > max_data_x for x in proposed_points_rates)
            y_exceeds_data = any(y > max_data_y for y in proposed_points_ranges)
            
            if y_exceeds_limit or x_exceeds_data or y_exceeds_data:
                # Calculate new limits including proposed points
                all_x = list(df_summary["AvgEtchRate"]) + proposed_points_rates
                all_y = list(df_summary["Range_nm"]) + proposed_points_ranges
                
                # Add some padding (10% on each side)
                x_range = max(all_x) - min(all_x)
                y_range = max(all_y) - min(all_y)
                x_padding = x_range * 0.1
                y_padding = y_range * 0.1
                
                new_xlim = (min(all_x) - x_padding, max(all_x) + x_padding)
                new_ylim = (max(0, min(all_y) - y_padding), max(all_y) + y_padding)
                
                plt.xlim(new_xlim)
                plt.ylim(new_ylim)
        
        # Count training points for legend (consistent with debug files)
        training_points = self._get_training_points_count_for_iteration(iteration_num, training_count)
        previous_points = training_points  # Use exact total training points
        
        # Add custom legend with point counts
        from matplotlib.patches import Circle
        from matplotlib.text import Text
        
        # Create custom legend element for x inside circle, y outside
        class CustomLegendElement:
            def __init__(self, label):
                self.label = label
            
            def get_label(self):
                return self.label
            
            def legend_artist(self, legend, orig_handle, fontsize, handlebox):
                # Create a circle with x inside and y outside
                x, y = handlebox.xdescent, handlebox.ydescent
                width, height = handlebox.width, handlebox.height
                
                # Draw circle
                circle = Circle((x + width/2, y + height/2), min(width, height)/4, 
                              facecolor='white', edgecolor='black', linewidth=2)
                
                # Add text elements
                text_x = Text(x + width/2, y + height/2, 'x', ha='center', va='center', 
                             fontsize=fontsize, weight='bold')
                text_y = Text(x + width/2 + min(width, height)/3, y + height/2, 'y', 
                             ha='center', va='center', fontsize=fontsize, weight='bold')
                
                return [circle, text_x, text_y]
        
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                       markersize=15, label=f'{previous_points} previous points'),
            plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2.5, label='Pareto Front'),
            CustomLegendElement('x = iteration num, y = recipe num')
        ]
        
        # Removed "Iteration X points" entry from legend as requested
        
        # Legend and grid
        plt.legend(handles=legend_elements, loc="upper left", fontsize=16)
        plt.grid(True, alpha=0.4, linewidth=1.5)
        
        # Only set hard limit if no dynamic adjustment was made
        if not (proposed_points_rates and proposed_points_ranges and 
                (any(y > 18 for y in proposed_points_ranges) or 
                 any(x > df_summary["AvgEtchRate"].max() for x in proposed_points_rates) or 
                 any(y > df_summary["Range_nm"].max() for y in proposed_points_ranges))):
            plt.ylim(0, 18)  # Hard-set y-axis limit to 18
        
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(plots_dir, "1_pareto_proposed.png")
        plt.savefig(plot_path, dpi=160)
        plt.close()
        
        return plot_path
    
    def plot_parity_with_horizontal_lines(self, iteration_num: int, recipes_df: pd.DataFrame, 
                                        selected_points: List[Tuple[float, float]], selected_uncertainties: List[Tuple[float, float]], 
                                        highlight_lots: List[str], plots_dir: str) -> Tuple[str, str]:
        """Plots 2 & 3: Parity plots with horizontal lines for proposed points"""
        if not selected_points or len(selected_points) < 3:
            print(f"[comprehensive_plots] No selected points for parity plots in iteration {iteration_num}")
            return None, None
        
        # Get LOOCV data for this iteration
        loocv_data = self._get_loocv_data_for_iteration(iteration_num, recipes_df)
        if loocv_data is None:
            print(f"[comprehensive_plots] No LOOCV data available for iteration {iteration_num}")
            return None, None
        
        # Use consistent colors
        highlight_colors = HIGHLIGHT_COLORS
        
        # Plot 2: Etch Rate Parity with horizontal lines
        plt.figure(figsize=(10, 7))
        
        # Plot historical LOOCV data (no label needed)
        plt.errorbar(loocv_data["loo_true_rate"], loocv_data["loo_pred_rate"], 
                    yerr=loocv_data["loo_std_rate"], fmt='o', alpha=0.6, ecolor='gray', 
                    capsize=3, markersize=8, linewidth=1)
        
        # Add x=y reference line (no label needed)
        min_val = min(loocv_data["loo_true_rate"].min(), loocv_data["loo_pred_rate"].min())
        max_val = max(loocv_data["loo_true_rate"].max(), loocv_data["loo_pred_rate"].max())
        plt.plot([max(0, min_val), max_val], [max(0, min_val), max_val], 'k--', alpha=0.5, linewidth=1.5)
        
        # Set better axis limits for cleaner scaling (less empty space)
        margin = (max_val - min_val) * 0.05
        plt.xlim(max(0, min_val - margin), max_val + margin)  # Ensure x-axis starts at 0 or positive
        plt.ylim(max(0, min_val - margin), max_val + margin)  # Ensure y-axis starts at 0 or positive
        
        # Add horizontal lines for proposed points
        for i, (rate, range_nm) in enumerate(selected_points[:3]):
            color = highlight_colors[i % len(highlight_colors)]
            point_num = i + 1
            
            symbol = self.get_symbol_for_point(point_num, iteration_num)
            
            # Get predicted rate from selected points
            pred_rate = rate
            
            # Add horizontal line extending full width
            plt.axhline(y=pred_rate, color=color, linestyle='--', alpha=0.7, linewidth=2)
            
            # Add very translucent uncertainty lines above and below
            if selected_uncertainties and i < len(selected_uncertainties):
                rate_uncertainty = selected_uncertainties[i][0]
                
                # Check if uncertainty is a valid number
                if (isinstance(rate_uncertainty, (int, float)) and 
                    not np.isnan(rate_uncertainty) and not np.isinf(rate_uncertainty) and
                    rate_uncertainty > 0):
                    
                    # Upper uncertainty line (slightly more visible)
                    plt.axhline(y=pred_rate + rate_uncertainty, color=color, linestyle='-', alpha=0.25, linewidth=1)
                    # Lower uncertainty line (slightly more visible)
                    plt.axhline(y=pred_rate - rate_uncertainty, color=color, linestyle='-', alpha=0.25, linewidth=1)
            
            # Add symbol label in the right whitespace (increased spacing from right edge)
            plt.text(plt.xlim()[1] * 0.90, pred_rate, symbol, color=color, 
                    fontsize=32, weight='bold', ha='right', va='center',
                    path_effects=[pe.withStroke(linewidth=3, foreground="white")])
        
        # Calculate R² and RMSE scores using ONLY baseline historical data (not experimental points)
        # This ensures fair comparison across iterations
        baseline_mask = ~loocv_data["LOTNAME"].isin(highlight_lots) if highlight_lots else pd.Series([True] * len(loocv_data))
        baseline_data = loocv_data[baseline_mask]
        
        if len(baseline_data) > 0:
            r2 = r2_score(baseline_data["loo_true_rate"], baseline_data["loo_pred_rate"])
            rmse = np.sqrt(mean_squared_error(baseline_data["loo_true_rate"], baseline_data["loo_pred_rate"]))
        else:
            r2 = np.nan
            rmse = np.nan
        
        # Add clean legend with R² and RMSE inside the plot (like legend in Pareto front)
        # Use same positioning logic as Pareto front legend for neat top-left placement
        plt.text(0.02, 0.98, f"R² = {r2:.2f}\nRMSE = {rmse:.2f}", 
                 fontsize=18, weight='bold', transform=plt.gca().transAxes,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9, edgecolor='black', linewidth=1),
                 verticalalignment='top')
        
        # Set plot properties
        plt.xlabel("Actual AvgEtchRate (nm/min)", fontsize=22, weight='bold')
        plt.ylabel("Predicted AvgEtchRate (nm/min)", fontsize=22, weight='bold')
        plt.title(f"Etch Rate Parity with Proposed Points - Iteration {iteration_num}", 
                 fontsize=20, weight='bold', pad=25)
        plt.grid(True, alpha=0.1, linewidth=0.5)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.gca().set_aspect(1/1.7, adjustable='box')  # Make internal plot rectangular (1:1.7 ratio - height >> width)
        plt.tight_layout(pad=2.0)  # Increased padding for better margins
        
        # Save plot
        rate_plot_path = os.path.join(plots_dir, "2_parity_rate_horizontal.png")
        plt.savefig(rate_plot_path, dpi=160)
        plt.close()
        
        # Plot 3: Range Parity with horizontal lines
        plt.figure(figsize=(10, 7))
        
        # Plot historical LOOCV data (convert to thickness range, no label needed)
        plt.errorbar(loocv_data["loo_true_range"] * 5.0, loocv_data["loo_pred_range"] * 5.0, 
                    yerr=loocv_data["loo_std_range"] * 5.0, fmt='o', alpha=0.6, ecolor='gray', 
                    capsize=2, markersize=6, linewidth=0.5)
        
        # Add x=y reference line (no label needed)
        min_val = min((loocv_data["loo_true_range"] * 5.0).min(), (loocv_data["loo_pred_range"] * 5.0).min())
        max_val = max((loocv_data["loo_true_range"] * 5.0).max(), (loocv_data["loo_pred_range"] * 5.0).max())
        plt.plot([max(0, min_val), max_val], [max(0, min_val), max_val], 'k--', alpha=0.5, linewidth=1.5)
        
        # Add horizontal lines for proposed points
        for i, (rate, range_nm) in enumerate(selected_points[:3]):
            color = highlight_colors[i % len(highlight_colors)]
            point_num = i + 1
            
            symbol = self.get_symbol_for_point(point_num, iteration_num)
            
            # Get predicted range from selected points (already in nm)
            pred_range = range_nm
            
            # Add horizontal line extending full width
            plt.axhline(y=pred_range, color=color, linestyle='--', alpha=0.7, linewidth=2)
            
            # Add very translucent uncertainty lines above and below
            if selected_uncertainties and i < len(selected_uncertainties):
                range_uncertainty = selected_uncertainties[i][1]
                
                # Check if uncertainty is a valid number
                if (isinstance(range_uncertainty, (int, float)) and 
                    not np.isnan(range_uncertainty) and not np.isinf(range_uncertainty) and
                    range_uncertainty > 0):
                    
                    # Upper uncertainty line (slightly more visible)
                    plt.axhline(y=pred_range + range_uncertainty, color=color, linestyle='-', alpha=0.25, linewidth=1)
                    # Lower uncertainty line (slightly more visible)
                    plt.axhline(y=pred_range - range_uncertainty, color=color, linestyle='-', alpha=0.25, linewidth=1)
            
            # Add symbol label in the right whitespace (increased spacing from right edge)
            plt.text(plt.xlim()[1] * 0.90, pred_range, symbol, color=color, 
                    fontsize=32, weight='bold', ha='right', va='center',
                    path_effects=[pe.withStroke(linewidth=3, foreground="white")])
        
        # Calculate R² and RMSE scores using ONLY baseline historical data (not experimental points)
        # This ensures fair comparison across iterations
        baseline_mask = ~loocv_data["LOTNAME"].isin(highlight_lots) if highlight_lots else pd.Series([True] * len(loocv_data))
        baseline_data = loocv_data[baseline_mask]
        
        if len(baseline_data) > 0:
            r2 = r2_score(baseline_data["loo_true_range"] * 5.0, baseline_data["loo_pred_range"] * 5.0)
            rmse = np.sqrt(mean_squared_error(baseline_data["loo_true_range"] * 5.0, baseline_data["loo_pred_range"] * 5.0))
        else:
            r2 = np.nan
            rmse = np.nan
        
        # Add clean legend with R² and RMSE inside the plot (like legend in Pareto front)
        # Use same positioning logic as Pareto front legend for neat top-left placement
        plt.text(0.02, 0.98, f"R² = {r2:.2f}\nRMSE = {rmse:.2f}", 
                 fontsize=18, weight='bold', transform=plt.gca().transAxes,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9, edgecolor='black', linewidth=1),
                 verticalalignment='top')
        
        # Set plot properties
        plt.xlabel("Actual Thickness Range (nm)", fontsize=22, weight='bold')
        plt.ylabel("Predicted Thickness Range (nm)", fontsize=22, weight='bold')
        plt.title(f"Thickness Range Parity with Proposed Points - Iteration {iteration_num}", 
                 fontsize=20, weight='bold', pad=25)
        plt.grid(True, alpha=0.1, linewidth=0.5)
        plt.xticks(fontsize=18)
        plt.gca().set_aspect(1/1.7, adjustable='box')  # Make internal plot rectangular (1:1.7 ratio - height >> width)
        plt.tight_layout(pad=2.0)  # Increased padding for better margins
        
        # Save plot
        range_plot_path = os.path.join(plots_dir, "3_parity_range_horizontal.png")
        plt.savefig(range_plot_path, dpi=160)
        plt.close()
        
        return rate_plot_path, range_plot_path
    
    def plot_front_with_predicted_and_actual(self, df_summary: pd.DataFrame, current_front: pd.DataFrame, 
                                           selected_points: List[Tuple[float, float]], selected_uncertainties: List[Tuple[float, float]], 
                                           highlight_lots: List[str], recipes_df: pd.DataFrame, iteration_num: int, plots_dir: str, training_count: int = None) -> str:
        """Plot 4: Pareto front with both predicted (faded) and actual (full opacity) points"""
        plt.figure(figsize=(12, 10))
        
        # Plot ALL historical data (always show the full dataset)
        plt.scatter(df_summary["AvgEtchRate"], df_summary["Range_nm"], s=80, edgecolor='k', 
                   alpha=0.6)
        
        # Plot Pareto front
        plt.plot(current_front["AvgEtchRate"], current_front["Range_nm"], 'r--', lw=2.5, 
                 label="Pareto front")
        plt.scatter(current_front["AvgEtchRate"], current_front["Range_nm"], s=100, 
                   facecolors='none', edgecolors='r', linewidth=1.5)
        
        # Plot predicted points (faded/translucent)
        if selected_points and len(selected_points) >= 3:
            highlight_colors = HIGHLIGHT_COLORS
            
            for i, (rate, range_nm) in enumerate(selected_points[:3]):
                # Check if coordinates are valid numbers (not nan, inf, or string)
                if (isinstance(rate, (int, float)) and isinstance(range_nm, (int, float)) and
                    not np.isnan(rate) and not np.isnan(range_nm) and
                    not np.isinf(rate) and not np.isinf(range_nm)):
                    
                    color = highlight_colors[i % len(highlight_colors)]
                    point_num = i + 1
                    
                    symbol = self.get_symbol_for_point(point_num, iteration_num)
                    
                    # Plot only the Unicode symbol for predicted point (faded)
                    # plt.scatter removed - only symbol text remains
                    
                    # Add symbol text with path effects for better visibility (more faded)
                    plt.text(rate, range_nm, symbol, color=color, fontsize=32, weight='bold', 
                            ha='center', va='center', zorder=11, alpha=0.4,
                            path_effects=[pe.withStroke(linewidth=3, foreground="white")])
                    
                    # Add uncertainty bars (faded) if valid
                    if selected_uncertainties and i < len(selected_uncertainties):
                        rate_uncertainty, range_uncertainty = selected_uncertainties[i]
                        
                        # Check if uncertainties are valid numbers
                        if (isinstance(rate_uncertainty, (int, float)) and 
                            isinstance(range_uncertainty, (int, float)) and
                            not np.isnan(rate_uncertainty) and not np.isnan(range_uncertainty) and
                            not np.isinf(rate_uncertainty) and not np.isinf(range_uncertainty) and
                            rate_uncertainty > 0 and range_uncertainty > 0):
                            
                            plt.errorbar(rate, range_nm, yerr=range_uncertainty, fmt='none', 
                                       color=color, alpha=0.3, capsize=5, capthick=2, elinewidth=2)
                            plt.errorbar(rate, range_nm, xerr=rate_uncertainty, fmt='none', 
                                       color=color, alpha=0.3, capsize=5, capthick=2, elinewidth=2)

        
        # Plot actual points (full opacity) if available
        if highlight_lots and recipes_df is not None:
            # Separate current iteration points from previous iteration points
            current_iter_lots = []
            previous_iter_lots = []
            
            # Get current iteration lots
            current_iter_recipes = self._get_proposed_recipes_for_iteration(recipes_df, iteration_num)
            if current_iter_recipes:
                for recipe in current_iter_recipes:
                    status = recipe.get("status", "")
                    if isinstance(status, str) and status.lower() == "completed":
                        lotname = recipe.get("lotname", "")
                        if lotname and isinstance(lotname, str) and lotname.strip():
                            current_iter_lots.append(lotname)
            
            # Get previous iteration lots
            if iteration_num > 1:
                previous_iter_recipes = self._get_proposed_recipes_for_iteration(recipes_df, iteration_num - 1)
                if previous_iter_recipes:
                    for recipe in previous_iter_recipes:
                        status = recipe.get("status", "")
                        if isinstance(status, str) and status.lower() == "completed":
                            lotname = recipe.get("lotname", "")
                            if lotname and isinstance(lotname, str) and lotname.strip():
                                previous_iter_lots.append(lotname)
            
            # Plot current iteration points (full opacity, full size, colored)
            for i, lot in enumerate(current_iter_lots):
                if lot in df_summary["LOTNAME"].values:
                    row = df_summary[df_summary["LOTNAME"] == lot]
                    actual_rate = row["AvgEtchRate"].iloc[0]
                    actual_range = row["Range_nm"].iloc[0]
                    
                    # Get the correct point number for this lot
                    point_num = i + 1
                    symbol = self.get_symbol_for_point(point_num, iteration_num)
                    
                    # Use highlight colors for current iteration
                    color = highlight_colors[i % len(highlight_colors)]
                    plt.text(actual_rate, actual_range, symbol, color=color, fontsize=32, weight='bold', 
                            ha='center', va='center', zorder=11,
                            path_effects=[pe.withStroke(linewidth=3, foreground="white")])
            
            # Plot previous iteration points (faded, smaller, black)
            for i, lot in enumerate(previous_iter_lots):
                if lot in df_summary["LOTNAME"].values:
                    row = df_summary[df_summary["LOTNAME"] == lot]
                    actual_rate = row["AvgEtchRate"].iloc[0]
                    actual_range = row["Range_nm"].iloc[0]
                    
                    # Get the correct point number for this lot
                    point_num = i + 1
                    symbol = self.get_symbol_for_point(point_num, iteration_num - 1)
                    
                    # Use black color, smaller size, and faded for previous iteration
                    plt.text(actual_rate, actual_range, symbol, color='black', fontsize=24, weight='bold', 
                            ha='center', va='center', zorder=10, alpha=0.6,
                            path_effects=[pe.withStroke(linewidth=2, foreground="white")])
        
        # Track actual points for dynamic axis adjustment (current AND previous iterations)
        actual_points_rates = []
        actual_points_ranges = []
        
        # Collect actual points coordinates for axis adjustment
        if highlight_lots and recipes_df is not None:
            # Current iteration points
            current_iter_recipes = self._get_proposed_recipes_for_iteration(recipes_df, iteration_num)
            if current_iter_recipes:
                for recipe in current_iter_recipes:
                    status = recipe.get("status", "")
                    if isinstance(status, str) and status.lower() == "completed":
                        lotname = recipe.get("lotname", "")
                        if lotname and isinstance(lotname, str) and lotname.strip():
                            if lotname in df_summary["LOTNAME"].values:
                                row = df_summary[df_summary["LOTNAME"] == lotname]
                                actual_rate = row["AvgEtchRate"].iloc[0]
                                actual_range = row["Range_nm"].iloc[0]
                                actual_points_rates.append(actual_rate)
                                actual_points_ranges.append(actual_range)
            
            # Previous iteration points (for dynamic axis adjustment)
            if iteration_num > 1:
                previous_iter_recipes = self._get_proposed_recipes_for_iteration(recipes_df, iteration_num - 1)
                if previous_iter_recipes:
                    for recipe in previous_iter_recipes:
                        status = recipe.get("status", "")
                        if isinstance(status, str) and status.lower() == "completed":
                            lotname = recipe.get("lotname", "")
                            if lotname and isinstance(lotname, str) and lotname.strip():
                                if lotname in df_summary["LOTNAME"].values:
                                    row = df_summary[df_summary["LOTNAME"] == lotname]
                                    actual_rate = row["AvgEtchRate"].iloc[0]
                                    actual_range = row["Range_nm"].iloc[0]
                                    actual_points_rates.append(actual_rate)
                                    actual_points_ranges.append(actual_range)
        
        # Calculate y-axis limit based on actual data
        ylim = self._calculate_ylim_for_iteration(df_summary, highlight_lots)
        
        # Always set appropriate axis limits based on data
        max_data_x = df_summary["AvgEtchRate"].max()
        max_data_y = df_summary["Range_nm"].max()
        min_data_x = df_summary["AvgEtchRate"].min()
        min_data_y = df_summary["Range_nm"].min()
        
        # Dynamically adjust axis limits to include actual points if they go outside bounds
        if actual_points_rates and actual_points_ranges:
            # Check if actual points exceed the hard-set y-limit of 18
            y_exceeds_limit = any(y > 18 for y in actual_points_ranges)
            
            # Also check if points go outside the natural data range
            x_exceeds_data = any(x > max_data_x for x in actual_points_rates)
            y_exceeds_data = any(y > max_data_y for y in actual_points_ranges)
            
            if y_exceeds_limit or x_exceeds_data or y_exceeds_data:
                # Calculate new limits including actual points
                all_x = list(df_summary["AvgEtchRate"]) + actual_points_rates
                all_y = list(df_summary["Range_nm"]) + actual_points_ranges
                
                # Add some padding (10% on each side)
                x_range = max(all_x) - min(all_x)
                y_range = max(all_y) - min(all_y)
                x_padding = x_range * 0.1
                y_padding = y_range * 0.1
                
                new_xlim = (min(all_x) - x_padding, max(all_x) + x_padding)
                new_ylim = (max(0, min(all_y) - y_padding), max(all_y) + y_padding)
                
                plt.xlim(new_xlim)
                plt.ylim(new_ylim)
            else:
                # Set limits based on data range with small padding
                x_padding = (max_data_x - min_data_x) * 0.05
                plt.xlim(max(0, min_data_x - x_padding), max_data_x + x_padding)
        else:
            # Set limits based on data range with small padding
            x_padding = (max_data_x - min_data_x) * 0.05
            plt.xlim(max(0, min_data_x - x_padding), max_data_x + x_padding)
        
        # Count training points for legend (consistent with debug files)
        training_points = self._get_training_points_count_for_iteration(iteration_num, training_count)
        previous_points = training_points  # Use exact total training points
        
        # Set labels and title
        plt.xlabel("Average Etch Rate (nm/min)", fontsize=22, weight='bold')
        plt.ylabel("Thickness Range (nm)", fontsize=22, weight='bold')
        plt.title(f"Pareto Front with Predicted and Actual Points - Iteration {iteration_num}", 
                 fontsize=24, weight='bold', pad=20)
        
        # Create legend with point counts
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                       markersize=15, label=f'{previous_points} previous points'),
            plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2.5, label='Pareto Front'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                       markersize=15, label='Predicted points (faded)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', 
                       markersize=15, label='Current iteration actual points')
        ]
        
        # Removed "Iteration X points" entry from legend as requested
        
        # Legend and grid
        plt.legend(handles=legend_elements, loc="upper left", fontsize=16)
        plt.grid(True, alpha=0.4, linewidth=1.5)
        
        # Only set hard limit if no dynamic adjustment was made
        if not (actual_points_rates and actual_points_ranges and 
                (any(y > 18 for y in actual_points_ranges) or 
                 any(x > df_summary["AvgEtchRate"].max() for x in actual_points_rates) or 
                 any(y > df_summary["Range_nm"].max() for y in actual_points_ranges))):
            plt.ylim(0, 18)  # Hard-set y-axis limit to 18
        
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(plots_dir, "4_pareto_predicted_actual.png")
        plt.savefig(plot_path, dpi=160)
        plt.close()
        
        return plot_path
    
    def plot_parity_with_actual_points(self, iteration_num: int, recipes_df: pd.DataFrame, 
                                      selected_points: List[Tuple[float, float]], selected_uncertainties: List[Tuple[float, float]], 
                                      highlight_lots: List[str], plots_dir: str, full_dataset: pd.DataFrame = None) -> Tuple[str, str]:
        """Plots 5 & 6: Parity plots with actual points instead of horizontal lines"""
        if not selected_points or len(selected_points) < 3:
            print(f"[comprehensive_plots] No selected points for parity plots in iteration {iteration_num}")
            return None, None
        
        # Get LOOCV data for this iteration
        loocv_data = self._get_loocv_data_for_iteration(iteration_num, recipes_df)
        if loocv_data is None:
            print(f"[comprehensive_plots] No LOOCV data available for iteration {iteration_num}")
            return None, None
        
        # Use consistent colors
        highlight_colors = HIGHLIGHT_COLORS
        
        # Plot 5: Etch Rate Parity with actual points
        plt.figure(figsize=(10, 7))
        
        # Plot historical LOOCV data (no label needed)
        plt.errorbar(loocv_data["loo_true_rate"], loocv_data["loo_pred_rate"], 
                    yerr=loocv_data["loo_std_rate"], fmt='o', alpha=0.6, ecolor='gray', 
                    capsize=3, markersize=8, linewidth=1)
        
        # Add x=y reference line (no label needed)
        min_val = min(loocv_data["loo_true_rate"].min(), loocv_data["loo_pred_rate"].min())
        max_val = max(loocv_data["loo_true_rate"].max(), loocv_data["loo_pred_rate"].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=1.5)
        
        # Track highlighted points for dynamic axis adjustment
        highlighted_actual_rates = []
        highlighted_pred_rates = []
        
        # Add highlighted points for current iteration only
        if highlight_lots and recipes_df is not None:
            # Get current iteration completed lots only
            current_iter_lots = []
            current_iter_recipes = self._get_proposed_recipes_for_iteration(recipes_df, iteration_num)
            if current_iter_recipes:
                for recipe in current_iter_recipes:
                    status = recipe.get("status", "")
                    if isinstance(status, str) and status.lower() == "completed":
                        lotname = recipe.get("lotname", "")
                        if lotname and isinstance(lotname, str) and lotname.strip():
                            current_iter_lots.append(lotname)
            
            # Plot current iteration points with predicted values from Excel and actual values from dataset
            for i, lot in enumerate(current_iter_lots):
                # Get predicted values from SharePoint Excel
                lot_recipe = None
                for recipe in current_iter_recipes:
                    if recipe.get("lotname", "") == lot:
                        lot_recipe = recipe
                        break
                
                if lot_recipe and full_dataset is not None:
                    # Get predicted values from Excel
                    pred_rate = lot_recipe.get("predicted_rate", None)
                    pred_range = lot_recipe.get("predicted_range", None)
                    
                    # Get actual values from full dataset
                    lot_data = full_dataset[full_dataset["LOTNAME"] == lot]
                    if len(lot_data) > 0:
                        actual_rate = lot_data["AvgEtchRate"].iloc[0]
                        actual_range = lot_data["RangeEtchRate"].iloc[0] * 5.0  # Convert to thickness range
                        
                        if pred_rate is not None and actual_rate is not None:
                            # Collect highlighted points for axis adjustment
                            highlighted_actual_rates.append(actual_rate)
                            highlighted_pred_rates.append(pred_rate)
                            
                            # Get the correct point number for this lot
                            point_num = i + 1
                            symbol = self.get_symbol_for_point(point_num, iteration_num)
                            
                            # Use highlight colors for current iteration
                            color = highlight_colors[i % len(highlight_colors)]
                            plt.text(actual_rate, pred_rate, symbol, color=color, fontsize=32, 
                                    weight='bold', ha='center', va='center', zorder=6,
                                    path_effects=[pe.withStroke(linewidth=3, foreground="white")])
        
        # Dynamically adjust axis limits to include highlighted points if they go outside bounds
        if highlighted_actual_rates and highlighted_pred_rates:
            # Get current axis limits
            xlim = plt.xlim()
            ylim = plt.ylim()
            
            # Calculate new limits including highlighted points
            all_x = list(loocv_data["loo_true_rate"]) + highlighted_actual_rates
            all_y = list(loocv_data["loo_pred_rate"]) + highlighted_pred_rates
            
            # Check if highlighted points are outside current bounds
            x_out_of_bounds = any(x < xlim[0] or x > xlim[1] for x in highlighted_actual_rates)
            y_out_of_bounds = any(y < ylim[0] or y > ylim[1] for y in highlighted_pred_rates)
            
            if x_out_of_bounds or y_out_of_bounds:
                # Add some padding (10% on each side)
                x_range = max(all_x) - min(all_x)
                y_range = max(all_y) - min(all_y)
                x_padding = x_range * 0.1
                y_padding = y_range * 0.1
                
                new_xlim = (min(all_x) - x_padding, max(all_x) + x_padding)
                new_ylim = (min(all_y) - y_padding, max(all_y) + y_padding)
                
                plt.xlim(new_xlim)
                plt.ylim(new_ylim)
                
                # Redraw the reference line with new limits
                min_val = min(new_xlim[0], new_ylim[0])
                max_val = max(new_xlim[1], new_ylim[1])
                plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=1.5, zorder=1)
        
        # Calculate R² and RMSE scores using ONLY baseline historical data (not experimental points)
        # This ensures fair comparison across iterations
        baseline_mask = ~loocv_data["LOTNAME"].isin(highlight_lots) if highlight_lots else pd.Series([True] * len(loocv_data))
        baseline_data = loocv_data[baseline_mask]
        
        if len(baseline_data) > 0:
            r2 = r2_score(baseline_data["loo_true_rate"], baseline_data["loo_pred_rate"])
            rmse = np.sqrt(mean_squared_error(baseline_data["loo_true_rate"], baseline_data["loo_pred_rate"]))
        else:
            r2 = np.nan
            rmse = np.nan
        
        # Add clean legend with R² and RMSE inside the plot (like legend in Pareto front)
        # Use same positioning logic as Pareto front legend for neat top-left placement
        plt.text(0.02, 0.98, f"R² = {r2:.2f}\nRMSE = {rmse:.2f}", 
                 fontsize=18, weight='bold', transform=plt.gca().transAxes,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9, edgecolor='black', linewidth=1),
                 verticalalignment='top')
        
        # Set plot properties
        plt.xlabel("Actual AvgEtchRate (nm/min)", fontsize=22, weight='bold')
        plt.ylabel("Predicted AvgEtchRate (nm/min)", fontsize=22, weight='bold')
        plt.title(f"Etch Rate Parity with Actual Points - Iteration {iteration_num}", 
                 fontsize=20, weight='bold', pad=25)
        plt.grid(True, alpha=0.1, linewidth=0.5)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.gca().set_aspect(1/1.7, adjustable='box')  # Make internal plot rectangular (1:1.7 ratio - height >> width)
        plt.tight_layout(pad=2.0)  # Increased padding for better margins
        
        # Save plot
        rate_plot_path = os.path.join(plots_dir, "5_parity_rate_actual.png")
        plt.savefig(rate_plot_path, dpi=160)
        plt.close()
        
        # Plot 6: Range Parity with actual points
        plt.figure(figsize=(10, 7))
        
        # Plot historical LOOCV data (convert to thickness range, no label needed)
        plt.errorbar(loocv_data["loo_true_range"] * 5.0, loocv_data["loo_pred_range"] * 5.0, 
                    yerr=loocv_data["loo_std_range"] * 5.0, fmt='o', alpha=0.6, ecolor='gray', 
                    capsize=2, markersize=6, linewidth=0.5)
        
        # Add x=y reference line (no label needed)
        min_val = min((loocv_data["loo_true_range"] * 5.0).min(), (loocv_data["loo_pred_range"] * 5.0).min())
        max_val = max((loocv_data["loo_true_range"] * 5.0).max(), (loocv_data["loo_pred_range"] * 5.0).max())
        plt.plot([max(0, min_val), max_val], [max(0, min_val), max_val], 'k--', alpha=0.5, linewidth=1.5)
        
        # Track highlighted points for dynamic axis adjustment
        highlighted_actual_ranges = []
        highlighted_pred_ranges = []
        
        # Add highlighted points for current iteration only
        if highlight_lots and recipes_df is not None:
            # Get current iteration completed lots only
            current_iter_lots = []
            current_iter_recipes = self._get_proposed_recipes_for_iteration(recipes_df, iteration_num)
            if current_iter_recipes:
                for recipe in current_iter_recipes:
                    status = recipe.get("status", "")
                    if isinstance(status, str) and status.lower() == "completed":
                        lotname = recipe.get("lotname", "")
                        if lotname and isinstance(lotname, str) and lotname.strip():
                            current_iter_lots.append(lotname)
            
            # Plot current iteration points with predicted values from Excel and actual values from dataset
            for i, lot in enumerate(current_iter_lots):
                # Get predicted values from SharePoint Excel
                lot_recipe = None
                for recipe in current_iter_recipes:
                    if recipe.get("lotname", "") == lot:
                        lot_recipe = recipe
                        break
                
                if lot_recipe and full_dataset is not None:
                    # Get predicted values from Excel
                    pred_rate = lot_recipe.get("predicted_rate", None)
                    pred_range = lot_recipe.get("predicted_range", None)
                    
                    # Get actual values from full dataset
                    lot_data = full_dataset[full_dataset["LOTNAME"] == lot]
                    if len(lot_data) > 0:
                        actual_rate = lot_data["AvgEtchRate"].iloc[0]
                        actual_range = lot_data["RangeEtchRate"].iloc[0] * 5.0  # Convert to thickness range
                        
                        if pred_range is not None and actual_range is not None:
                            # Collect highlighted points for axis adjustment
                            highlighted_actual_ranges.append(actual_range)
                            highlighted_pred_ranges.append(pred_range)
                            
                            # Get the correct point number for this lot
                            point_num = i + 1
                            symbol = self.get_symbol_for_point(point_num, iteration_num)
                            
                            # Use highlight colors for current iteration
                            color = highlight_colors[i % len(highlight_colors)]
                            plt.text(actual_range, pred_range, symbol, color=color, fontsize=32, 
                                    weight='bold', ha='center', va='center', zorder=6,
                                    path_effects=[pe.withStroke(linewidth=3, foreground="white")])
        
        # Dynamically adjust axis limits to include highlighted points if they go outside bounds
        if highlighted_actual_ranges and highlighted_pred_ranges:
            # Get current axis limits
            xlim = plt.xlim()
            ylim = plt.ylim()
            
            # Check if highlighted points are outside current bounds
            x_out_of_bounds = any(x < xlim[0] or x > xlim[1] for x in highlighted_actual_ranges)
            y_out_of_bounds = any(y < ylim[0] or y > ylim[1] for y in highlighted_pred_ranges)
            
            if x_out_of_bounds or y_out_of_bounds:
                # Only adjust the axis limits to accommodate the highlighted points
                # Don't change the overall scale of the plot
                min_x = min(xlim[0], min(highlighted_actual_ranges))
                max_x = max(xlim[1], max(highlighted_actual_ranges))
                min_y = min(ylim[0], min(highlighted_pred_ranges))
                max_y = max(ylim[1], max(highlighted_pred_ranges))
                
                # Add small padding (5% on each side)
                x_range = max_x - min_x
                y_range = max_y - min_y
                x_padding = x_range * 0.05
                y_padding = y_range * 0.05
                
                new_xlim = (min_x - x_padding, max_x + x_padding)
                new_ylim = (min_y - y_padding, max_y + y_padding)
                
                plt.xlim(new_xlim)
                plt.ylim(new_ylim)
                
                # Redraw the reference line with new limits
                min_val = min(new_xlim[0], new_ylim[0])
                max_val = max(new_xlim[1], new_ylim[1])
                plt.plot([max(0, min_val), max_val], [max(0, min_val), max_val], 'k--', alpha=0.5, linewidth=1.5, zorder=1)
        
        # Calculate R² and RMSE scores using ONLY baseline historical data (not experimental points)
        # This ensures fair comparison across iterations
        baseline_mask = ~loocv_data["LOTNAME"].isin(highlight_lots) if highlight_lots else pd.Series([True] * len(loocv_data))
        baseline_data = loocv_data[baseline_mask]
        
        if len(baseline_data) > 0:
            r2 = r2_score(baseline_data["loo_true_range"] * 5.0, baseline_data["loo_pred_range"] * 5.0)
            rmse = np.sqrt(mean_squared_error(baseline_data["loo_true_range"] * 5.0, baseline_data["loo_pred_range"] * 5.0))
        else:
            r2 = np.nan
            rmse = np.nan
        
        # Add clean legend with R² and RMSE inside the plot (like legend in Pareto front)
        # Use same positioning logic as Pareto front legend for neat top-left placement
        plt.text(0.02, 0.98, f"R² = {r2:.2f}\nRMSE = {rmse:.2f}", 
                 fontsize=18, weight='bold', transform=plt.gca().transAxes,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9, edgecolor='black', linewidth=1),
                 verticalalignment='top')
        
        # Set plot properties
        plt.xlabel("Actual Thickness Range (nm)", fontsize=22, weight='bold')
        plt.ylabel("Predicted Thickness Range (nm)", fontsize=22, weight='bold')
        plt.title(f"Thickness Range Parity with Actual Points - Iteration {iteration_num}", 
                 fontsize=20, weight='bold', pad=25)
        plt.grid(True, alpha=0.1, linewidth=0.5)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.gca().set_aspect(1/1.7, adjustable='box')  # Make internal plot rectangular (1:1.7 ratio - height >> width)
        plt.tight_layout(pad=2.0)  # Increased padding for better margins
        
        # Save plot
        range_plot_path = os.path.join(plots_dir, "6_parity_range_actual.png")
        plt.savefig(range_plot_path, dpi=160)
        plt.close()
        
        return rate_plot_path, range_plot_path
    
    def plot_metrics_for_iteration(self, iteration_num: int, plots_dir: str) -> Tuple[str, str]:
        """Plot 7 & 8: Metrics plots"""
        metrics_csv = os.path.join(ROLLING_DIR, "metrics_over_time.csv")
        if not os.path.exists(metrics_csv):
            return None, None
            
        df = pd.read_csv(metrics_csv)
        
        # RMSE plot
        plt.figure(figsize=(10, 6))
        plt.plot(df['train_end_date'], df['rmse_rate'], 'o-', label='Rate RMSE', alpha=0.7)
        plt.plot(df['train_end_date'], df['rmse_range'], 's-', label='Range RMSE', alpha=0.7)
        plt.xlabel('Date')
        plt.ylabel('RMSE')
        plt.title(f'Iteration {iteration_num}: Model Performance Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        rmse_path = os.path.join(plots_dir, "metrics_rmse.png")
        plt.savefig(rmse_path, dpi=160)
        plt.close()
        
        # Coverage plot
        plt.figure(figsize=(10, 6))
        plt.plot(df['train_end_date'], df['coverage_rate_1s'], 'o-', label='Rate Coverage', alpha=0.7)
        plt.plot(df['train_end_date'], df['coverage_range_1s'], 's-', label='Range Coverage', alpha=0.7)
        plt.xlabel('Date')
        plt.ylabel('Coverage')
        plt.title(f'Iteration {iteration_num}: Model Coverage Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        coverage_path = os.path.join(plots_dir, "metrics_coverage.png")
        plt.savefig(coverage_path, dpi=160)
        plt.close()
        
        return rmse_path, coverage_path
    
    def plot_debug_pareto_with_hypothetical_completion(self, df_summary: pd.DataFrame, current_front: pd.DataFrame, 
                                                      selected_points: List[Tuple[float, float]], 
                                                      iteration_num: int, plots_dir: str) -> str:
        """
        Plot 9: DEBUG - Pareto front with hypothetical completion
        
        This plot shows what the Pareto front would look like if the proposed recipes
        were perfect predictions and became part of the actual dataset. This helps
        validate if the Pareto logic is working correctly.
        
        Args:
            df_summary: Historical data summary
            current_front: Current Pareto front
            selected_points: Proposed recipes for this iteration
            iteration_num: Current iteration number
            plots_dir: Directory to save the plot
        """
        plt.figure(figsize=(12, 10))
        
        # Plot historical data (light blue, no label)
        plt.scatter(df_summary["AvgEtchRate"], df_summary["Range_nm"], s=80, edgecolor='k', 
                   alpha=0.6, color='lightblue', label="Historical data")
        
        # Plot current Pareto front (red dashed line)
        plt.plot(current_front["AvgEtchRate"], current_front["Range_nm"], 'r--', lw=2.5, 
                 label="Current Pareto front")
        plt.scatter(current_front["AvgEtchRate"], current_front["Range_nm"], s=100, 
                   color='red', edgecolor='k', alpha=0.8, zorder=3)
        
        # Create hypothetical dataset by adding proposed recipes to historical data
        if selected_points and len(selected_points) > 0:
            # Extract proposed recipe data
            proposed_rates = []
            proposed_ranges = []
            
            for i, recipe in enumerate(selected_points):
                # Handle both tuple format (rate, range) and dict format
                if isinstance(recipe, tuple):
                    rate, range_nm = recipe
                else:
                    rate = recipe.get("Pred_avg_etch_rate", recipe.get("rate", 0))
                    range_nm = recipe.get("Range_nm", recipe.get("range_nm", 0))
                
                if rate > 0 and range_nm > 0:
                    proposed_rates.append(rate)
                    proposed_ranges.append(range_nm)
            
            if proposed_rates:
                # Plot proposed recipes as if they were actual data (large, colored circles)
                colors = ['purple', 'orange', 'green']
                for i, (rate, range_nm) in enumerate(zip(proposed_rates, proposed_ranges)):
                    color = colors[i % len(colors)]
                    symbol = self.get_symbol_for_point(i + 1, iteration_num)
                    
                    plt.scatter(rate, range_nm, s=200, color=color, edgecolor='k', 
                               alpha=0.9, zorder=5, label=f"Proposed recipe {i+1}")
                    
                    # Add symbol annotation
                    plt.text(rate, range_nm, symbol, color='white', fontsize=20, 
                            weight='bold', ha='center', va='center', zorder=6,
                            path_effects=[pe.withStroke(linewidth=2, foreground="black")])
                
                print(f"[debug_plot] Processing {len(proposed_rates)} proposed recipes for hypothetical Pareto front")
                
                # Calculate NEW Pareto front including proposed recipes
                # Create combined dataset
                combined_data = df_summary[["AvgEtchRate", "Range_nm"]].copy()
                proposed_data = pd.DataFrame({
                    "AvgEtchRate": proposed_rates,
                    "Range_nm": proposed_ranges
                })
                combined_data = pd.concat([combined_data, proposed_data], ignore_index=True)
                
                print(f"[debug_plot] Combined dataset: {len(combined_data)} total points (historical + proposed)")
                
                # Calculate new Pareto front
                new_front = self._pareto_front(combined_data)
                
                print(f"[debug_plot] New Pareto front has {len(new_front)} points")
                
                # Plot NEW Pareto front (blue dashed line, thicker)
                plt.plot(new_front["AvgEtchRate"], new_front["Range_nm"], 'b--', lw=3.5, 
                         label="Hypothetical Pareto front (with proposed recipes)")
                plt.scatter(new_front["AvgEtchRate"], new_front["Range_nm"], s=120, 
                           color='blue', edgecolor='k', alpha=0.9, zorder=4)
                
                # Add improvement analysis
                improvement_text = []
                for i, (rate, range_nm) in enumerate(zip(proposed_rates, proposed_ranges)):
                    # Check if this point dominates any existing Pareto front points
                    dominates = self._dominates_existing((rate, range_nm), 
                                                     current_front[["AvgEtchRate", "Range_nm"]].values)
                    if dominates:
                        improvement_text.append(f"Recipe {i+1}: PUSHES front forward ✅")
                    else:
                        improvement_text.append(f"Recipe {i+1}: Fills gap only ⚠️")
                
                # Add improvement summary to plot
                if improvement_text:
                    improvement_str = "\n".join(improvement_text)
                    plt.text(0.02, 0.02, f"Improvement Analysis:\n{improvement_str}", 
                             fontsize=14, weight='bold', transform=plt.gca().transAxes,
                             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', 
                                      alpha=0.9, edgecolor='black', linewidth=1),
                             verticalalignment='bottom')
        
        # Set plot properties
        plt.xlabel("Average Etch Rate (nm/min)", fontsize=22, weight='bold')
        plt.ylabel("Thickness Range (nm)", fontsize=22, weight='bold')
        plt.title(f"DEBUG: Pareto Front with Hypothetical Completion - Iteration {iteration_num}", 
                 fontsize=20, weight='bold', pad=25)
        plt.grid(True, alpha=0.3, linewidth=0.5)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.legend(fontsize=16, loc='upper left', bbox_to_anchor=(0.02, 0.98))
        plt.tight_layout(pad=2.0)
        
        # Save plot
        debug_plot_path = os.path.join(plots_dir, "9_debug_pareto_hypothetical.png")
        plt.savefig(debug_plot_path, dpi=160)
        plt.close()
        
        print(f"[comprehensive_plots] Created debug plot: {debug_plot_path}")
        return debug_plot_path

    def _get_loocv_data_for_iteration(self, iteration_num: int, recipes_df: pd.DataFrame):
        """Get LOOCV data for a specific iteration with progressive training"""
        try:
            # For iteration 1, use iteration-specific LOOCV if available, otherwise global
            iter_dir = os.path.join(ITERATIONS_DIR, f"iteration_{iteration_num}")
            loocv_path = os.path.join(iter_dir, "loocv_predictions.csv")
            
            if os.path.exists(loocv_path):
                print(f"[comprehensive_plots] Using iteration-specific LOOCV for iteration {iteration_num}")
                return pd.read_csv(loocv_path)
            else:
                # Fallback to global LOOCV
                global_loocv_path = os.path.join(ROLLING_DIR, "loocv_predictions.csv")
                if os.path.exists(global_loocv_path):
                    print(f"[comprehensive_plots] Using global LOOCV for iteration {iteration_num}")
                    return pd.read_csv(global_loocv_path)
            
            return None
        except Exception as e:
            print(f"[comprehensive_plots] Error loading LOOCV data: {e}")
            return None

    def _get_proposed_recipes_for_iteration(self, recipes_df: pd.DataFrame, iteration_num: int):
        """Get proposed recipes for a specific iteration"""
        try:
            if recipes_df is None or iteration_num < 1:
                return []
            
            # Calculate iteration based on row position (every 3 rows = 1 iteration)
            # For iteration 1, read rows 0,1,2; for iteration 2, read rows 3,4,5; etc.
            start_idx = (iteration_num - 1) * POINTS_PER_ITERATION
            end_idx = start_idx + POINTS_PER_ITERATION
            
            if start_idx >= len(recipes_df):
                return []
            
            iter_recipes = recipes_df.iloc[start_idx:end_idx]
            
            if len(iter_recipes) == 0:
                return []
            
            # Extract the recipe data in the same format as data_manager
            proposed_recipes = []
            for i, (_, recipe) in enumerate(iter_recipes.iterrows()):
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
        except Exception as e:
            print(f"[comprehensive_plots] Error getting proposed recipes: {e}")
            return []

    def _dominates_existing(self, point: Tuple[float, float], existing_points: np.ndarray) -> bool:
        """Check if a point dominates any existing Pareto front points"""
        rate, range_nm = point
        
        for existing_point in existing_points:
            existing_rate, existing_range = existing_point
            
            # Check if the new point dominates the existing point
            # (better or equal in both objectives, and strictly better in at least one)
            if (rate >= existing_rate and range_nm <= existing_range and 
                (rate > existing_rate or range_nm < existing_range)):
                return True
        
        return False

    def _pareto_front(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Pareto front from data"""
        try:
            # Sort by etch rate (ascending) and range (ascending)
            sorted_data = data.sort_values(['AvgEtchRate', 'Range_nm'])
            
            pareto_points = []
            for _, point in sorted_data.iterrows():
                rate = point['AvgEtchRate']
                range_nm = point['Range_nm']
                
                # Check if this point is dominated by any existing Pareto point
                dominated = False
                for pareto_point in pareto_points:
                    # A point is dominated if there exists another point with
                    # higher-or-equal rate (better) and lower-or-equal range (better),
                    # with at least one strict improvement
                    if (pareto_point['AvgEtchRate'] >= rate and 
                        pareto_point['Range_nm'] <= range_nm and
                        (pareto_point['AvgEtchRate'] > rate or pareto_point['Range_nm'] < range_nm)):
                        dominated = True
                        break
                
                if not dominated:
                    # Remove any existing Pareto points that this point dominates
                    pareto_points = [p for p in pareto_points if not (
                        rate >= p['AvgEtchRate'] and range_nm <= p['Range_nm'] and
                        (rate > p['AvgEtchRate'] or range_nm < p['Range_nm'])
                    )]
                    pareto_points.append({'AvgEtchRate': rate, 'Range_nm': range_nm})
            
            # Sort Pareto front by etch rate
            pareto_df = pd.DataFrame(pareto_points)
            if len(pareto_df) > 0:
                pareto_df = pareto_df.sort_values('AvgEtchRate')
            
            return pareto_df
        except Exception as e:
            print(f"[comprehensive_plots] Error calculating Pareto front: {e}")
            return pd.DataFrame()
