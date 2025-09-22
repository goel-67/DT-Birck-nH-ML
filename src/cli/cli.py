"""
Command-line interface for Pareto optimization system.
Provides the same interface as pareto.py with all command-line options.
"""

import argparse
import os
import sys
from typing import Optional

from ..core.config import SNAPSHOTS_DIR, ROLLING_DIR, PLOTS_DIR, ITERATIONS_DIR
from ..data.data_manager import DataManager
from ..visualization.plotter import Plotter


class CLI:
    """Command-line interface for the Pareto optimization system."""
    
    def __init__(self):
        self.data_manager = DataManager()
        self.plotter = Plotter()
    
    def view_plots(self, target_date: Optional[str] = None):
        """View plots from a specific date or the latest available date"""
        if target_date is None:
            # Interactive prompt for date
            print("\nAvailable snapshot dates:")
            if not os.path.exists(SNAPSHOTS_DIR):
                print("No snapshots directory found")
                return
            
            available_dates = [d for d in os.listdir(SNAPSHOTS_DIR) if os.path.isdir(os.path.join(SNAPSHOTS_DIR, d))]
            if not available_dates:
                print("No snapshots found")
                return
            
            available_dates.sort()
            for i, date in enumerate(available_dates, 1):
                print(f"  {i}. {date}")
            
            try:
                choice = input(f"\nEnter date number (1-{len(available_dates)}) or date (YYYY-MM-DD): ").strip()
                
                # Check if it's a number
                if choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(available_dates):
                        target_date = available_dates[idx]
                    else:
                        print("Invalid number selection")
                        return
                else:
                    # Assume it's a date string
                    target_date = choice
                    
            except (KeyboardInterrupt, EOFError):
                print("\nCancelled")
                return
        
        self._view_historical_plots(target_date)
    
    def view_latest(self):
        """View the latest available plots"""
        latest_date = self._get_latest_snapshot_date()
        if latest_date:
            print(f"Viewing latest plots from: {latest_date}")
            self._view_historical_plots(latest_date)
        else:
            print("No snapshots found")
    
    def view_iteration(self, iteration_num: int):
        """View plots for a specific iteration"""
        iter_dir = os.path.join(ITERATIONS_DIR, f"iteration_{iteration_num}")
        if not os.path.exists(iter_dir):
            print(f"Iteration {iteration_num} not found")
            return
        
        plots_dir = os.path.join(iter_dir, "plots")
        if not os.path.exists(plots_dir):
            print(f"No plots found for iteration {iteration_num}")
            return
        
        print(f"Viewing plots for iteration {iteration_num}...")
        # This would open the plots in the default image viewer
        # For now, just list the available plots
        plots = [f for f in os.listdir(plots_dir) if f.endswith('.png')]
        for plot in plots:
            print(f"  {plot}")
    
    def list_iterations(self):
        """List all available iterations"""
        if not os.path.exists(ITERATIONS_DIR):
            print("No iterations directory found")
            return
        
        iterations = [d for d in os.listdir(ITERATIONS_DIR) if d.startswith('iteration_')]
        if not iterations:
            print("No iterations found")
            return
        
        iterations.sort(key=lambda x: int(x.split('_')[1]))
        print("\nAvailable iterations:")
        for iteration in iterations:
            iter_num = iteration.split('_')[1]
            iter_dir = os.path.join(ITERATIONS_DIR, iteration)
            plots_dir = os.path.join(iter_dir, "plots")
            
            if os.path.exists(plots_dir):
                plot_count = len([f for f in os.listdir(plots_dir) if f.endswith('.png')])
                print(f"  Iteration {iter_num}: {plot_count} plots")
            else:
                print(f"  Iteration {iter_num}: no plots")
    
    def clear_cache(self):
        """Clear all cached data"""
        for f in os.listdir(ROLLING_DIR):
            if f.endswith('.csv'):
                os.remove(os.path.join(ROLLING_DIR, f))
        for f in os.listdir(PLOTS_DIR):
            if f.endswith('.png'):
                os.remove(os.path.join(PLOTS_DIR, f))
        print("Cache cleared")
    
    def clear_backtesting_cache(self):
        """Clear only the backtesting cache to force rebuild"""
        backtest_path = os.path.join(ROLLING_DIR, "predictions_by_date.csv")
        if os.path.exists(backtest_path):
            os.remove(backtest_path)
            print("Backtesting cache cleared - will rebuild on next run")
        else:
            print("No backtesting cache found")
    
    def clear_iteration_cache(self):
        """Clear all iteration cache data"""
        if not os.path.exists(ITERATIONS_DIR):
            print("No iterations directory found")
            return
        
        import shutil
        shutil.rmtree(ITERATIONS_DIR)
        os.makedirs(ITERATIONS_DIR, exist_ok=True)
        print("Iteration cache cleared")
    
    def regenerate_iterations(self):
        """Regenerate all iterations with comprehensive plotting system"""
        print("[regenerate] Starting regeneration of all iterations...")
        
        # This would need to be implemented with the full iteration system
        # For now, just print a message
        print("[regenerate] Regeneration not yet implemented in modular version")
        print("[regenerate] Use the original pareto.py for full iteration regeneration")
    
    def _get_latest_snapshot_date(self) -> Optional[str]:
        """Get the latest snapshot date"""
        if not os.path.exists(SNAPSHOTS_DIR):
            return None
        
        available_dates = [d for d in os.listdir(SNAPSHOTS_DIR) if os.path.isdir(os.path.join(SNAPSHOTS_DIR, d))]
        if not available_dates:
            return None
        
        available_dates.sort()
        return available_dates[-1]
    
    def _view_historical_plots(self, target_date: str):
        """View historical plots from a specific date"""
        target_dir = os.path.join(SNAPSHOTS_DIR, target_date)
        if not os.path.exists(target_dir):
            print(f"Date {target_date} not found")
            return
        
        plots_dir = os.path.join(target_dir, "plots")
        if not os.path.exists(plots_dir):
            print(f"No plots found for date {target_date}")
            return
        
        print(f"Viewing plots from {target_date}...")
        plots = [f for f in os.listdir(plots_dir) if f.endswith('.png')]
        for plot in plots:
            print(f"  {plot}")
        
        # This would open the plots in the default image viewer
        # For now, just list them


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Pareto optimization script with enhanced plotting and iteration system"
    )
    parser.add_argument("--date", type=str, help="View plots from specific date (YYYY-MM-DD)")
    parser.add_argument("--last", action="store_true", help="View latest available plots")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode to select date for viewing plots")
    parser.add_argument("--clear-cache", action="store_true", help="Clear all cached data")
    parser.add_argument("--clear-backtesting", action="store_true", help="Clear only backtesting cache to force rebuild")
    parser.add_argument("--iteration", type=int, help="View plots for specific iteration number")
    parser.add_argument("--list-iterations", action="store_true", help="List all available iterations")
    parser.add_argument("--clear-iterations", action="store_true", help="Clear all iteration cache data")
    parser.add_argument("--regenerate-iterations", action="store_true", help="Regenerate all iterations with comprehensive plotting system")
    
    args = parser.parse_args()
    
    cli = CLI()
    
    if args.clear_cache:
        cli.clear_cache()
    elif args.clear_backtesting:
        cli.clear_backtesting_cache()
    elif args.date:
        cli.view_plots(args.date)
    elif args.last:
        cli.view_latest()
    elif args.interactive:
        cli.view_plots()  # Will prompt interactively
    elif args.iteration is not None:
        cli.view_iteration(args.iteration)
    elif args.list_iterations:
        cli.list_iterations()
    elif args.clear_iterations:
        cli.clear_iteration_cache()
    elif args.regenerate_iterations:
        cli.regenerate_iterations()
    else:
        # Default: run main optimization
        from main import main as run_main
        run_main()


if __name__ == "__main__":
    main()
