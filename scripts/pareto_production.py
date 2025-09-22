#!/usr/bin/env python3
"""
Production script for Pareto optimization system.
Designed to run in cron job environment.
"""

import sys
import os
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pareto_optimization.log'),
        logging.StreamHandler()
    ]
)

def main():
    """Main execution function."""
    
    try:
        logging.info("Starting Pareto optimization system...")
        
        # Import and run the main system
        from src.core.main import main as pareto_main
        
        # Run the system
        pareto_main()
        
        logging.info("Pareto optimization system completed successfully")
        
    except Exception as e:
        logging.error(f"Error in Pareto optimization system: {e}")
        import traceback
        logging.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
