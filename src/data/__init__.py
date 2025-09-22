"""
Data management components for Pareto optimization.

This module contains:
- data_manager.py: Data loading, processing, and validation
- excel_manager.py: Excel file handling and recipe management
- dataset_new.py: Database connectivity and data extraction
"""

# Import only what's needed to avoid circular imports
from .excel_manager import ExcelManager

__all__ = ['ExcelManager']
