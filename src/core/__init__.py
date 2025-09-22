"""
Core system components for Pareto optimization.

This module contains:
- main.py: Main entry point and system orchestration
- config.py: Configuration and environment settings
"""

from .main import ParetoSystem, main
from .config import *

__all__ = ['ParetoSystem', 'main']
