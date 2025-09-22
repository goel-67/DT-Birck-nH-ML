"""
Command-line interface components for Pareto optimization.

This module contains:
- cli.py: Command-line interface for system interaction
- iteration_manager.py: Iteration data management and storage
"""

from .cli import CLI
from .iteration_manager import IterationManager

__all__ = ['CLI', 'IterationManager']
