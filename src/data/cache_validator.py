"""
Cache Validation System for Pareto optimization.
Ensures data integrity and consistency between file cache and SQLite database.
"""

import os
import hashlib
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from .database_manager import DatabaseManager
from ..core.config import CACHE_DIRECTORY, CODE_VERSION


class CacheValidator:
    """Validates cache integrity and consistency"""
    
    def __init__(self, database_manager: DatabaseManager):
        self.db_manager = database_manager
        self.cache_dir = CACHE_DIRECTORY
        self.validation_results = {}
    
    def validate_cache_integrity(self) -> Dict[str, Any]:
        """Comprehensive cache validation"""
        print("[validation] Starting cache integrity validation...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'unknown',
            'checks': {},
            'errors': [],
            'warnings': []
        }
        
        # Run all validation checks
        checks = [
            self._validate_file_cache_structure,
            self._validate_database_structure,
            self._validate_data_consistency,
            self._validate_iteration_completeness,
            self._validate_data_freshness,
            self._validate_file_hashes
        ]
        
        for check_func in checks:
            try:
                check_name = check_func.__name__.replace('_validate_', '').replace('_', ' ')
                print(f"[validation] Running {check_name}...")
                check_result = check_func()
                results['checks'][check_name] = check_result
                
                if not check_result['passed']:
                    results['errors'].extend(check_result.get('errors', []))
                if check_result.get('warnings'):
                    results['warnings'].extend(check_result['warnings'])
                    
            except Exception as e:
                error_msg = f"Validation check {check_func.__name__} failed: {str(e)}"
                results['errors'].append(error_msg)
                results['checks'][check_name] = {
                    'passed': False,
                    'error': error_msg
                }
        
        # Determine overall status
        if results['errors']:
            results['overall_status'] = 'failed'
        elif results['warnings']:
            results['overall_status'] = 'warning'
        else:
            results['overall_status'] = 'passed'
        
        print(f"[validation] Cache validation {results['overall_status']}")
        return results
    
    def _validate_file_cache_structure(self) -> Dict[str, Any]:
        """Validate file cache directory structure"""
        result = {'passed': True, 'errors': [], 'warnings': []}
        
        required_dirs = [
            'iterations',
            'rolling',
            'snapshots',
            'database_exports'
        ]
        
        for dir_name in required_dirs:
            dir_path = os.path.join(self.cache_dir, dir_name)
            if not os.path.exists(dir_path):
                result['errors'].append(f"Required directory missing: {dir_path}")
                result['passed'] = False
            elif not os.path.isdir(dir_path):
                result['errors'].append(f"Path exists but is not a directory: {dir_path}")
                result['passed'] = False
        
        # Check for manifest file
        manifest_path = os.path.join(self.cache_dir, 'manifest.json')
        if not os.path.exists(manifest_path):
            result['warnings'].append("Manifest file missing - cache may be incomplete")
        
        return result
    
    def _validate_database_structure(self) -> Dict[str, Any]:
        """Validate SQLite database structure"""
        result = {'passed': True, 'errors': [], 'warnings': []}
        
        try:
            db_info = self.db_manager.get_database_info()
            
            # Check if database exists and has expected tables
            if not os.path.exists(self.db_manager.db_path):
                result['errors'].append("SQLite database file missing")
                result['passed'] = False
                return result
            
            expected_tables = [
                'iterations', 'highlight_lots', 'pareto_fronts', 
                'model_predictions', 'system_metadata', 'historical_data',
                'training_data_snapshots', 'processing_logs', 'data_versioning'
            ]
            
            for table in expected_tables:
                if table not in db_info['tables']:
                    result['errors'].append(f"Database table missing: {table}")
                    result['passed'] = False
                elif db_info['tables'][table] == 0:
                    result['warnings'].append(f"Database table empty: {table}")
            
            # Check database size
            if db_info['database_size_mb'] < 0.01:  # Less than 10KB
                result['warnings'].append("Database file is very small - may be empty")
                
        except Exception as e:
            result['errors'].append(f"Database validation failed: {str(e)}")
            result['passed'] = False
        
        return result
    
    def _validate_data_consistency(self) -> Dict[str, Any]:
        """Validate consistency between file cache and database"""
        result = {'passed': True, 'errors': [], 'warnings': []}
        
        try:
            # Get iterations from database
            db_iterations = self.db_manager.get_all_iterations()
            
            # Get iterations from file cache
            file_iterations = self._get_file_cache_iterations()
            
            # Compare iteration counts
            if len(db_iterations) != len(file_iterations):
                result['errors'].append(
                    f"Iteration count mismatch: DB={len(db_iterations)}, Files={len(file_iterations)}"
                )
                result['passed'] = False
            
            # Check for missing iterations in either system
            missing_in_db = set(file_iterations) - set(db_iterations)
            missing_in_files = set(db_iterations) - set(file_iterations)
            
            if missing_in_db:
                result['errors'].append(f"Iterations in files but not in DB: {missing_in_db}")
                result['passed'] = False
            
            if missing_in_files:
                result['warnings'].append(f"Iterations in DB but not in files: {missing_in_files}")
            
            # Validate iteration data completeness
            for iteration_num in db_iterations:
                db_data = self.db_manager.get_iteration_data(iteration_num)
                if not db_data:
                    result['errors'].append(f"No data found for iteration {iteration_num} in DB")
                    result['passed'] = False
                    continue
                
                # Check for required fields
                required_fields = ['highlight_lots', 'pareto_front', 'predictions']
                for field in required_fields:
                    if not db_data.get(field):
                        result['warnings'].append(f"Missing {field} for iteration {iteration_num}")
                
        except Exception as e:
            result['errors'].append(f"Data consistency validation failed: {str(e)}")
            result['passed'] = False
        
        return result
    
    def _validate_iteration_completeness(self) -> Dict[str, Any]:
        """Validate that each iteration has complete data"""
        result = {'passed': True, 'errors': [], 'warnings': []}
        
        try:
            iterations = self.db_manager.get_all_iterations()
            
            for iteration_num in iterations:
                data = self.db_manager.get_iteration_data(iteration_num)
                if not data:
                    result['errors'].append(f"No data for iteration {iteration_num}")
                    result['passed'] = False
                    continue
                
                # Check highlight lots count
                expected_lots = 3  # 3 points per iteration
                actual_lots = len(data.get('highlight_lots', []))
                if actual_lots != expected_lots:
                    result['warnings'].append(
                        f"Iteration {iteration_num}: Expected {expected_lots} highlight lots, got {actual_lots}"
                    )
                
                # Check Pareto front data
                pareto_points = len(data.get('pareto_front', []))
                if pareto_points == 0:
                    result['errors'].append(f"No Pareto front data for iteration {iteration_num}")
                    result['passed'] = False
                
                # Check predictions
                predictions = len(data.get('predictions', []))
                if predictions == 0:
                    result['warnings'].append(f"No predictions for iteration {iteration_num}")
                
        except Exception as e:
            result['errors'].append(f"Iteration completeness validation failed: {str(e)}")
            result['passed'] = False
        
        return result
    
    def _validate_data_freshness(self) -> Dict[str, Any]:
        """Validate data freshness and detect stale data"""
        result = {'passed': True, 'errors': [], 'warnings': []}
        
        try:
            # Check last processing time
            last_run = self.db_manager.get_system_metadata('last_run')
            if last_run:
                last_run_dt = datetime.fromisoformat(last_run.replace('Z', '+00:00'))
                days_since_last_run = (datetime.now() - last_run_dt).days
                
                if days_since_last_run > 7:
                    result['warnings'].append(f"Data is {days_since_last_run} days old")
                elif days_since_last_run > 30:
                    result['errors'].append(f"Data is very stale: {days_since_last_run} days old")
                    result['passed'] = False
            
            # Check for recent processing logs
            db_info = self.db_manager.get_database_info()
            if db_info['tables']['processing_logs'] == 0:
                result['warnings'].append("No processing logs found")
            
        except Exception as e:
            result['errors'].append(f"Data freshness validation failed: {str(e)}")
            result['passed'] = False
        
        return result
    
    def _validate_file_hashes(self) -> Dict[str, Any]:
        """Validate file integrity using hashes"""
        result = {'passed': True, 'errors': [], 'warnings': []}
        
        try:
            # Check manifest file hash if it exists
            manifest_path = os.path.join(self.cache_dir, 'manifest.json')
            if os.path.exists(manifest_path):
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
                
                # Validate dataset hash
                if 'hashes' in manifest and 'dataset' in manifest['hashes']:
                    expected_hash = manifest['hashes']['dataset']
                    actual_hash = self._calculate_file_hash('full_dataset.csv')
                    
                    if expected_hash != actual_hash:
                        result['errors'].append("Dataset file hash mismatch - file may be corrupted")
                        result['passed'] = False
                
                # Validate code hash
                if 'hashes' in manifest and 'code' in manifest['hashes']:
                    expected_hash = manifest['hashes']['code']
                    actual_hash = self._calculate_code_hash()
                    
                    if expected_hash != actual_hash:
                        result['warnings'].append("Code hash mismatch - code may have changed")
            
        except Exception as e:
            result['errors'].append(f"File hash validation failed: {str(e)}")
            result['passed'] = False
        
        return result
    
    def _get_file_cache_iterations(self) -> List[int]:
        """Get iteration numbers from file cache"""
        iterations = []
        iterations_dir = os.path.join(self.cache_dir, 'iterations')
        
        if not os.path.exists(iterations_dir):
            return iterations
        
        for item in os.listdir(iterations_dir):
            if item.startswith('iteration_'):
                try:
                    iteration_num = int(item.split('_')[1])
                    iterations.append(iteration_num)
                except (ValueError, IndexError):
                    continue
        
        return sorted(iterations)
    
    def _calculate_file_hash(self, filename: str) -> str:
        """Calculate SHA256 hash of a file"""
        filepath = os.path.join(self.cache_dir, filename)
        if not os.path.exists(filepath):
            return ""
        
        with open(filepath, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    
    def _calculate_code_hash(self) -> str:
        """Calculate hash of source code"""
        # This would hash the source code files
        # For now, return a placeholder
        return hashlib.sha256(CODE_VERSION.encode()).hexdigest()
    
    def generate_validation_report(self, results: Dict[str, Any]) -> str:
        """Generate a human-readable validation report"""
        report = []
        report.append("=" * 60)
        report.append("CACHE VALIDATION REPORT")
        report.append("=" * 60)
        report.append(f"Timestamp: {results['timestamp']}")
        report.append(f"Overall Status: {results['overall_status'].upper()}")
        report.append("")
        
        # Summary
        total_checks = len(results['checks'])
        passed_checks = sum(1 for check in results['checks'].values() if check.get('passed', False))
        report.append(f"Checks: {passed_checks}/{total_checks} passed")
        report.append(f"Errors: {len(results['errors'])}")
        report.append(f"Warnings: {len(results['warnings'])}")
        report.append("")
        
        # Detailed results
        for check_name, check_result in results['checks'].items():
            status = "✅ PASS" if check_result.get('passed', False) else "❌ FAIL"
            report.append(f"{check_name.title()}: {status}")
            
            if check_result.get('errors'):
                for error in check_result['errors']:
                    report.append(f"  Error: {error}")
            
            if check_result.get('warnings'):
                for warning in check_result['warnings']:
                    report.append(f"  Warning: {warning}")
        
        # Errors and warnings
        if results['errors']:
            report.append("")
            report.append("CRITICAL ERRORS:")
            for error in results['errors']:
                report.append(f"  • {error}")
        
        if results['warnings']:
            report.append("")
            report.append("WARNINGS:")
            for warning in results['warnings']:
                report.append(f"  • {warning}")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def save_validation_report(self, results: Dict[str, Any], output_path: str = None) -> str:
        """Save validation report to file"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.cache_dir, f"validation_report_{timestamp}.txt")
        
        report = self.generate_validation_report(results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"[validation] Report saved to: {output_path}")
        return output_path
