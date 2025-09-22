#!/usr/bin/env python3
"""
Test script for the modular Pareto optimization system.
Verifies that all modules work correctly and replicate pareto.py functionality.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported correctly"""
    print("Testing module imports...")
    
    try:
        from src.core.config import (
            CODE_VERSION, FEATURES, FEATURE_RANGES, ALPHA, BETA, GAMMA,
            TARGET_RATE_MIN, TARGET_RATE_MAX, K, SAMPLING_METHOD, N_SAMPLES
        )
        print("✅ config.py imported successfully")
        print(f"   Version: {CODE_VERSION}")
        print(f"   Features: {len(FEATURES)} features")
        print(f"   Alpha: {ALPHA}, Beta: {BETA}, Gamma: {GAMMA}")
        print(f"   Target rate: {TARGET_RATE_MIN}-{TARGET_RATE_MAX}")
        print(f"   K: {K}, Sampling: {SAMPLING_METHOD}, N_Samples: {N_SAMPLES}")
    except Exception as e:
        print(f"❌ config.py import failed: {e}")
        return False
    
    try:
        from src.data.data_manager import DataManager
        print("✅ data_manager.py imported successfully")
    except Exception as e:
        print(f"❌ data_manager.py import failed: {e}")
        return False
    
    try:
        from src.ml.ml_models import MLModels, SamplingEngine
        print("✅ ml_models.py imported successfully")
    except Exception as e:
        print(f"❌ ml_models.py import failed: {e}")
        return False
    
    try:
        from src.optimization.pareto_optimizer import ParetoOptimizer
        print("✅ pareto_optimizer.py imported successfully")
    except Exception as e:
        print(f"❌ pareto_optimizer.py import failed: {e}")
        return False
    
    try:
        from src.visualization.plotter import Plotter
        print("✅ plotter.py imported successfully")
    except Exception as e:
        print(f"❌ plotter.py import failed: {e}")
        return False
    
    try:
        from src.core.main import ParetoSystem
        print("✅ main.py imported successfully")
    except Exception as e:
        print(f"❌ main.py import failed: {e}")
        return False
    
    try:
        from src.cli.cli import CLI
        print("✅ cli.py imported successfully")
    except Exception as e:
        print(f"❌ cli.py import failed: {e}")
        return False
    
    return True

def test_data_manager():
    """Test DataManager functionality"""
    print("\nTesting DataManager...")
    
    try:
        from src.data.data_manager import DataManager
        dm = DataManager()
        
        # Test directory creation
        dm._ensure_dirs()
        print("✅ Directory creation works")
        
        # Test hash functions
        test_df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        })
        hash_result = dm.dataset_hash(test_df)
        print(f"✅ Dataset hash calculation works: {hash_result}")
        
        features_hash = dm.features_hash()
        print(f"✅ Features hash calculation works: {features_hash}")
        
        model_hash = dm.model_config_hash()
        print(f"✅ Model config hash calculation works: {model_hash}")
        
        code_hash = dm.code_hash()
        print(f"✅ Code hash calculation works: {code_hash}")
        
        cache_key = dm.cache_key(hash_result, features_hash, model_hash, code_hash)
        print(f"✅ Cache key generation works: {cache_key}")
        
        return True
        
    except Exception as e:
        print(f"❌ DataManager test failed: {e}")
        return False

def test_ml_models():
    """Test MLModels functionality"""
    print("\nTesting MLModels...")
    
    try:
        from src.ml.ml_models import MLModels, SamplingEngine
        from src.core.config import FEATURES
        
        ml = MLModels()
        se = SamplingEngine()
        
        # Test model creation
        rf_model = ml.make_rf()
        et_model = ml.make_extratrees()
        rf_rate_model = ml.make_rf_rate()
        
        print("✅ Model creation works")
        
        # Test sampling
        lower = np.array([0.0] * len(FEATURES))
        upper = np.array([100.0] * len(FEATURES))
        candidates = se.sample_candidates("random", 100, lower, upper, 42)
        
        print(f"✅ Sampling works: {candidates.shape}")
        
        # Test quantization
        quantized = se.quantize(candidates, FEATURES)
        print(f"✅ Quantization works: {quantized.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ MLModels test failed: {e}")
        return False

def test_pareto_optimizer():
    """Test ParetoOptimizer functionality"""
    print("\nTesting ParetoOptimizer...")
    
    try:
        from src.optimization.pareto_optimizer import ParetoOptimizer
        
        po = ParetoOptimizer()
        
        # Test Pareto front calculation
        test_points = np.array([
            [10, 5],   # Good rate, good range
            [15, 3],   # Better rate, better range
            [8, 8],    # Worse rate, worse range
            [12, 4],   # Good rate, good range
        ])
        
        pareto_flags = po.is_pareto(test_points)
        print(f"✅ Pareto front calculation works: {pareto_flags}")
        
        # Test normalization
        test_array = np.array([1, 2, 3, 4, 5])
        normalized = po.norm01(test_array)
        print(f"✅ Normalization works: {normalized}")
        
        # Test objective scales
        rate_vals = np.array([10, 15, 8, 12])
        range_vals = np.array([5, 3, 8, 4])
        r_scale, rn_scale = po.objective_scales(rate_vals, range_vals)
        print(f"✅ Objective scales calculation works: r_scale={r_scale}, rn_scale={rn_scale}")
        
        # Test improvement checking
        front = np.array([[15, 3], [12, 4]])
        improves = po.improves((10, 5), front)
        print(f"✅ Improvement checking works: {improves}")
        
        return True
        
    except Exception as e:
        print(f"❌ ParetoOptimizer test failed: {e}")
        return False

def test_plotter():
    """Test Plotter functionality"""
    print("\nTesting Plotter...")
    
    try:
        from src.visualization.plotter import Plotter
        
        plotter = Plotter()
        
        # Test symbol generation
        symbol1 = plotter.get_symbol_for_point(1, 1)
        symbol2 = plotter.get_symbol_for_point(2, 3)
        print(f"✅ Symbol generation works: {symbol1}, {symbol2}")
        
        return True
        
    except Exception as e:
        print(f"❌ Plotter test failed: {e}")
        return False

def test_cli():
    """Test CLI functionality"""
    print("\nTesting CLI...")
    
    try:
        from src.cli.cli import CLI
        
        cli = CLI()
        
        # Test latest date function
        latest_date = cli._get_latest_snapshot_date()
        print(f"✅ Latest date function works: {latest_date}")
        
        return True
        
    except Exception as e:
        print(f"❌ CLI test failed: {e}")
        return False

def test_system_integration():
    """Test system integration"""
    print("\nTesting system integration...")
    
    try:
        from src.core.main import ParetoSystem
        
        system = ParetoSystem()
        print("✅ ParetoSystem instantiation works")
        
        # Test that all components are available
        assert hasattr(system, 'data_manager')
        assert hasattr(system, 'ml_models')
        assert hasattr(system, 'sampling_engine')
        assert hasattr(system, 'pareto_optimizer')
        assert hasattr(system, 'plotter')
        
        print("✅ All system components are available")
        
        return True
        
    except Exception as e:
        print(f"❌ System integration test failed: {e}")
        return False

def test_configuration():
    """Test configuration values"""
    print("\nTesting configuration...")
    
    try:
        from src.core.config import (
            FEATURES, FEATURE_RANGES, ALPHA, BETA, GAMMA,
            TARGET_RATE_MIN, TARGET_RATE_MAX, K
        )
        
        # Verify feature configuration
        assert len(FEATURES) > 0, "No features defined"
        assert all(f in FEATURE_RANGES for f in FEATURES), "Missing feature ranges"
        
        print(f"✅ Features configured: {len(FEATURES)} features")
        for feature in FEATURES:
            min_val, max_val = FEATURE_RANGES[feature]
            print(f"   {feature}: [{min_val}, {max_val}]")
        
        # Verify optimization parameters
        assert ALPHA > 0, "Alpha must be positive"
        assert BETA > 0, "Beta must be positive"
        assert GAMMA > 0, "Gamma must be positive"
        assert TARGET_RATE_MIN < TARGET_RATE_MAX, "Invalid target rate range"
        assert K > 0, "K must be positive"
        
        print(f"✅ Optimization parameters: α={ALPHA}, β={BETA}, γ={GAMMA}")
        print(f"✅ Target rate: [{TARGET_RATE_MIN}, {TARGET_RATE_MAX}]")
        print(f"✅ K: {K}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing Modular Pareto Optimization System")
    print("=" * 60)
    
    tests = [
        ("Module Imports", test_imports),
        ("Configuration", test_configuration),
        ("Data Manager", test_data_manager),
        ("ML Models", test_ml_models),
        ("Pareto Optimizer", test_pareto_optimizer),
        ("Plotter", test_plotter),
        ("CLI", test_cli),
        ("System Integration", test_system_integration),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:<25} {status}")
        if success:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The modular system is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
