import numpy as np
from typing import List, Tuple, Optional
from ..core.config import *

class CandidateSampler:
    def __init__(self, method: str = SAMPLING_METHOD, n_samples: int = N_SAMPLES, seed: int = SAMPLING_SEED):
        self.method = method
        self.n_samples = n_samples
        self.seed = seed
        
    def sample_candidates(self, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        """Sample candidate points in the feature space"""
        if self.method == "sobol":
            return self._sobol_sample(lower, upper)
        elif self.method == "random":
            return self._random_sample(lower, upper)
        else:
            raise ValueError(f"Unknown sampling method: {self.method}")
    
    def _sobol_sample(self, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        """Generate Sobol sequence samples"""
        try:
            from scipy.stats import qmc
            
            # Create Sobol sampler
            sampler = qmc.Sobol(d=len(lower), seed=self.seed)
            
            # Generate samples in [0,1] range
            if ROUND_SOBOL_TO_POW2:
                # Find next power of 2
                n_pow2 = 2**int(np.ceil(np.log2(self.n_samples)))
                samples = sampler.random(n_pow2)
            else:
                samples = sampler.random(self.n_samples)
            
            # Scale to feature ranges
            scaled_samples = lower + samples * (upper - lower)
            
            # Truncate to requested number of samples
            return scaled_samples[:self.n_samples]
            
        except ImportError:
            print("[sampler] scipy not available, falling back to random sampling")
            return self._random_sample(lower, upper)
    
    def _random_sample(self, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        """Generate random samples"""
        np.random.seed(self.seed)
        return np.random.uniform(lower, upper, (self.n_samples, len(lower)))
    
    def quantize_features(self, X: np.ndarray, features: List[str]) -> np.ndarray:
        """Quantize features to realistic process values"""
        X_quantized = X.copy()
        
        for i, feature in enumerate(features):
            if feature in FEATURE_RANGES:
                min_val, max_val = FEATURE_RANGES[feature]
                
                # Quantize to realistic step sizes
                if feature == "Etch_AvgO2Flow":
                    step = 1.0  # 1 sccm steps
                elif feature == "Etch_Avgcf4Flow":
                    step = 0.5  # 0.5 sccm steps
                elif feature == "Etch_Avg_Rf1_Pow":
                    step = 1.0  # 1W steps
                elif feature == "Etch_Avg_Rf2_Pow":
                    step = 1.0  # 1W steps
                elif feature == "Etch_AvgPres":
                    step = 0.1  # 0.1 mTorr steps
                elif feature == "Etch_Avg_ChTemp":
                    step = 1.0  # 1°C steps
                elif feature == "Etch_Avg_ElecTemp":
                    step = 1.0  # 1°C steps
                else:
                    step = (max_val - min_val) / 100  # Default to 100 steps
                
                # Quantize values
                X_quantized[:, i] = np.round(X[:, i] / step) * step
                
                # Ensure values are within bounds
                X_quantized[:, i] = np.clip(X_quantized[:, i], min_val, max_val)
        
        return X_quantized

def _sample_candidates(method: str, n_samples: int, lower: np.ndarray, upper: np.ndarray, seed: int) -> np.ndarray:
    """Sample candidates - compatibility function for existing code"""
    sampler = CandidateSampler(method, n_samples, seed)
    return sampler.sample_candidates(lower, upper)

def _quantize(X: np.ndarray, features: List[str]) -> np.ndarray:
    """Quantize features - compatibility function for existing code"""
    sampler = CandidateSampler()
    return sampler.quantize_features(X, features)
