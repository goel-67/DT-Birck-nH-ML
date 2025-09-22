"""
ML models module for Pareto optimization system.
Handles model training, prediction, uncertainty estimation, and LOOCV.
"""

import warnings
# Suppress trivial warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.gaussian_process")
warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")
warnings.filterwarnings("ignore", message=".*noise_level.*")
warnings.filterwarnings("ignore", message=".*close to the specified lower bound.*")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, WhiteKernel, Matern
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, KFold

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import randint, uniform
from typing import Tuple, Optional, List, Dict, Any

from ..core.config import (
    FEATURES, FEATURE_RANGES, MODEL_CHOICE_FIRST_TWO_ITERATIONS,
    TARGET_RATE_MIN, TARGET_RATE_MAX, SAMPLING_METHOD, N_SAMPLES,
    SAMPLING_SEED, ROUND_SOBOL_TO_POW2, _HAS_QMC, GPR_HYPERPARAMETER_OPTIMIZATION
)


class MLModels:
    """Manages all machine learning models for the Pareto optimization system."""
    
    def __init__(self):
        self.rate_model = None
        self.range_model = None
        self.rate_model_type = None
        self.range_model_type = None
        self.rate_params = {}
        self.range_params = {}
    
    def make_extratrees(self) -> Pipeline:
        """Create an ExtraTrees regression model for etch rate prediction (iterations 3+)"""
        return Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", ExtraTreesRegressor(n_estimators=100, random_state=0, n_jobs=-1))
        ])
    
    def make_rf(self) -> Pipeline:
        """Create a RandomForest regression model for range prediction (all iterations)"""
        return Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1))
        ])
    
    def make_rf_rate(self) -> Pipeline:
        """Create a RandomForest regression model for etch rate prediction (iterations 1-2)"""
        return Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1))
        ])
    
    def make_gpr(self) -> Pipeline:
        """Create a Gaussian Process Regression model with Matern kernel (matching notebook)"""
        # Use Matern kernel as base (matching notebook)
        kernel = C(1.0, (1e-2, 1e2)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5) + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1e0))
        return Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", GaussianProcessRegressor(
                kernel=kernel, normalize_y=True, n_restarts_optimizer=1, random_state=0
            ))
        ])
    
    def tune_gpr(self, X: np.ndarray, y: np.ndarray) -> Tuple[Pipeline, Dict]:
        """Tune GPR hyperparameters using RandomizedSearchCV (matching notebook exactly)"""
        print(f"[gpr_tuning] Starting GPR hyperparameter optimization for {len(X)} samples")
        
        # Create base model with Matern kernel (matching notebook)
        k_mat = C(1.0, (1e-2, 1e2)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5) + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1e0))
        k_rbf = C(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1e0))
        
        base = GaussianProcessRegressor(kernel=k_mat, normalize_y=True, n_restarts_optimizer=1, random_state=0)
        pipe = Pipeline([("scaler", StandardScaler()), ("regressor", base)])
        
        # Parameter grid exactly matching notebook
        param_grid = {
            "regressor__kernel": [k_mat, k_rbf],
            "regressor__alpha": np.logspace(-7, -3, 5)
        }
        
        # Use KFold with 3 splits exactly matching notebook
        cv = KFold(n_splits=3, shuffle=True, random_state=0)
        
        # Use 6 iterations exactly matching notebook
        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_grid,
            n_iter=6,
            cv=cv,
            n_jobs=-1,
            refit=True,
            random_state=0
        )
        
        search.fit(X, y)
        
        print(f"[gpr_tuning] Best CV score: {-search.best_score_:.4f}")
        print(f"[gpr_tuning] Best parameters: {search.best_params_}")
        
        return search.best_estimator_, search.best_params_
    
    def pred_stats(self, pipeline_model: Pipeline, X_in: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate prediction statistics (mean and std) from ensemble model or GPR"""
        scaler = pipeline_model.named_steps["scaler"]
        regressor = pipeline_model.named_steps["regressor"]
        Xs = scaler.transform(X_in)
        
        if isinstance(regressor, GaussianProcessRegressor):
            # GPR provides mean and std directly
            mu, std = regressor.predict(Xs, return_std=True)
            return mu, std
        else:
            # Ensemble models (RF, ExtraTrees) - calculate mean and std across trees
            per_tree = np.stack([t.predict(Xs) for t in regressor.estimators_], axis=0)
            return per_tree.mean(axis=0), per_tree.std(axis=0)
    
    def train_models_for_iteration(self, X: np.ndarray, y_rate: np.ndarray, y_range: np.ndarray, 
                                  iteration_num: int) -> Tuple[Pipeline, Pipeline, Dict, Dict]:
        """Train models for a specific iteration with appropriate model selection"""
        print(f"[models] Training models for iteration {iteration_num}")
        
        rate_params = {}
        range_params = {}
        
        # Model selection based on iteration number
        if iteration_num <= 2:
            # Iterations 1-2: Random Forest for both rate and range
            print(f"[models] Iteration {iteration_num}: Using Random Forest for both rate and range")
            rate_model = self.make_rf_rate().fit(X, y_rate)
            range_model = self.make_rf().fit(X, y_range)
            self.rate_model_type = "RandomForest"
            self.range_model_type = "RandomForest"
            # Extract actual model parameters
            rate_params = rate_model.get_params()
            range_params = range_model.get_params()
        elif iteration_num <= 4:
            # Iterations 3-4: Extra Trees for rate, Random Forest for range
            print(f"[models] Iteration {iteration_num}: Using Extra Trees for rate, Random Forest for range")
            rate_model = self.make_extratrees().fit(X, y_rate)
            range_model = self.make_rf().fit(X, y_range)
            self.rate_model_type = "ExtraTrees"
            self.range_model_type = "RandomForest"
            # Extract actual model parameters
            rate_params = rate_model.get_params()
            range_params = range_model.get_params()
        else:
            # Iterations 5+: GPR for both rate and range
            if GPR_HYPERPARAMETER_OPTIMIZATION:
                print(f"[models] Iteration {iteration_num}: Using GPR with hyperparameter optimization for both rate and range")
                rate_model, rate_params = self.tune_gpr(X, y_rate)
                range_model, range_params = self.tune_gpr(X, y_range)
            else:
                print(f"[models] Iteration {iteration_num}: Using GPR with default parameters for both rate and range")
                rate_model = self.make_gpr()
                range_model = self.make_gpr()
                rate_model.fit(X, y_rate)
                range_model.fit(X, y_range)
                # Extract actual GPR parameters
                rate_params = rate_model.get_params()
                range_params = range_model.get_params()
            self.rate_model_type = "GPR"
            self.range_model_type = "GPR"
        
        self.rate_model = rate_model
        self.range_model = range_model
        self.rate_params = rate_params
        self.range_params = range_params
        
        return rate_model, range_model, rate_params, range_params
        
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Predict etch rate and range with uncertainties"""
        if self.rate_model is None or self.range_model is None:
            raise ValueError("Models must be trained before prediction")
        
        mu_rate, sd_rate = self.pred_stats(self.rate_model, X)
        mu_range, sd_range = self.pred_stats(self.range_model, X)
        
        # Convert range to nm (multiply by 5)
        mu_range_nm = mu_range * 5.0
        sd_range_nm = sd_range * 5.0
        
        return mu_rate, sd_rate, mu_range_nm, sd_range_nm
    
    def calculate_uncertainties_for_iteration(self, iteration_num: int, recipes_df: pd.DataFrame, 
                                            data_manager) -> pd.DataFrame:
        """Calculate uncertainties for recipes in a specific iteration"""
        print(f"[uncertainty] Calculating uncertainties for iteration {iteration_num}")
        
        # Load the full dataset
        try:
            df = data_manager.load_dataset()
            print(f"[uncertainty] Loaded dataset with {len(df)} rows")
        except Exception as e:
            print(f"[uncertainty] Error loading dataset: {e}")
            return recipes_df
        
        # Get the proper training data for this iteration
        training_data = data_manager.get_training_data_for_iteration(df, recipes_df, iteration_num)
        
        if training_data.empty:
            print(f"[uncertainty] Iteration {iteration_num}: No training data available")
            return recipes_df
        
        # Prepare features and targets
        X = training_data[FEATURES].values
        y_rate = training_data['AvgEtchRate'].values
        y_range = training_data['RangeEtchRate'].values
        
        # Train models
        rate_model, range_model, rate_params, range_params = self.train_models_for_iteration(X, y_rate, y_range, iteration_num)
        
        # Get the first 3 recipes from this iteration for uncertainty calculation
        iteration_recipes = recipes_df.head(3)
        
        if len(iteration_recipes) == 0:
            print(f"[uncertainty] No recipes found for iteration {iteration_num}")
            return recipes_df
        
        # Extract features for these recipes
        recipe_features = []
        for _, recipe in iteration_recipes.iterrows():
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
        rate_mean, rate_std = self.pred_stats(rate_model, recipe_features)
        range_mean, range_std = self.pred_stats(range_model, recipe_features)
        
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
        return recipes_df_copy
    
    def run_loocv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run Leave-One-Out Cross Validation on the dataset"""
        X = df[FEATURES].astype(float).values
        y_rate = df["AvgEtchRate"].values
        y_range = df["RangeEtchRate"].values
        n = len(df)
        
        yp_r = np.zeros(n)
        ys_r = np.zeros(n)
        yp_g = np.zeros(n)
        ys_g = np.zeros(n)
        
        for i in range(n):
            m = np.ones(n, dtype=bool)
            m[i] = False
            
            # Use Random Forest for both rate and range in LOOCV (baseline)
            mdl_r = self.make_rf_rate().fit(X[m], y_rate[m])
            mdl_g = self.make_rf().fit(X[m], y_range[m])
            
            mu_r, sd_r = self.pred_stats(mdl_r, X[i:i+1])
            mu_g, sd_g = self.pred_stats(mdl_g, X[i:i+1])
            
            yp_r[i] = mu_r[0]
            ys_r[i] = sd_r[0]
            yp_g[i] = mu_g[0]
            ys_g[i] = sd_g[0]
        
        out = df[["LOTNAME", "FIMAP_FILE", "Date", "AvgEtchRate", "RangeEtchRate"]].copy()
        out = out.rename(columns={"AvgEtchRate": "loo_true_rate", "RangeEtchRate": "loo_true_range"})
        out["loo_pred_rate"] = yp_r
        out["loo_std_rate"] = ys_r
        out["loo_pred_range"] = yp_g
        out["loo_std_range"] = ys_g
        
        return out
    
    def run_loocv_iteration_specific(self, df: pd.DataFrame, iteration_num: int, 
                                   recipes_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Run LOOCV on cumulative data up to a specific iteration with proper model selection"""
        print(f"[loocv] Running iteration-specific LOOCV for iteration {iteration_num}")
        
        # Use the same model selection logic as train_models_for_iteration
        if iteration_num <= 2:
            # Iterations 1-2: Random Forest for both rate and range
            rate_model_type = "RandomForest"
            range_model_type = "RandomForest"
            print(f"[loocv] Iteration {iteration_num}: Using Random Forest for both rate and range")
        elif iteration_num <= 4:
            # Iterations 3-4: Extra Trees for rate, Random Forest for range
            rate_model_type = "ExtraTrees"
            range_model_type = "RandomForest"
            print(f"[loocv] Iteration {iteration_num}: Using Extra Trees for rate, Random Forest for range")
        else:
            # Iterations 5+: GPR for both rate and range
            rate_model_type = "GPR"
            range_model_type = "GPR"
            print(f"[loocv] Iteration {iteration_num}: Using GPR for both rate and range")
        
        # For LOOCV, we use the full dataset but with iteration-specific model selection
        # This gives us a fair comparison of how the models would perform
        df_filtered = df.copy()
        print(f"[loocv] Iteration {iteration_num}: Using {len(df_filtered)} training points for LOOCV")
        
        X = df_filtered[FEATURES].astype(float).values
        y_rate = df_filtered["AvgEtchRate"].values
        y_range = df_filtered["RangeEtchRate"].values
        n = len(df_filtered)
        
        yp_r = np.zeros(n)
        ys_r = np.zeros(n)
        yp_g = np.zeros(n)
        ys_g = np.zeros(n)
        
        for i in range(n):
            m = np.ones(n, dtype=bool)
            m[i] = False
            
            # Choose model based on iteration (same logic as train_models_for_iteration)
            if iteration_num <= 2:
                # Iterations 1-2: Random Forest for both
                mdl_r = self.make_rf_rate().fit(X[m], y_rate[m])
                mdl_g = self.make_rf().fit(X[m], y_range[m])
            elif iteration_num <= 4:
                # Iterations 3-4: Extra Trees for rate, Random Forest for range
                mdl_r = self.make_extratrees().fit(X[m], y_rate[m])
                mdl_g = self.make_rf().fit(X[m], y_range[m])
            else:
                # Iterations 5+: GPR for both
                if GPR_HYPERPARAMETER_OPTIMIZATION:
                    mdl_r, _ = self.tune_gpr(X[m], y_rate[m])
                    mdl_g, _ = self.tune_gpr(X[m], y_range[m])
                else:
                    mdl_r = self.make_gpr().fit(X[m], y_rate[m])
                    mdl_g = self.make_gpr().fit(X[m], y_range[m])
            
            mu_r, sd_r = self.pred_stats(mdl_r, X[i:i+1])
            mu_g, sd_g = self.pred_stats(mdl_g, X[i:i+1])
            
            yp_r[i] = mu_r[0]
            ys_r[i] = sd_r[0]
            yp_g[i] = mu_g[0]
            ys_g[i] = sd_g[0]
        
        out = df_filtered[["LOTNAME", "FIMAP_FILE", "Date", "AvgEtchRate", "RangeEtchRate"]].copy()
        out = out.rename(columns={"AvgEtchRate": "loo_true_rate", "RangeEtchRate": "loo_true_range"})
        out["loo_pred_rate"] = yp_r
        out["loo_std_rate"] = ys_r
        out["loo_pred_range"] = yp_g
        out["loo_std_range"] = ys_g
        out["iteration_num"] = iteration_num
        out["rate_model_type"] = rate_model_type
        out["range_model_type"] = range_model_type
        
        print(f"[loocv] Completed LOOCV for iteration {iteration_num} with {rate_model_type} for rate, {range_model_type} for range")
        return out


class SamplingEngine:
    """Handles candidate sampling for Pareto optimization."""
    
    def __init__(self):
        pass
    
    def _nearest_pow2(self, n: int) -> int:
        """Find nearest power of 2"""
        if n <= 1:
            return 1
        a = 1 << (n-1).bit_length()
        b = a >> 1
        return a if (a - n) <= (n - b) else b
    
    def sample_candidates(self, method: str, n: int, lower: np.ndarray, 
                         upper: np.ndarray, seed: int) -> np.ndarray:
        """Sample candidate points using specified method"""
        d = len(lower)
        m = method.lower()
        
        if m == "sobol":
            # Always ensure Sobol uses powers of 2 for optimal performance
            n = self._nearest_pow2(n)
            print(f"[sampling] Sobol sampling: adjusted N_SAMPLES from {n//self._nearest_pow2(n)} to {n} (power of 2)")
        elif m == "sobol" and ROUND_SOBOL_TO_POW2:
            n = self._nearest_pow2(n)
        
        if m == "random" or not _HAS_QMC:
            rng = np.random.default_rng(seed)
            u = rng.random((n, d))
            return lower + u * (upper - lower)
        
        if m == "sobol":
            from scipy.stats import qmc
            eng = qmc.Sobol(d=d, scramble=True, seed=seed)
            u = eng.random(n)
            return qmc.scale(u, lower, upper)
        
        if m == "lhs":
            from scipy.stats import qmc
            eng = qmc.LatinHypercube(d=d, seed=seed)
            u = eng.random(n)
            return qmc.scale(u, lower, upper)
        
        # Fallback to random
        rng = np.random.default_rng(seed)
        u = rng.random((n, d))
        return lower + u * (upper - lower)
    
    def quantize(self, X: np.ndarray, cols: List[str]) -> np.ndarray:
        """Quantize features to appropriate precision"""
        Xq = X.copy()
        for j, col in enumerate(cols):
            kl = col.lower()
            if "flow" in kl or "pres" in kl:
                Xq[:, j] = np.round(Xq[:, j] * 10) / 10.0
            elif "pow" in kl:
                Xq[:, j] = np.round(Xq[:, j])
        return Xq
    
    def generate_candidates(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate candidate points with predictions and uncertainties"""
        # Sample candidates
        lower = np.array([FEATURE_RANGES[c][0] for c in FEATURES], float)
        upper = np.array([FEATURE_RANGES[c][1] for c in FEATURES], float)
        
        Xcand = self.sample_candidates(SAMPLING_METHOD, N_SAMPLES, lower, upper, SAMPLING_SEED)
        Xcand = self.quantize(Xcand, FEATURES)
        
        return Xcand, lower, upper
    
    def filter_candidates(self, Xcand: np.ndarray, mu_r: np.ndarray, mu_g_nm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Filter candidates based on target rate and rf2 > rf1 constraints"""
        # Apply target rate filter
        mask = (mu_r >= TARGET_RATE_MIN) & (mu_r <= TARGET_RATE_MAX)
        if not np.any(mask):
            mask = np.ones(len(mu_r), dtype=bool)
        
        # Apply rf2 > rf1 filter
        rf_mask = np.ones(len(mu_r), dtype=bool)
        for i in range(len(mu_r)):
            if mask[i]:  # Only check if already passed target rate filter
                rf1_power = Xcand[i, FEATURES.index("Etch_Avg_Rf1_Pow")]
                rf2_power = Xcand[i, FEATURES.index("Etch_Avg_Rf2_Pow")]
                if rf2_power <= rf1_power:
                    rf_mask[i] = False
        
        # Combine both filters
        combined_mask = mask & rf_mask
        
        if not np.any(combined_mask):
            print(f"[sampling] Warning: No candidates meet both target rate and rf2 > rf1 requirements")
            print(f"[sampling] Target rate filter: {np.sum(mask)} candidates")
            print(f"[sampling] Rf2 > rf1 filter: {np.sum(rf_mask)} candidates")
            print(f"[sampling] Combined filter: {np.sum(combined_mask)} candidates")
            # Fallback to just target rate filter if no candidates meet both
            combined_mask = mask
        
        Xb = Xcand[combined_mask]
        mur = mu_r[combined_mask]
        mug = mu_g_nm[combined_mask]
        
        print(f"[sampling] Filtered candidates: {np.sum(mask)} (target rate) → {np.sum(combined_mask)} (with rf2 > rf1)")
        
        return Xb, mur, mug, mask, rf_mask
