#!/usr/bin/env python3
"""
Generate parity plots and residual plots for 6 models using project hyperparameters:
- Targets: AvgEtchRate, Thickness Range (RangeEtchRate * 5)
- Models: RandomForest, ExtraTrees, GPR
- Protocol: LOOCV
- Error bars on parity plots:
  * AvgEtchRate: std of 41 site rates / sqrt(41)
  * Thickness Range: proxy absolute uncertainty = (RangeEtchRate / AvgEtchRate) * (RangeEtchRate * 5)
    (dimensionless relative error multiplied by thickness range value)

Outputs saved to parity_test_outputs/ as PNG files.
"""

import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure project src is importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

# Project imports
from src.core.config import DATASET_CSV, FEATURES, GPR_HYPERPARAMETER_OPTIMIZATION
from src.ml.ml_models import MLModels


OUTPUT_DIR = "parity_test_outputs"


def _ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATASET_CSV)
    return df


def build_features_and_targets(df: pd.DataFrame):
    X = df[FEATURES].astype(float).values
    y_rate = df["AvgEtchRate"].astype(float).values
    y_range_thickness = (df["RangeEtchRate"].astype(float).values * 5.0)
    return X, y_rate, y_range_thickness


def compute_uncertainties(df: pd.DataFrame):
    # AvgEtchRate uncertainty = std across 41 sites / sqrt(41)
    rate_cols = [f"Rate_{i}_nm_per_min" for i in range(1, 42)]
    site_matrix = df[rate_cols].astype(float).values
    std_across_sites = np.nanstd(site_matrix, axis=1, ddof=1)
    yerr_rate = std_across_sites / math.sqrt(41.0)
    yerr_rate = np.nan_to_num(yerr_rate, nan=0.0, posinf=0.0, neginf=0.0)

    # Thickness Range proxy absolute uncertainty
    # relative_error = RangeEtchRate / AvgEtchRate
    # absolute_uncertainty_for_thickness = relative_error * (RangeEtchRate * 5)
    rel_err = (df["RangeEtchRate"].astype(float).values) / np.clip(df["AvgEtchRate"].astype(float).values, 1e-9, None)
    y_range_thickness = (df["RangeEtchRate"].astype(float).values * 5.0)
    yerr_range_thickness = rel_err * y_range_thickness

    return yerr_rate, yerr_range_thickness


def loocv_predict(model_builder, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    preds = np.zeros(n, dtype=float)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        mdl = model_builder()
        mdl.fit(X[mask], y[mask])
        preds[i] = float(mdl.predict(X[i:i+1])[0])
    return preds


def r2_rmse(y_true: np.ndarray, y_pred: np.ndarray):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = math.sqrt(np.mean((y_true - y_pred) ** 2))
    return r2, rmse


def plot_parity(y_true, y_pred, y_err, title, path):
    plt.figure(figsize=(7, 7), dpi=160)
    plt.rcParams.update({'font.size': 18})
    plt.errorbar(
        y_true,
        y_pred,
        yerr=y_err,
        fmt='o',
        color='#1f77b4',
        alpha=0.9,
        ecolor='black',
        elinewidth=2.0,
        capsize=0,
        zorder=2,
    )
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    pad = 0.02 * (hi - lo) if hi > lo else 1.0
    plt.plot([lo - pad, hi + pad], [lo - pad, hi + pad], 'k--', linewidth=1.2)
    r2, rmse = r2_rmse(y_true, y_pred)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(f"{title}\nR2={r2:.3f}, RMSE={rmse:.3f}")
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()


def plot_residuals(y_true, y_pred, title, path):
    resid = y_true - y_pred
    plt.figure(figsize=(7, 7), dpi=160)
    plt.rcParams.update({'font.size': 18})
    plt.scatter(y_pred, resid, s=56, alpha=0.85)
    plt.axhline(0.0, color='k', linestyle='--', linewidth=1.2)
    plt.xlabel("Predicted")
    plt.ylabel("Residual (True - Pred)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()


def main():
    _ensure_output_dir()
    df = load_dataset()
    X, y_rate, y_range_th = build_features_and_targets(df)
    yerr_rate, yerr_range_th = compute_uncertainties(df)

    models = MLModels()

    # Builders using project hyperparameters
    def rf_rate():
        return models.make_rf_rate()

    def rf_range():
        return models.make_rf()

    def et_rate():
        return models.make_extratrees()

    def et_range():
        # Use same ET hyperparameters for range as for rate (scaler + ExtraTreesRegressor)
        return models.make_extratrees()

    def gpr_builder():
        # Respect project flag; if hyperopt is on we still fit default per-sample for parity test simplicity
        if GPR_HYPERPARAMETER_OPTIMIZATION:
            # Fallback to default make_gpr for deterministic, quick LOOCV
            return models.make_gpr()
        return models.make_gpr()

    # RF
    rf_rate_pred = loocv_predict(rf_rate, X, y_rate)
    rf_range_pred = loocv_predict(rf_range, X, y_range_th)
    plot_parity(y_rate, rf_rate_pred, yerr_rate, "AvgEtchRate – RF (LOOCV)", os.path.join(OUTPUT_DIR, "parity_AvgEtchRate_RF.png"))
    plot_parity(y_range_th, rf_range_pred, yerr_range_th, "ThicknessRange – RF (LOOCV)", os.path.join(OUTPUT_DIR, "parity_ThicknessRange_RF.png"))

    # Extra Trees
    et_rate_pred = loocv_predict(et_rate, X, y_rate)
    et_range_pred = loocv_predict(et_range, X, y_range_th)
    plot_parity(y_rate, et_rate_pred, yerr_rate, "AvgEtchRate – ET (LOOCV)", os.path.join(OUTPUT_DIR, "parity_AvgEtchRate_ET.png"))
    plot_parity(y_range_th, et_range_pred, yerr_range_th, "ThicknessRange – ET (LOOCV)", os.path.join(OUTPUT_DIR, "parity_ThicknessRange_ET.png"))

    # GPR
    def gpr_rate():
        return gpr_builder()

    def gpr_range():
        return gpr_builder()

    gpr_rate_pred = loocv_predict(gpr_rate, X, y_rate)
    gpr_range_pred = loocv_predict(gpr_range, X, y_range_th)
    plot_parity(y_rate, gpr_rate_pred, yerr_rate, "AvgEtchRate – GPR (LOOCV)", os.path.join(OUTPUT_DIR, "parity_AvgEtchRate_GPR.png"))
    plot_parity(y_range_th, gpr_range_pred, yerr_range_th, "ThicknessRange – GPR (LOOCV)", os.path.join(OUTPUT_DIR, "parity_ThicknessRange_GPR.png"))

    # Residual plots for both GPR models
    plot_residuals(y_rate, gpr_rate_pred, "AvgEtchRate – GPR Residuals", os.path.join(OUTPUT_DIR, "residuals_AvgEtchRate_GPR.png"))
    plot_residuals(y_range_th, gpr_range_pred, "ThicknessRange – GPR Residuals", os.path.join(OUTPUT_DIR, "residuals_ThicknessRange_GPR.png"))

    print(f"Saved outputs to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()


