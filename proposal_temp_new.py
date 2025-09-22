#!/usr/bin/env python3
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Tuple, List, Optional

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel as C, RBF, WhiteKernel, Matern, DotProduct
)
from scipy.stats import qmc, norm

# Optional SHAP
_HAS_SHAP = True
try:
    import shap
except Exception:
    _HAS_SHAP = False

# =========================
# Config
# =========================
CSV_PATH = "full_dataset.csv"
SEED = 42

# GPR kernel strategy: "MATERN_ARD" (default) or "RBF_SEARCH"
GPR_STRATEGY = "MATERN_ARD"

# Acquisition: "DISTANCE" (default) or "EI"
ACQ_STRATEGY = "DISTANCE"

NS_OBOL = 8192
N_EI_SAMPLES = 256
N_PROPOSALS = 3
PF_DENSITY = 200

X_FEATURES = [
    "Etch_AvgO2Flow",
    "Etch_Avg_Rf1_Pow",
    "Etch_Avg_Rf2_Pow",
    "Etch_AvgPres",
    "Etch_Avgcf4Flow",
]

Y1_COL = "AvgEtchRate"    # maximize
Y2_COL = "RangeEtchRate"  # multiplied to get thickness
THICKNESS_MULT = 5.0      # ThicknessRange = RangeEtchRate * THICKNESS_MULT

RATE_MIN = 35.0
RATE_MAX = 110.0

BOUNDS_LB = np.array([10.0, 0.0, 50.0, 4.0, 10.0])
BOUNDS_UB = np.array([90.0, 100.0, 700.0, 100.0, 90.0])
RES = np.array([0.1, 1.0, 1.0, 0.1, 0.1])  # hardware resolution for grouping & candidates

OUTPUT_DIR = "plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

rng = np.random.default_rng(SEED)
plt.rcParams["figure.dpi"] = 100

def RMSE(y_true, y_pred): return mean_squared_error(y_true, y_pred, squared=False)

# =========================
# Data
# =========================
@dataclass
class DataBundle:
    df: pd.DataFrame
    X: np.ndarray
    y_rate: np.ndarray
    y_thk: np.ndarray
    groups: np.ndarray      # grouped recipe ids
    point_cols: List[str]

def _num(df, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

def _slugify(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s).strip("_")

def make_recipe_groups(X: np.ndarray) -> np.ndarray:
    # snap to hardware resolution to define same-recipe groups
    Xs = (np.round(X / RES) * RES).astype(float)
    # hash rows to group ids
    keys = [tuple(row) for row in Xs]
    unique, inv = np.unique(keys, return_inverse=True, axis=0)
    return inv  # group id per sample

def load_data(path) -> DataBundle:
    df = pd.read_csv(path)
    need = X_FEATURES + [Y1_COL, Y2_COL]
    _num(df, need)

    # Optional 41-point columns
    point_cols = [f"Rate_{i}_nm_per_min" for i in range(1, 42)]
    if all(col in df.columns for col in point_cols):
        _num(df, point_cols)
    else:
        point_cols = []

    # Drop non-finite required fields
    df = df.dropna(subset=need).copy()
    df["ThicknessRange"] = df[Y2_COL] * THICKNESS_MULT

    X = df[X_FEATURES].to_numpy(float)
    y_rate = df[Y1_COL].to_numpy(float)
    y_thk  = df["ThicknessRange"].to_numpy(float)

    # Build groups by recipe
    groups = make_recipe_groups(X)
    return DataBundle(df=df, X=X, y_rate=y_rate, y_thk=y_thk, groups=groups, point_cols=point_cols)

# =========================
# Alpha vectors (keep your thickness logic as-is)
# =========================
def compute_alpha_vectors(df: pd.DataFrame, point_cols: List[str]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if not point_cols:
        return None, None
    pts = df[point_cols].to_numpy(dtype=float)
    std41 = np.nanstd(pts, axis=1, ddof=1)
    alpha_rate = (std41 / np.sqrt(41.0)) ** 2  # CLT on mean etch rate

    avg_rate = df[Y1_COL].to_numpy(dtype=float)
    rng_rate = df[Y2_COL].to_numpy(dtype=float)
    eps = 1e-12
    denom = np.where(np.abs(avg_rate) < eps, eps, avg_rate)
    alpha_range_rel = (rng_rate / denom) ** 2   # relative error proxy
    alpha_thickness = (alpha_range_rel * (THICKNESS_MULT ** 2)).astype(float)  # keep as-is
    return alpha_rate.astype(float), alpha_thickness

# =========================
# Models
# =========================
def make_gpr_matern_ard():
    # ARD Matern + linear trend + white noise
    k = (C(1.0, (1e-3, 1e3)) *
         Matern(length_scale=np.ones(len(X_FEATURES)),
                length_scale_bounds=(1e-2, 1e2),
                nu=1.5)) + DotProduct(sigma_0=1.0) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e0))
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("gpr", GaussianProcessRegressor(kernel=k, normalize_y=True, random_state=SEED, n_restarts_optimizer=1))
    ])
    return pipe

def make_gpr_rbf_grid():
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("gpr", GaussianProcessRegressor(
            kernel=C(1.0)*RBF(np.ones(len(X_FEATURES))) + WhiteKernel(1e-3),
            normalize_y=True,
            alpha=1e-10,
            n_restarts_optimizer=3,
            random_state=SEED
        ))
    ])
    kernels = []
    for c in (0.1, 1.0, 10.0):
        for w in (1e-5, 1e-3, 1e-1):
            for ls in (0.1, 1.0, 10.0):
                kernels.append(C(c)*RBF(length_scale=np.full(len(X_FEATURES), ls)) + WhiteKernel(w))
    grid = {"gpr__kernel": kernels}
    return pipe, grid

def tune_model_gpr(X, y, groups):
    if GPR_STRATEGY.upper() == "MATERN_ARD":
        return make_gpr_matern_ard()
    pipe, grid = make_gpr_rbf_grid()
    n_groups = len(np.unique(groups))
    n_splits = min(10, max(2, n_groups))
    cv = GroupKFold(n_splits=n_splits)
    rs = RandomizedSearchCV(
        pipe, param_distributions=grid,
        n_iter=min(25, len(grid["gpr__kernel"])),
        scoring="neg_root_mean_squared_error",
        cv=cv.split(X, y, groups=groups),
        n_jobs=-1, random_state=SEED, refit=True, verbose=0
    )
    rs.fit(X, y)
    return rs.best_estimator_

def clone_best_kernel(best_estimator):
    g = best_estimator.named_steps["gpr"]
    kernel = getattr(g, "kernel_", g.kernel)
    new = Pipeline([
        ("scaler", StandardScaler()),
        ("gpr", GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            alpha=getattr(g, "alpha", 1e-10),
            n_restarts_optimizer=0,
            random_state=SEED
        ))
    ])
    return new

def grouped_cv_predict(pipe, X, y, groups, alpha_vec: Optional[np.ndarray] = None, n_splits: int = 10):
    # Out-of-fold predictions using GroupKFold
    n_groups = len(np.unique(groups))
    n_splits = min(n_splits, max(2, n_groups))
    gkf = GroupKFold(n_splits=n_splits)

    y_pred = np.full_like(y, np.nan, dtype=float)
    y_std  = np.full_like(y, np.nan, dtype=float)

    for tr_idx, te_idx in gkf.split(X, y, groups):
        m = clone(pipe)
        if isinstance(m.named_steps["gpr"], GaussianProcessRegressor) and alpha_vec is not None:
            m.set_params(**{"gpr__alpha": alpha_vec[tr_idx]})
        m.fit(X[tr_idx], y[tr_idx])
        Xt = m.named_steps["scaler"].transform(X[te_idx])
        mu, sd = m.named_steps["gpr"].predict(Xt, return_std=True)
        y_pred[te_idx] = mu
        y_std[te_idx]  = sd

    mask = np.isfinite(y_pred) & np.isfinite(y_std)
    r2 = r2_score(y[mask], y_pred[mask])
    rmse = RMSE(y[mask], y_pred[mask])
    return y_pred, y_std, r2, rmse

def predict_mu_sd(pipe, Xq):
    Xt = pipe.named_steps["scaler"].transform(Xq)
    mu, sd = pipe.named_steps["gpr"].predict(Xt, return_std=True)
    return mu, sd

# =========================
# Pareto utilities
# =========================
def pareto_front_indices_maxmin(x, y):
    idx = np.argsort(-x)
    xs, ys = x[idx], y[idx]
    keep = []
    best_y = np.inf
    for i in range(len(xs)):
        if ys[i] <= best_y + 1e-12:
            keep.append(idx[i])
            best_y = min(best_y, ys[i])
    return np.array(keep, dtype=int)

def build_pareto_step_dense(px, py, density=200):
    if len(px) <= 1:
        return np.column_stack([px, py])
    order = np.argsort(-px)
    px = px[order]; py = py[order]
    segs = []
    for i in range(len(px)):
        if i > 0:
            yv = np.linspace(py[i-1], py[i], density)
            xv = np.full_like(yv, px[i])
            segs.append(np.column_stack([xv, yv]))
        if i < len(px)-1:
            xh = np.linspace(px[i], px[i+1], density)
            yh = np.full_like(xh, py[i])
            segs.append(np.column_stack([xh, yh]))
    return np.vstack(segs)

def signed_distance_to_pf(r, t, pf_points, norm=None):
    P = pf_points
    if norm is None:
        d = np.sqrt((P[:,0] - r)**2 + (P[:,1] - t)**2)
    else:
        sx, sy = norm
        d = np.sqrt(((P[:,0] - r)/sx)**2 + ((P[:,1] - t)/sy)**2)
    dmin = float(np.min(d))
    sgn = -1.0 if is_dominated_by_pf(r, t, pf_points) else +1.0
    return sgn * dmin

def is_dominated_by_pf(r, t, pf_points):
    rp = pf_points[:, 0]; tp = pf_points[:, 1]
    ge_r = rp >= r - 1e-12
    le_t = tp <= t + 1e-12
    mask = ge_r & le_t
    if not np.any(mask):
        return False
    strict = (rp[mask] > r + 1e-12) | (tp[mask] < t - 1e-12)
    return bool(np.any(strict))

# =========================
# Candidates
# =========================
def round_to_res(X, res):
    Z = X.copy()
    for j in range(Z.shape[1]):
        Z[:, j] = np.round(Z[:, j] / res[j]) * res[j]
    return Z

def sobol_candidates(n):
    m = int(np.ceil(np.log2(max(2, n))))
    engine = qmc.Sobol(d=len(X_FEATURES), scramble=True, seed=SEED)
    U = engine.random_base2(m)
    Xc = BOUNDS_LB + U * (BOUNDS_UB - BOUNDS_LB)
    Xc = round_to_res(Xc, RES)
    Xc = np.unique(Xc, axis=0)
    return Xc

# =========================
# Acquisition A: EI over signed distance
# =========================
def expected_improvement_signed(mu_r, sd_r, mu_t, sd_t, pf_points,
                                s_star=0.0, ns=N_EI_SAMPLES, norm=None):
    n = mu_r.shape[0]
    out = np.zeros(n, dtype=float)
    for i in range(n):
        r = rng.normal(mu_r[i], max(sd_r[i], 1e-12), size=ns)
        t = rng.normal(mu_t[i], max(sd_t[i], 1e-12), size=ns)
        r = np.clip(r, RATE_MIN, RATE_MAX)
        si = np.empty(ns, dtype=float)
        for s in range(ns):
            si[s] = signed_distance_to_pf(r[s], t[s], pf_points, norm=norm)
        out[i] = float(np.mean(np.maximum(0.0, si - s_star)))
    return out

def propose_batched_ei(pipe_r, pipe_t, X_train, y_rate, y_thk,
                       rate_min=RATE_MIN, rate_max=RATE_MAX,
                       pf_density=PF_DENSITY):
    pf_idx = pareto_front_indices_maxmin(y_rate, y_thk)
    px, py = y_rate[pf_idx], y_thk[pf_idx]
    pf_dense = build_pareto_step_dense(px, py, density=pf_density)

    rx_span = max(1e-9, np.max(y_rate) - np.min(y_rate))
    ty_span = max(1e-9, np.max(y_thk) - np.min(y_thk))
    norm_scales = (rx_span, ty_span)
    s_star = 0.0

    Xcand = sobol_candidates(NS_OBOL)
    pr_mu, pr_sd = predict_mu_sd(pipe_r, Xcand)
    pt_mu, pt_sd = predict_mu_sd(pipe_t, Xcand)
    ok = (pr_mu >= rate_min) & (pr_mu <= rate_max)
    Xcand = Xcand[ok]
    pr_mu, pr_sd, pt_mu, pt_sd = pr_mu[ok], pr_sd[ok], pt_mu[ok], pt_sd[ok]

    proposals = []
    Xb = X_train.copy(); yr = y_rate.copy(); yt = y_thk.copy()
    pf_dense_curr = pf_dense.copy(); norm_curr = norm_scales

    for _ in range(N_PROPOSALS):
        if len(Xcand) == 0: break
        ei = expected_improvement_signed(pr_mu, pr_sd, pt_mu, pt_sd,
                                         pf_points=pf_dense_curr,
                                         s_star=s_star, ns=N_EI_SAMPLES,
                                         norm=norm_curr)
        j = int(np.argmax(ei))
        xp = Xcand[j]; pr, ur = pr_mu[j], pr_sd[j]; pt, ut = pt_mu[j], pt_sd[j]
        proposals.append((xp, pr, ur, pt, ut))

        # fantasy update (alpha scalar as requested)
        Xb = np.vstack([Xb, xp.reshape(1, -1)]); yr = np.concatenate([yr, [pr]]); yt = np.concatenate([yt, [pt]])
        pr_pipe = clone_best_kernel(pipe_r); pr_pipe.set_params(gpr__alpha=1e-10); pr_pipe.fit(Xb, yr)
        pt_pipe = clone_best_kernel(pipe_t); pt_pipe.set_params(gpr__alpha=1e-10); pt_pipe.fit(Xb, yt)
        pipe_r, pipe_t = pr_pipe, pt_pipe

        pf_idx_new = pareto_front_indices_maxmin(yr, yt)
        px_new, py_new = yr[pf_idx_new], yt[pf_idx_new]
        pf_dense_curr = build_pareto_step_dense(px_new, py_new, density=pf_density)
        rx_span = max(1e-9, np.max(yr) - np.min(yr)); ty_span = max(1e-9, np.max(yt) - np.min(yt))
        norm_curr = (rx_span, ty_span)

        Xcand = np.delete(Xcand, j, axis=0)
        if len(Xcand) == 0: break
        pr_mu, pr_sd = predict_mu_sd(pipe_r, Xcand); pt_mu, pt_sd = predict_mu_sd(pipe_t, Xcand)
        ok = (pr_mu >= rate_min) & (pr_mu <= rate_max)
        Xcand = Xcand[ok]; pr_mu, pr_sd, pt_mu, pt_sd = pr_mu[ok], pr_sd[ok], pt_mu[ok], pt_sd[ok]

    return proposals

# =========================
# Acquisition B: Distance-max batch (default)
# =========================
def euclidean_distance_to_pf(mu_r, mu_t, pf_points, norm=None):
    n = mu_r.shape[0]
    out = np.zeros(n, dtype=float)
    for i in range(n):
        out[i] = signed_distance_to_pf(mu_r[i], mu_t[i], pf_points, norm=norm)
    return out

def propose_batched_distance(pipe_r, pipe_t, X_train, y_rate, y_thk,
                             rate_min=RATE_MIN, rate_max=RATE_MAX,
                             pf_density=PF_DENSITY):
    pf_idx = pareto_front_indices_maxmin(y_rate, y_thk)
    px, py = y_rate[pf_idx], y_thk[pf_idx]
    pf_dense = build_pareto_step_dense(px, py, density=pf_density)

    rx_span = max(1e-9, np.max(y_rate) - np.min(y_rate))
    ty_span = max(1e-9, np.max(y_thk) - np.min(y_thk))
    norm_scales = (rx_span, ty_span)

    Xcand = sobol_candidates(NS_OBOL)
    pr_mu, pr_sd = predict_mu_sd(pipe_r, Xcand)
    pt_mu, pt_sd = predict_mu_sd(pipe_t, Xcand)
    ok = (pr_mu >= rate_min) & (pr_mu <= rate_max)
    Xcand = Xcand[ok]
    pr_mu, pr_sd, pt_mu, pt_sd = pr_mu[ok], pr_sd[ok], pt_mu[ok], pt_sd[ok]

    proposals = []
    Xb = X_train.copy(); yr = y_rate.copy(); yt = y_thk.copy()
    pf_dense_curr = pf_dense.copy(); norm_curr = norm_scales

    for _ in range(N_PROPOSALS):
        if len(Xcand) == 0: break
        dist_signed = euclidean_distance_to_pf(pr_mu, pt_mu, pf_points=pf_dense_curr, norm=norm_curr)
        j = int(np.argmax(dist_signed))
        xp = Xcand[j]; pr, ur = pr_mu[j], pr_sd[j]; pt, ut = pt_mu[j], pt_sd[j]
        proposals.append((xp, pr, ur, pt, ut))

        # fantasy update (alpha scalar as requested)
        Xb = np.vstack([Xb, xp.reshape(1, -1)]); yr = np.concatenate([yr, [pr]]); yt = np.concatenate([yt, [pt]])
        pr_pipe = clone_best_kernel(pipe_r); pr_pipe.set_params(gpr__alpha=1e-10); pr_pipe.fit(Xb, yr)
        pt_pipe = clone_best_kernel(pipe_t); pt_pipe.set_params(gpr__alpha=1e-10); pt_pipe.fit(Xb, yt)
        pipe_r, pipe_t = pr_pipe, pt_pipe

        pf_idx_new = pareto_front_indices_maxmin(yr, yt)
        px_new, py_new = yr[pf_idx_new], yt[pf_idx_new]
        pf_dense_curr = build_pareto_step_dense(px_new, py_new, density=pf_density)
        rx_span = max(1e-9, np.max(yr) - np.min(yr)); ty_span = max(1e-9, np.max(yt) - np.min(yt))
        norm_curr = (rx_span, ty_span)

        Xcand = np.delete(Xcand, j, axis=0)
        if len(Xcand) == 0: break
        pr_mu, pr_sd = predict_mu_sd(pipe_r, Xcand); pt_mu, pt_sd = predict_mu_sd(pipe_t, Xcand)
        ok = (pr_mu >= rate_min) & (pr_mu <= rate_max)
        Xcand = Xcand[ok]; pr_mu, pr_sd, pt_mu, pt_sd = pr_mu[ok], pr_sd[ok], pt_mu[ok], pt_sd[ok]

    return proposals

# =========================
# Plotting (robust)
# =========================
def parity_plot(y, yhat, ysd, title, fname=None):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float); ysd = np.asarray(ysd, float)
    mask = np.isfinite(y) & np.isfinite(yhat) & np.isfinite(ysd)
    y, yhat, ysd = y[mask], yhat[mask], ysd[mask]
    if y.size == 0:
        print(f"[WARN] {title}: no finite points to plot."); return
    plt.figure(figsize=(6,6))
    plt.errorbar(y, yhat, yerr=ysd, fmt='o', ms=4, elinewidth=0.8, capsize=2, alpha=0.6)
    lo = float(np.nanmin([np.nanmin(y), np.nanmin(yhat)]))
    hi = float(np.nanmax([np.nanmax(y), np.nanmax(yhat)]))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi: lo, hi = 0.0, 1.0
    plt.plot([lo, hi], [lo, hi], 'k--', lw=1)
    plt.xlim(lo, hi); plt.ylim(lo, hi)
    plt.xlabel("Actual"); plt.ylabel("Predicted (Grouped CV)")
    plt.title(title); plt.tight_layout()
    if fname is None: fname = os.path.join(OUTPUT_DIR, _slugify(title) + ".png")
    plt.savefig(fname, dpi=160); plt.show(); plt.close()

def residual_plots(y, yhat, ysd, title_prefix):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float); ysd = np.asarray(ysd, float)
    mask = np.isfinite(y) & np.isfinite(yhat) & np.isfinite(ysd)
    y, yhat, ysd = y[mask], yhat[mask], ysd[mask]
    if y.size == 0:
        print(f"[WARN] {title_prefix}: no finite data for residual plots."); return

    resid = y - yhat
    nz = np.where(ysd <= 1e-12, 1e-12, ysd)
    z = resid / nz

    # Residual vs predicted
    plt.figure(figsize=(6,4.5))
    plt.scatter(yhat, resid, s=18, alpha=0.7)
    plt.axhline(0, color='k', lw=1)
    plt.xlabel("Predicted"); plt.ylabel("Residual (y - ŷ)")
    plt.title(f"{title_prefix} Residual vs Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, _slugify(f"{title_prefix}_resid_vs_pred") + ".png"), dpi=160)
    plt.show(); plt.close()

    # Normalized residuals distribution + coverage
    finite_z = z[np.isfinite(z)]
    if finite_z.size == 0:
        print(f"[WARN] {title_prefix}: no finite normalized residuals."); return
    plt.figure(figsize=(6,4.5))
    xs = np.linspace(-4, 4, 400)
    plt.hist(finite_z, bins=20, density=True, alpha=0.7)
    plt.plot(xs, norm.pdf(xs, 0, 1), 'k--', lw=1)
    cover1 = np.mean(np.abs(finite_z) <= 1.0)
    cover2 = np.mean(np.abs(finite_z) <= 2.0)
    plt.xlabel("Normalized residual z"); plt.ylabel("Density")
    plt.title(f"{title_prefix} Norm Residuals (|z|<=1:{cover1:.2%}, |z|<=2:{cover2:.2%})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, _slugify(f"{title_prefix}_norm_resid_hist") + ".png"), dpi=160)
    plt.show(); plt.close()

def shap_global_and_dependence(pipe, X, feature_names, title_prefix, groups):
    if not _HAS_SHAP:
        print("SHAP not installed, skipping SHAP plots."); return
    f = lambda A: pipe.predict(A)
    n = X.shape[0]
    # stratified background by groups
    grp_ids = np.unique(groups)
    bg_idx = []
    for g in grp_ids:
        idx = np.where(groups == g)[0]
        take = min( max(1, int(np.ceil(len(idx)*0.1))), len(idx) )
        bg_idx.extend(list(rng.choice(idx, size=take, replace=False)))
    bg_idx = np.array(sorted(set(bg_idx)))
    X_bg = X[bg_idx]

    expl = shap.KernelExplainer(f, X_bg)
    n_samp = min(500, n)
    samp_idx = rng.choice(n, size=n_samp, replace=False)
    X_samp = X[samp_idx]
    shap_vals = expl.shap_values(X_samp, nsamples="auto")

    # Summary
    fig = plt.figure()
    shap.summary_plot(shap_vals, X_samp, feature_names=feature_names, show=False)
    plt.title(f"{title_prefix} SHAP Summary")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, _slugify(f"{title_prefix}_shap_summary") + ".png"), dpi=160)
    plt.close(fig)

    # Dependence
    for j, name in enumerate(feature_names):
        fig = plt.figure()
        shap.dependence_plot(j, shap_vals, X_samp, feature_names=feature_names, show=False, display_features=X_samp)
        plt.title(f"{title_prefix} SHAP Dependence: {name}")
        plt.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, _slugify(f"{title_prefix}_shap_dependence_{name}") + ".png"), dpi=160)
        plt.close(fig)

def pareto_plot(x, y, title, proposals=None, fname=None):
    idx = pareto_front_indices_maxmin(x, y)
    px, py = x[idx], y[idx]
    pf_dense = build_pareto_step_dense(px, py, density=200)
    plt.figure(figsize=(7.2,5.2))
    plt.scatter(x, y, s=24, alpha=0.7, label="Observed")
    if len(pf_dense) > 1:
        plt.plot(pf_dense[:,0], pf_dense[:,1], lw=2, label="Pareto step")
    else:
        plt.scatter(px, py, c="C1", s=40, label="Pareto point")
    if proposals:
        ux = [p[1] for p in proposals]; uy = [p[3] for p in proposals]
        plt.scatter(ux, uy, marker='x', s=120, c='red', label="Proposed (pred.)")
    plt.xlabel("AvgEtchRate (maximize)"); plt.ylabel("ThicknessRange (minimize)")
    plt.title(title); plt.legend(); plt.tight_layout()
    if fname is None: fname = os.path.join(OUTPUT_DIR, _slugify(title) + ".png")
    plt.savefig(fname, dpi=160); plt.show(); plt.close()

# =========================
# Main
# =========================
def main():
    data = load_data(CSV_PATH)
    df, X, y_rate, y_thk, groups = data.df, data.X, data.y_rate, data.y_thk, data.groups

    # Alphas (keep thickness alpha as-is)
    alpha_rate_vec, alpha_thk_vec = compute_alpha_vectors(df, data.point_cols)

    # Build models
    pipe_r = tune_model_gpr(X, y_rate, groups)
    pipe_t = tune_model_gpr(X, y_thk,  groups)

    # Grouped-CV diagnostics
    yhat_r, ysd_r, r2_r, rmse_r = grouped_cv_predict(pipe_r, X, y_rate, groups, alpha_vec=alpha_rate_vec, n_splits=10)
    yhat_t, ysd_t, r2_t, rmse_t = grouped_cv_predict(pipe_t, X, y_thk,  groups, alpha_vec=alpha_thk_vec,  n_splits=10)

    # Plots (robust)
    parity_plot(y_rate, yhat_r, ysd_r, f"AvgEtchRate GroupedCV  R2={r2_r:.3f}  RMSE={rmse_r:.3f}")
    parity_plot(y_thk,  yhat_t, ysd_t, f"ThicknessRange GroupedCV  R2={r2_t:.3f}  RMSE={rmse_t:.3f}")
    residual_plots(y_rate, yhat_r, ysd_r, "AvgEtchRate")
    residual_plots(y_thk,  yhat_t, ysd_t, "ThicknessRange")

    # Fit on all data for proposals
    pr_fit = clone_best_kernel(pipe_r)
    if alpha_rate_vec is not None: pr_fit.set_params(**{"gpr__alpha": alpha_rate_vec})
    pr_fit.fit(X, y_rate)

    pt_fit = clone_best_kernel(pipe_t)
    if alpha_thk_vec is not None: pt_fit.set_params(**{"gpr__alpha": alpha_thk_vec})
    pt_fit.fit(X, y_thk)

    # SHAP
    shap_global_and_dependence(pr_fit, X, X_FEATURES, "AvgEtchRate", groups)
    shap_global_and_dependence(pt_fit, X, X_FEATURES, "ThicknessRange", groups)

    # Current PF
    pareto_plot(y_rate, y_thk, "Current Pareto Front")

    # Proposals (default: distance)
    if ACQ_STRATEGY.upper() == "EI":
        proposals = propose_batched_ei(pr_fit, pt_fit, X, y_rate, y_thk,
                                       rate_min=RATE_MIN, rate_max=RATE_MAX,
                                       pf_density=PF_DENSITY)
        title = "Pareto + Proposed Recipes (EI)"
    else:
        proposals = propose_batched_distance(pr_fit, pt_fit, X, y_rate, y_thk,
                                             rate_min=RATE_MIN, rate_max=RATE_MAX,
                                             pf_density=PF_DENSITY)
        title = "Pareto + Proposed Recipes (Distance)"

    pareto_plot(y_rate, y_thk, title, proposals=proposals)

    # Output table
    rows = []
    for p in proposals:
        xp, pr, ur, pt, ut = p
        rows.append({
            "Etch_Avgcf4Flow":  xp[4],
            "Etch_AvgO2Flow":   xp[0],
            "Etch_Avg_Rf2_Pow": xp[2],
            "Etch_Avg_Rf1_Pow": xp[1],
            "Etch_AvgPres":     xp[3],
            "Pred_AvgEtchRate": pr,
            "Unc_AvgEtchRate":  ur,
            "Pred_ThicknessRange": pt,
            "Unc_ThicknessRange":  ut,
        })
    props_df = pd.DataFrame(rows)
    if not props_df.empty:
        print("\nProposed recipes:")
        print(props_df.to_string(index=False))
        props_df.to_csv("proposed_recipes.csv", index=False)
        print("\nSaved: proposed_recipes.csv")

    print(f"\nAvgEtchRate GroupedCV: R2={r2_r:.4f} RMSE={rmse_r:.4f}")
    print(f"ThicknessRange GroupedCV: R2={r2_t:.4f} RMSE={rmse_t:.4f}")

if __name__ == "__main__":
    main()
