#!/usr/bin/env python3
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Tuple, List, Optional

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, RandomizedSearchCV, KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, WhiteKernel, Matern
from scipy.stats import qmc, norm

# Optional SHAP (will skip gracefully if unavailable)
_HAS_SHAP = True
try:
    import shap
except Exception:
    _HAS_SHAP = False

# =========================
# Config
# =========================
CSV_PATH = "full_dataset.csv"     # set absolute path if needed
SEED = 42

# Choose GPR type: "MATERN" (default) or "RBF_SEARCH"
GPR_STRATEGY = "MATERN"

# Choose acquisition: "EI" (default) or "DISTANCE"
ACQ_STRATEGY = "DISTANCE"

# Post-hoc calibration of predictive uncertainty for diagnostics (LOOCV-based)
# If True, scale predicted standard deviations so normalized residuals have unit variance.
CALIBRATE_UNCERTAINTY = True

# Enforce X-space diversity in distance mode by excluding candidates within
# a standardized L2 radius of already selected recipes.
USE_XSPACE_EXCLUSION = True
EXCLUSION_RADIUS_STD = 1.5  # radius in standardized feature space

NS_OBOL = 262144          # Sobol candidates (pre snap/dedup)
N_EI_SAMPLES = 256       # Monte Carlo samples for EI of signed distance
N_PROPOSALS = 3          # batch size
PF_DENSITY = 200         # densify PF step curve

X_FEATURES = [
    "Etch_AvgO2Flow",
    "Etch_Avg_Rf1_Pow",
    "Etch_Avg_Rf2_Pow",
    "Etch_AvgPres",
    "Etch_Avgcf4Flow",
]

Y1_COL = "AvgEtchRate"      # maximize
Y2_COL = "RangeEtchRate"    # to be multiplied to thickness
THICKNESS_MULT = 5.0        # thickness range = RangeEtchRate * 5

RATE_MIN = 35.0
RATE_MAX = 110.0

# Parameter bounds (order matches X_FEATURES)
BOUNDS_LB = np.array([10.0, 0.0, 50.0, 4.0, 10.0])
BOUNDS_UB = np.array([90.0, 100.0, 700.0, 100.0, 90.0])
RES = np.array([0.1, 1.0, 1.0, 0.1, 0.1])

OUTPUT_DIR = "plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

rng = np.random.default_rng(SEED)
plt.rcParams["figure.dpi"] = 100

def RMSE(y_true, y_pred): return mean_squared_error(y_true, y_pred, squared=False)

# =========================
# Post-hoc uncertainty calibration (LOOCV-based)
# =========================
def compute_calibration_scale(y: np.ndarray, yhat: np.ndarray, ysd: np.ndarray) -> float:
    """
    Compute a scalar factor s such that scaling predicted stds by s makes
    normalized residuals have unit standard deviation.

    New sigma = s * old sigma, with s = std(z), z = (y - yhat) / sigma.
    """
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    ysd = np.asarray(ysd, float)
    nz = np.where(ysd <= 1e-12, 1e-12, ysd)
    z = (y - yhat) / nz
    finite = np.isfinite(z)
    if not np.any(finite):
        return 1.0
    s = float(np.std(z[finite], ddof=1))
    if not np.isfinite(s) or s <= 0.0:
        return 1.0
    return s

# =========================
# Data
# =========================
@dataclass
class DataBundle:
    df: pd.DataFrame
    X: np.ndarray
    y_rate: np.ndarray
    y_thk: np.ndarray
    point_cols: List[str]

def _num(df, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

def load_data(path) -> DataBundle:
    df = pd.read_csv(path)
    need = X_FEATURES + [Y1_COL, Y2_COL]
    _num(df, need)
    # Optional 41-point columns if available
    point_cols = [f"Rate_{i}_nm_per_min" for i in range(1, 42)]
    if all(col in df.columns for col in point_cols):
        _num(df, point_cols)
    else:
        point_cols = []

    df = df.dropna(subset=need).copy()
    df["ThicknessRange"] = df[Y2_COL] * THICKNESS_MULT
    X = df[X_FEATURES].to_numpy(float)
    y_rate = df[Y1_COL].to_numpy(float)
    y_thk  = df["ThicknessRange"].to_numpy(float)
    return DataBundle(df=df, X=X, y_rate=y_rate, y_thk=y_thk, point_cols=point_cols)

# =========================
# GPR modeling
# =========================
def make_gpr_matern():
    # From your code 1 (default choice)
    kernel = C(1.0, (1e-2, 1e2)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5) \
             + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1e0))
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("gpr", GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=SEED, n_restarts_optimizer=1))
    ])
    return pipe

def make_gpr_rbf_grid():
    # Original code 2 approach with a small kernel grid
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("gpr", GaussianProcessRegressor(
            kernel=C(1.0)*RBF(1.0) + WhiteKernel(1e-3),
            normalize_y=True,
            alpha=1e-10,
            n_restarts_optimizer=5,
            random_state=SEED
        ))
    ])
    kernels = []
    for c in (0.1, 1.0, 10.0):
        for l in (0.1, 1.0, 10.0):
            for w in (1e-5, 1e-3, 1e-1):
                kernels.append(C(c)*RBF(l) + WhiteKernel(w))
    grid = {"gpr__kernel": kernels}
    return pipe, grid

def tune_model_gpr(X, y):
    # Default to Matern; otherwise do RBF grid search
    if GPR_STRATEGY.upper() == "MATERN":
        return make_gpr_matern()
    pipe, grid = make_gpr_rbf_grid()
    cv = KFold(n_splits=min(3, max(2, len(y))), shuffle=True, random_state=SEED)
    rs = RandomizedSearchCV(
        pipe, param_distributions=grid,
        n_iter=min(25, len(grid["gpr__kernel"])),
        scoring="neg_root_mean_squared_error",
        cv=cv, n_jobs=-1, random_state=SEED, refit=True, verbose=0
    )
    rs.fit(X, y)
    return rs.best_estimator_

def clone_best_kernel(best_estimator):
    g = best_estimator.named_steps["gpr"]
    kernel = g.kernel_ if hasattr(g, "kernel_") else g.kernel
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

def loocv_predict(pipe, X, y, alpha_vec: Optional[np.ndarray] = None):
    loo = LeaveOneOut()
    y_pred = np.zeros_like(y, dtype=float)
    y_std  = np.zeros_like(y, dtype=float)
    for tr, te in loo.split(X):
        m = clone(pipe)
        if isinstance(m.named_steps["gpr"], GaussianProcessRegressor) and alpha_vec is not None:
            m.set_params(**{"gpr__alpha": alpha_vec[tr]})
        m.fit(X[tr], y[tr])
        Xt = m.named_steps["scaler"].transform(X[te])
        mu, sd = m.named_steps["gpr"].predict(Xt, return_std=True)
        y_pred[te] = mu
        y_std[te]  = sd
    r2 = r2_score(y, y_pred)
    rmse = RMSE(y, y_pred)
    return y_pred, y_std, r2, rmse

def predict_mu_sd(pipe, Xq):
    Xt = pipe.named_steps["scaler"].transform(Xq)
    mu, sd = pipe.named_steps["gpr"].predict(Xt, return_std=True)
    return mu, sd

# =========================
# Alpha vectors from point data (heteroscedastic noise)
# =========================
def compute_alpha_vectors(df: pd.DataFrame, point_cols: List[str]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    - For rate: CLT variance of the mean from 41 points per sample.
    - For thickness: relative error propagated onto ThicknessRange target.
    """
    if not point_cols:
        return None, None
    pts = df[point_cols].to_numpy(dtype=float)
    std41 = np.nanstd(pts, axis=1, ddof=1)
    alpha_rate = (std41 / np.sqrt(41.0)) ** 2  # CLT on the mean
    avg_rate = df[Y1_COL].to_numpy(dtype=float)
    rng_rate = df[Y2_COL].to_numpy(dtype=float)
    eps = 1e-12
    denom = np.where(np.abs(avg_rate) < eps, eps, avg_rate)
    alpha_range_rel = (rng_rate / denom) ** 2     # relative variance
    alpha_thickness = (alpha_range_rel * (THICKNESS_MULT ** 2)).astype(float)
    return alpha_rate.astype(float), alpha_thickness

# =========================
# Pareto helpers (max rate, min thickness)
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

def is_dominated_by_pf(r, t, pf_points):
    rp = pf_points[:, 0]; tp = pf_points[:, 1]
    ge_r = rp >= r - 1e-12
    le_t = tp <= t + 1e-12
    mask = ge_r & le_t
    if not np.any(mask):
        return False
    strict = (rp[mask] > r + 1e-12) | (tp[mask] < t - 1e-12)
    return bool(np.any(strict))

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

def signed_distance_to_pf_with_extras(r, t, pf_points, extra_points=None, norm=None):
    """
    Signed distance to the union of PF curve and any extra anchor points.
    Sign is determined by PF dominance (not by extras) to preserve the
    max-rate/min-thickness objective orientation, while the extras act
    as repulsive anchors that encourage spread among proposals.
    """
    if extra_points is None or len(extra_points) == 0:
        return signed_distance_to_pf(r, t, pf_points, norm=norm)
    if norm is None:
        P = pf_points
        E = np.asarray(extra_points, dtype=float)
        d_pf = np.sqrt((P[:,0] - r)**2 + (P[:,1] - t)**2)
        d_ex = np.sqrt((E[:,0] - r)**2 + (E[:,1] - t)**2)
    else:
        sx, sy = norm
        P = pf_points
        E = np.asarray(extra_points, dtype=float)
        d_pf = np.sqrt(((P[:,0] - r)/sx)**2 + ((P[:,1] - t)/sy)**2)
        d_ex = np.sqrt(((E[:,0] - r)/sx)**2 + ((E[:,1] - t)/sy)**2)
    dmin = float(np.min(np.concatenate([d_pf, d_ex])))
    sgn = -1.0 if is_dominated_by_pf(r, t, pf_points) else +1.0
    return sgn * dmin

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
# Acquisition A: EI over signed distance to PF
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
    # PF + normalization
    pf_idx = pareto_front_indices_maxmin(y_rate, y_thk)
    px, py = y_rate[pf_idx], y_thk[pf_idx]
    pf_dense = build_pareto_step_dense(px, py, density=pf_density)

    rx_span = max(1e-9, np.max(y_rate) - np.min(y_rate))
    ty_span = max(1e-9, np.max(y_thk) - np.min(y_thk))
    norm_scales = (rx_span, ty_span)

    s_star = 0.0  # PF points are ~0

    # Candidates + predictions
    Xcand = sobol_candidates(NS_OBOL)
    pr_mu, pr_sd = predict_mu_sd(pipe_r, Xcand)
    pt_mu, pt_sd = predict_mu_sd(pipe_t, Xcand)
    ok = (pr_mu >= rate_min) & (pr_mu <= rate_max)
    Xcand = Xcand[ok]
    pr_mu, pr_sd, pt_mu, pt_sd = pr_mu[ok], pr_sd[ok], pt_mu[ok], pt_sd[ok]

    proposals = []
    Xb = X_train.copy()
    yr = y_rate.copy()
    yt = y_thk.copy()
    pf_dense_curr = pf_dense.copy()
    norm_curr = norm_scales

    for _ in range(N_PROPOSALS):
        if len(Xcand) == 0:
            break

        ei = expected_improvement_signed(pr_mu, pr_sd, pt_mu, pt_sd,
                                         pf_points=pf_dense_curr,
                                         s_star=s_star, ns=N_EI_SAMPLES,
                                         norm=norm_curr)
        j = int(np.argmax(ei))
        xp = Xcand[j]
        pr, ur = pr_mu[j], pr_sd[j]
        pt, ut = pt_mu[j], pt_sd[j]
        proposals.append((xp, pr, ur, pt, ut))

        # Fantasy update (alpha = scalar 1e-10 as requested)
        Xb = np.vstack([Xb, xp.reshape(1, -1)])
        yr = np.concatenate([yr, [pr]])
        yt = np.concatenate([yt, [pt]])

        pr_pipe = clone_best_kernel(pipe_r)
        pr_pipe.set_params(gpr__alpha=1e-10)
        pr_pipe.fit(Xb, yr)

        pt_pipe = clone_best_kernel(pipe_t)
        pt_pipe.set_params(gpr__alpha=1e-10)
        pt_pipe.fit(Xb, yt)

        pipe_r, pipe_t = pr_pipe, pt_pipe

        # Update PF and normalization
        pf_idx_new = pareto_front_indices_maxmin(yr, yt)
        px_new, py_new = yr[pf_idx_new], yt[pf_idx_new]
        pf_dense_curr = build_pareto_step_dense(px_new, py_new, density=pf_density)
        rx_span = max(1e-9, np.max(yr) - np.min(yr))
        ty_span = max(1e-9, np.max(yt) - np.min(yt))
        norm_curr = (rx_span, ty_span)

        # Remove chosen & refresh predictions
        Xcand = np.delete(Xcand, j, axis=0)
        if len(Xcand) == 0:
            break
        pr_mu, pr_sd = predict_mu_sd(pipe_r, Xcand)
        pt_mu, pt_sd = predict_mu_sd(pipe_t, Xcand)
        ok = (pr_mu >= rate_min) & (pr_mu <= rate_max)
        Xcand = Xcand[ok]
        pr_mu, pr_sd, pt_mu, pt_sd = pr_mu[ok], pr_sd[ok], pt_mu[ok], pt_sd[ok]

    return proposals

# =========================
# Acquisition B: Euclidean distance maximization (normalized)
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
    selected_obj_points = []  # list of (rate_pred, thickness_pred) for repulsion
    selected_X = []

    # Prepare standardizer for X-space distances (reuse model scaler if available)
    scaler = None
    if hasattr(pipe_r, "named_steps") and "scaler" in pipe_r.named_steps:
        scaler = pipe_r.named_steps["scaler"]
    Xb = X_train.copy()
    yr = y_rate.copy()
    yt = y_thk.copy()
    pf_dense_curr = pf_dense.copy()
    norm_curr = norm_scales

    for _ in range(N_PROPOSALS):
        if len(Xcand) == 0:
            break

        # Apply X-space exclusion BEFORE computing distances
        if USE_XSPACE_EXCLUSION and len(selected_X) > 0 and len(Xcand) > 0:
            if scaler is not None:
                Xsel_std = scaler.transform(np.vstack(selected_X))
                Xcand_std = scaler.transform(Xcand)
            else:
                # fallback: standardize by bounds
                Xsel_std = (np.vstack(selected_X) - BOUNDS_LB) / (BOUNDS_UB - BOUNDS_LB)
                Xcand_std = (Xcand - BOUNDS_LB) / (BOUNDS_UB - BOUNDS_LB)
            # compute min distance to any selected point
            dmin = np.full(Xcand.shape[0], np.inf, dtype=float)
            for xs in Xsel_std:
                d = np.linalg.norm(Xcand_std - xs, axis=1)
                dmin = np.minimum(dmin, d)
            keep = dmin >= EXCLUSION_RADIUS_STD
            if np.any(~keep):
                Xcand = Xcand[keep]
                pr_mu = pr_mu[keep]
                pr_sd = pr_sd[keep]
                pt_mu = pt_mu[keep]
                pt_sd = pt_sd[keep]

        # Distance to union of PF and already selected proposal predictions
        dist_vals = np.zeros_like(pr_mu)
        if len(selected_obj_points) == 0:
            dist_vals = euclidean_distance_to_pf(pr_mu, pt_mu, pf_points=pf_dense_curr, norm=norm_curr)
        else:
            dist_vals = np.array([
                signed_distance_to_pf_with_extras(pr_mu[i], pt_mu[i],
                                                  pf_points=pf_dense_curr,
                                                  extra_points=np.array(selected_obj_points),
                                                  norm=norm_curr)
                for i in range(len(pr_mu))
            ], dtype=float)
        j = int(np.argmax(dist_vals))
        xp = Xcand[j]
        pr, ur = pr_mu[j], pr_sd[j]
        pt, ut = pt_mu[j], pt_sd[j]
        proposals.append((xp, pr, ur, pt, ut))
        selected_obj_points.append((pr, pt))
        selected_X.append(xp.copy())

        # Fantasy update (alpha = scalar 1e-10)
        Xb = np.vstack([Xb, xp.reshape(1, -1)])
        yr = np.concatenate([yr, [pr]])
        yt = np.concatenate([yt, [pt]])

        pr_pipe = clone_best_kernel(pipe_r)
        pr_pipe.set_params(gpr__alpha=1e-10)
        pr_pipe.fit(Xb, yr)

        pt_pipe = clone_best_kernel(pipe_t)
        pt_pipe.set_params(gpr__alpha=1e-10)
        pt_pipe.fit(Xb, yt)

        pipe_r, pipe_t = pr_pipe, pt_pipe

        # Update PF and normalization
        pf_idx_new = pareto_front_indices_maxmin(yr, yt)
        px_new, py_new = yr[pf_idx_new], yt[pf_idx_new]
        pf_dense_curr = build_pareto_step_dense(px_new, py_new, density=pf_density)
        rx_span = max(1e-9, np.max(yr) - np.min(yr))
        ty_span = max(1e-9, np.max(yt) - np.min(yt))
        norm_curr = (rx_span, ty_span)

        # Remove chosen candidate
        Xcand = np.delete(Xcand, j, axis=0)
        pr_mu = np.delete(pr_mu, j, axis=0)
        pr_sd = np.delete(pr_sd, j, axis=0)
        pt_mu = np.delete(pt_mu, j, axis=0)
        pt_sd = np.delete(pt_sd, j, axis=0)
        if len(Xcand) == 0:
            break
        pr_mu, pr_sd = predict_mu_sd(pipe_r, Xcand)
        pt_mu, pt_sd = predict_mu_sd(pipe_t, Xcand)
        ok = (pr_mu >= rate_min) & (pr_mu <= rate_max)
        Xcand = Xcand[ok]
        pr_mu, pr_sd, pt_mu, pt_sd = pr_mu[ok], pr_sd[ok], pt_mu[ok], pt_sd[ok]

    return proposals

# =========================
# Plotting
# =========================
def _slugify(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s).strip("_")

def parity_plot(y, yhat, ysd, title, fname=None):
    plt.figure(figsize=(6,6))
    plt.errorbar(y, yhat, yerr=ysd, fmt='o', ms=4, elinewidth=0.8, capsize=2, alpha=0.5)
    lo = min(float(np.min(y)), float(np.min(yhat)))
    hi = max(float(np.max(y)), float(np.max(yhat)))
    plt.plot([lo,hi],[lo,hi],'k--',lw=1)
    plt.xlabel("Actual")
    plt.ylabel("Predicted (LOOCV)")
    plt.title(title)
    plt.tight_layout()
    if fname is None:
        fname = os.path.join(OUTPUT_DIR, _slugify(title) + ".png")
    plt.savefig(fname, dpi=160)
    plt.show()

def residual_plots(y, yhat, ysd, title_prefix):
    resid = y - yhat
    nz = np.where(ysd <= 1e-12, 1e-12, ysd)
    z = resid / nz

    # Residual vs predicted
    plt.figure(figsize=(6,4.5))
    plt.scatter(yhat, resid, s=18, alpha=0.7)
    plt.axhline(0, color='k', lw=1)
    plt.xlabel("Predicted")
    plt.ylabel("Residual (y - yhat)")
    plt.title(f"{title_prefix} Residual vs Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, _slugify(f"{title_prefix}_resid_vs_pred") + ".png"), dpi=160)
    plt.show()

    # Normalized residuals distribution + coverage
    finite_z = z[np.isfinite(z)]
    if finite_z.size == 0:
        print(f"[WARN] {title_prefix}: no finite normalized residuals.")
        return
    plt.figure(figsize=(6,4.5))
    xs = np.linspace(-4, 4, 400)
    plt.hist(finite_z, bins=20, density=True, alpha=0.7)
    plt.plot(xs, norm.pdf(xs, 0, 1), 'k--', lw=1)
    cover1 = np.mean(np.abs(finite_z) <= 1.0)
    cover2 = np.mean(np.abs(finite_z) <= 2.0)
    plt.xlabel("Normalized residual ( (y - yhat) / sigma )")
    plt.ylabel("Density")
    plt.title(f"{title_prefix} Normalized Residuals (|z|<=1:{cover1:.2%}, |z|<=2:{cover2:.2%})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, _slugify(f"{title_prefix}_norm_resid_hist") + ".png"), dpi=160)
    plt.show()

def partial_residual_plots_with_shap(pipe, X, y, yhat, title_prefix, feature_names):
    if not _HAS_SHAP:
        print("SHAP not installed, skipping partial residual plots.")
        return
    # Model-agnostic explainer
    f = lambda A: pipe.predict(A)
    n = X.shape[0]
    n_bg = min(100, n)
    bg_idx = rng.choice(n, size=n_bg, replace=False)
    X_bg = X[bg_idx]

    expl = shap.KernelExplainer(f, X_bg)
    n_samp = min(500, n)
    samp_idx = rng.choice(n, size=n_samp, replace=False)
    X_samp = X[samp_idx]
    shap_vals = expl.shap_values(X_samp, nsamples="auto")

    resid_all = (y - yhat)
    resid_samp = resid_all[samp_idx]

    for j, name in enumerate(feature_names):
        # Reduced/partial residual style: residual + shap contribution of feature j
        pr = resid_samp + shap_vals[:, j]
        plt.figure(figsize=(6,4.5))
        plt.scatter(X_samp[:, j], pr, s=18, alpha=0.7)
        plt.xlabel(name)
        plt.ylabel("Partial residual")
        plt.title(f"{title_prefix} Partial Residual: {name}")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, _slugify(f"{title_prefix}_partial_residual_{name}") + ".png"), dpi=160)
        plt.show()

def shap_global_and_dependence(pipe, X, feature_names, title_prefix):
    if not _HAS_SHAP:
        print("SHAP not installed, skipping SHAP plots.")
        return
    f = lambda A: pipe.predict(A)
    n = X.shape[0]
    n_bg = min(100, n)
    bg_idx = rng.choice(n, size=n_bg, replace=False)
    X_bg = X[bg_idx]

    expl = shap.KernelExplainer(f, X_bg)
    n_samp = min(500, n)
    samp_idx = rng.choice(n, size=n_samp, replace=False)
    X_samp = X[samp_idx]
    shap_vals = expl.shap_values(X_samp, nsamples="auto")

    # Global summary
    plt.figure()
    shap.summary_plot(shap_vals, X_samp, feature_names=feature_names, show=False)
    plt.title(f"{title_prefix} SHAP Summary")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, _slugify(f"{title_prefix}_shap_summary") + ".png"), dpi=160)
    plt.show()

    # Dependence per feature
    for j, name in enumerate(feature_names):
        plt.figure()
        shap.dependence_plot(j, shap_vals, X_samp, feature_names=feature_names, show=False)
        plt.title(f"{title_prefix} SHAP Dependence: {name}")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, _slugify(f"{title_prefix}_shap_dependence_{name}") + ".png"), dpi=160)
        plt.show()

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
        ux = [p[1] for p in proposals]
        uy = [p[3] for p in proposals]
        plt.scatter(ux, uy, marker='x', s=120, c='red', label="Proposed (predicted)")
    plt.xlabel("AvgEtchRate (maximize)")
    plt.ylabel("ThicknessRange (minimize)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if fname is None:
        fname = os.path.join(OUTPUT_DIR, _slugify(title) + ".png")
    plt.savefig(fname, dpi=160)
    plt.show()

# =========================
# Main
# =========================
def main():
    data = load_data(CSV_PATH)
    df = data.df
    X = data.X
    y_rate = data.y_rate
    y_thk = data.y_thk

    # Heteroscedastic alphas (CLT + relative error) for initial training/LOOCV
    alpha_rate_vec, alpha_thk_vec = compute_alpha_vectors(df, data.point_cols)

    # Build models
    pipe_r = tune_model_gpr(X, y_rate)
    pipe_t = tune_model_gpr(X, y_thk)

    # LOOCV with alphas
    yhat_r, ysd_r, r2_r, rmse_r = loocv_predict(pipe_r, X, y_rate, alpha_vec=alpha_rate_vec)
    yhat_t, ysd_t, r2_t, rmse_t = loocv_predict(pipe_t, X, y_thk,  alpha_vec=alpha_thk_vec)

    # Optional post-hoc calibration of uncertainties for diagnostics
    if CALIBRATE_UNCERTAINTY:
        scale_r = compute_calibration_scale(y_rate, yhat_r, ysd_r)
        scale_t = compute_calibration_scale(y_thk,  yhat_t, ysd_t)
        if np.isfinite(scale_r) and scale_r > 0:
            ysd_r = ysd_r * scale_r
        if np.isfinite(scale_t) and scale_t > 0:
            ysd_t = ysd_t * scale_t

    # Diagnostics
    parity_plot(y_rate, yhat_r, ysd_r, f"AvgEtchRate LOOCV  R2={r2_r:.3f}  RMSE={rmse_r:.3f}")
    parity_plot(y_thk,  yhat_t, ysd_t, f"ThicknessRange LOOCV  R2={r2_t:.3f}  RMSE={rmse_t:.3f}")

    residual_plots(y_rate, yhat_r, ysd_r, "AvgEtchRate")
    residual_plots(y_thk,  yhat_t, ysd_t, "ThicknessRange")

    # Refit on all data with heteroscedastic alpha (initial fit)
    pr_fit = clone_best_kernel(pipe_r)
    if alpha_rate_vec is not None:
        pr_fit.set_params(**{"gpr__alpha": alpha_rate_vec})
    pr_fit.fit(X, y_rate)

    pt_fit = clone_best_kernel(pipe_t)
    if alpha_thk_vec is not None:
        pt_fit.set_params(**{"gpr__alpha": alpha_thk_vec})
    pt_fit.fit(X, y_thk)

    # SHAP global + dependence (skips if shap unavailable)
    shap_global_and_dependence(pr_fit, X, X_FEATURES, "AvgEtchRate")
    shap_global_and_dependence(pt_fit, X, X_FEATURES, "ThicknessRange")

    # Partial residual / reduced residual (via SHAP contributions)
    partial_residual_plots_with_shap(pr_fit, X, y_rate, yhat_r, "AvgEtchRate", X_FEATURES)
    partial_residual_plots_with_shap(pt_fit, X, y_thk,  yhat_t, "ThicknessRange", X_FEATURES)

    # Current PF
    pareto_plot(y_rate, y_thk, "Current Pareto Front")

    # Proposals
    if ACQ_STRATEGY.upper() == "DISTANCE":
        proposals = propose_batched_distance(pr_fit, pt_fit, X, y_rate, y_thk,
                                             rate_min=RATE_MIN, rate_max=RATE_MAX,
                                             pf_density=PF_DENSITY)
        title = "Pareto + Proposed Recipes (Distance)"
    else:
        proposals = propose_batched_ei(pr_fit, pt_fit, X, y_rate, y_thk,
                                       rate_min=RATE_MIN, rate_max=RATE_MAX,
                                       pf_density=PF_DENSITY)
        title = "Pareto + Proposed Recipes (EI)"

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

    # Analyze normalized residuals for both models
    resid_r = y_rate - yhat_r
    resid_t = y_thk - yhat_t
    nz_r = np.where(ysd_r <= 1e-12, 1e-12, ysd_r)
    nz_t = np.where(ysd_t <= 1e-12, 1e-12, ysd_t)
    z_r = resid_r / nz_r
    z_t = resid_t / nz_t
    
    # Count finite normalized residuals
    finite_r = np.isfinite(z_r)
    finite_t = np.isfinite(z_t)
    finite_both = finite_r & finite_t
    
    if np.any(finite_r):
        z_r_finite = z_r[finite_r]
        median_r = np.median(np.abs(z_r_finite))
        bottom50_r = np.sum(np.abs(z_r_finite) <= median_r)
    else:
        bottom50_r = 0
    
    if np.any(finite_t):
        z_t_finite = z_t[finite_t]
        median_t = np.median(np.abs(z_t_finite))
        bottom50_t = np.sum(np.abs(z_t_finite) <= median_t)
    else:
        bottom50_t = 0
    
    if np.any(finite_both):
        z_r_both = z_r[finite_both]
        z_t_both = z_t[finite_both]
        median_r_both = np.median(np.abs(z_r_both))
        median_t_both = np.median(np.abs(z_t_both))
        bottom50_both = np.sum((np.abs(z_r_both) <= median_r_both) & 
                              (np.abs(z_t_both) <= median_t_both))
    else:
        bottom50_both = 0
    
    print(f"\nAvgEtchRate LOOCV: R2={r2_r:.4f} RMSE={rmse_r:.4f}")
    print(f"ThicknessRange LOOCV: R2={r2_t:.4f} RMSE={rmse_t:.4f}")
    print(f"\nNormalized Residual Analysis:")
    print(f"AvgEtchRate bottom 50th percentile: {bottom50_r} points")
    print(f"ThicknessRange bottom 50th percentile: {bottom50_t} points")
    print(f"Both models bottom 50th percentile: {bottom50_both} points")

if __name__ == "__main__":
    main()
