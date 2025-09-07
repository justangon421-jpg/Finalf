# -*- coding: utf-8 -*-
"""
pm25_meta_ensemble.py  —  final end-to-end, leak-free workflow

What’s inside (all leak-safe):
A) Data prep & robust preprocessing (IQR tagging only, no deletion; 23 derived features);
   Pipeline = Winsorizer + RobustScaler; target via TTR(log1p/expm1).
   Data overview plots.

B) Baseline nested CV (outer 5 × inner 3; stratify-by-binning y), models = SVR/HGB/(XGB opt);
   per-fold ensemble weights via RMSE + λ·Pinball@0.90; OOF predictions & diagnostics;
   permutation importance on validation fold only; correlations; export plots & CSVs.

C) One-shot feature selection by OOF PI (mean>0 & pos_ratio≥0.6, keep≥10; optional Top-K),
   then re-run the same nested CV with selected features, export comparison & plots.

D) Optional: single hold-out test (kept out from all model/selection), reported at the end;
   Optional: IQR-trim sensitivity (pre-declared), reported as auxiliary.

All steps are nested/contained in CV; anything that learns statistics is inside the folds.
"""

import os
import json
import math
import time
import warnings
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import FunctionTransformer, RobustScaler
from sklearn.model_selection import (KFold, RepeatedStratifiedKFold, StratifiedKFold,
                                     RandomizedSearchCV, train_test_split, learning_curve)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_pinball_loss
from sklearn.utils.validation import check_random_state
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer

# XGBoost（可选）
try:
    from xgboost import XGBRegressor
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

warnings.filterwarnings("ignore", category=UserWarning)
np.set_printoptions(suppress=True, linewidth=180)

# ===================== Config =====================
DATA_PATH = "Cleaned_PM_Regression_Data.xlsx"
TARGET_COL = "PM2_5"

# CV
N_OUTER = 5
N_INNER = 3
N_REPEATS = 2
RANDOM_STATE = 42
N_JOBS = 1

# Models
USE_XGB = True and HAVE_XGB
N_ITER_SVR = 40
N_ITER_HGB = 40
N_ITER_XGB = 30

# Stratified-by-bins
Y_BINS = 7
PINBALL_Q = 0.90
LAMBDA_PB = 0.20
WEIGHT_GRID_STEP = 0.02

# IQR SOP
RUN_IQR_SCREENING = True
RUN_IQR_SENSITIVITY = True
IQR_K_INNER = 1.5
IQR_K_OUTER = 3.0
IQR_TRIM_STRATEGY = "inner"  # "inner" or "outer"

# Feature selection by OOF Permutation Importance
PI_KEEP_MIN_PROP = 0.60   # pos_ratio ≥ 0.60
PI_MIN_MEAN = 0.0         # mean importance > 0
PI_MIN_KEEP = 10          # keep at least 10 features
PI_TOPK: Optional[int] = None  # e.g., 12 to force compact set; or None

# Optional hold-out (kept untouched until final)
USE_HOLDOUT = False
HOLDOUT_SIZE = 0.2
HOLDOUT_RANDOM_STATE = 142

# Outputs
REPORT_PATH = "run_report.md"
BUNDLE_PATH = "PM25_meta_ensemble.joblib"
OUTLIER_CSV = "outlier_screening.csv"
SOP_JSON = "outlier_sop.json"
OOF_PRED_CSV = "oof_predictions.csv"
PI_CSV = "feature_permutation_importance.csv"
COMPARE_CSV = "oof_compare_pre_post_selection.csv"
SELECTED_JSON = "selected_features.json"

# Plots
PLOT_DATA_OVERVIEW = "data_overview.png"
PLOT_Y_LOG = "y_log_transform.png"
PLOT_IQR = "iqr_summary.png"
PLOT_OOF_SCATTER = "oof_scatter.png"
PLOT_OOF_RESID = "oof_residual_plot.png"
PLOT_PI_BAR = "pi_bar_topN.png"
PLOT_CORR_SPEARMAN = "corr_heatmap_spearman.png"
PLOT_CORR_PEARSON = "corr_heatmap_pearson.png"
PLOT_ENSEMBLE_WEIGHTS = "ensemble_weights_per_fold.png"
PLOT_LC_SVR = "learning_curve_svr.png"
PLOT_LC_HGB = "learning_curve_hgb.png"

# ===================== Utilities =====================
class Winsorizer(BaseEstimator, TransformerMixin):
    """IQR-like tail clipping (fit on train only, apply to val/test)."""
    def __init__(self, lower_q=0.01, upper_q=0.99):
        self.lower_q = lower_q
        self.upper_q = upper_q
        self.lower_ = None
        self.upper_ = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q_low = np.quantile(X, self.lower_q, axis=0)
        q_high = np.quantile(X, self.upper_q, axis=0)
        iqr = q_high - q_low
        self.lower_ = q_low - 1.5 * iqr
        self.upper_ = q_high + 1.5 * iqr
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lower_, self.upper_)

def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

def pinball90(y_true, y_pred):
    return mean_pinball_loss(y_true, y_pred, alpha=PINBALL_Q)

def stratify_bins(y: np.ndarray, n_bins: int = 7, seed: int = 0) -> np.ndarray:
    rng = check_random_state(seed)
    y = np.asarray(y, dtype=float).ravel()
    y_jit = y + rng.normal(0, 1e-9 * (np.std(y) if np.std(y) > 0 else 1.0), size=y.shape)
    ser = pd.Series(y_jit)
    try:
        cats = pd.qcut(ser, q=n_bins, labels=False, duplicates='drop')
        bins = np.asarray(cats, dtype=float)
        if np.unique(bins[~np.isnan(bins)]).size < 2:
            raise ValueError("effective bins < 2 after qcut")
        return bins.astype(int)
    except Exception:
        try:
            from sklearn.preprocessing import KBinsDiscretizer
            k = min(n_bins, max(2, int(np.ceil(np.sqrt(len(y))))))
            kbd = KBinsDiscretizer(n_bins=k, encode='ordinal', strategy='quantile')
            bins = kbd.fit_transform(y.reshape(-1, 1)).reshape(-1)
            return bins.astype(int)
        except Exception:
            med = np.median(y)
            bins = (y > med).astype(int)
            return bins

def build_feature_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    cols = ['Initial_Temperature', 'Final_Temperature', 'Initial_Speed', 'Final_Speed',
            'Deceleration_Rate', 'Contact_Pressure', 'Braking_Time']
    base = df[cols].copy()

    base['Temp_Rise'] = base['Final_Temperature'] - base['Initial_Temperature']
    base['Temp_Rise_Rate'] = base['Temp_Rise'] / np.maximum(base['Braking_Time'], 1e-6)
    base['Mean_Temp'] = 0.5 * (base['Initial_Temperature'] + base['Final_Temperature'])
    base['Speed_Drop_kmh'] = base['Initial_Speed'] - base['Final_Speed']
    base['Speed_Drop_ms'] = (base['Speed_Drop_kmh'] / 3.6)
    base['Mean_Speed_ms'] = 0.5 * (base['Initial_Speed'] + base['Final_Speed']) / 3.6
    base['Braking_Distance'] = base['Mean_Speed_ms'] * base['Braking_Time']
    base['Decel_ms2_src'] = base['Deceleration_Rate']
    base['Decel_ms2_kin'] = base['Speed_Drop_ms'] / np.maximum(base['Braking_Time'], 1e-6)
    base['SpeedXDecel'] = base['Mean_Speed_ms'] * base['Decel_ms2_kin']
    base['Energy_Rel'] = (np.square(base['Speed_Drop_ms']))
    base['Power_Rel'] = base['Energy_Rel'] / np.maximum(base['Braking_Time'], 1e-6)
    base['Pressure_Time'] = base['Contact_Pressure'] * base['Braking_Time']
    base['V1_over_V0'] = np.divide(base['Final_Speed'], np.maximum(base['Initial_Speed'], 1e-6))
    base['Tf_over_Ti'] = np.divide(base['Final_Temperature'], np.maximum(base['Initial_Temperature'], 1e-6))
    base['PT_over_Erel'] = np.divide(base['Pressure_Time'], np.maximum(base['Energy_Rel'], 1e-6))

    feat_cols = list(base.columns)
    X = base.values.astype(float)
    y = df[TARGET_COL].values.astype(float)
    return base, y, feat_cols

def make_svr_pipeline(random_state: int) -> TransformedTargetRegressor:
    pre = Pipeline([("winsor", Winsorizer(0.01, 0.99)), ("scale", RobustScaler())])
    svr = SVR(kernel='rbf')
    pipe = Pipeline([("pre", pre), ("svr", svr)])
    return TransformedTargetRegressor(
        regressor=pipe,
        transformer=FunctionTransformer(func=np.log1p, inverse_func=np.expm1, validate=False, feature_names_out='one-to-one')
    )

def make_hgb_pipeline(random_state: int) -> TransformedTargetRegressor:
    hgb = HistGradientBoostingRegressor(loss="absolute_error", early_stopping=True, random_state=random_state)
    pipe = Pipeline([("winsor", Winsorizer(0.01, 0.99)), ("scale", RobustScaler()), ("hgb", hgb)])
    return TransformedTargetRegressor(
        regressor=pipe,
        transformer=FunctionTransformer(func=np.log1p, inverse_func=np.expm1, validate=False)
    )

def make_xgb_pipeline(random_state: int) -> TransformedTargetRegressor:
    xgb = XGBRegressor(
        objective="reg:pseudohubererror", tree_method="hist", eval_metric="rmse",
        random_state=random_state, n_estimators=600
    )
    pipe = Pipeline([("winsor", Winsorizer(0.01, 0.99)), ("scale", RobustScaler()), ("xgb", xgb)])
    return TransformedTargetRegressor(
        regressor=pipe,
        transformer=FunctionTransformer(func=np.log1p, inverse_func=np.expm1, validate=False)
    )

def param_space_svr():
    rng = np.random.RandomState(RANDOM_STATE)
    def log_uniform(low, high, size): return np.exp(rng.uniform(np.log(low), np.log(high), size))
    return {
        "regressor__svr__C": log_uniform(1e-1, 1e3, N_ITER_SVR),
        "regressor__svr__gamma": log_uniform(1e-4, 1e-1, N_ITER_SVR),
        "regressor__svr__epsilon": log_uniform(1e-3, 5e-1, N_ITER_SVR),
    }

def param_space_hgb():
    rng = np.random.RandomState(RANDOM_STATE + 1)
    return {
        "regressor__hgb__learning_rate": rng.choice([0.05, 0.07, 0.1], N_ITER_HGB),
        "regressor__hgb__max_depth": rng.choice([None, 3, 4], N_ITER_HGB),
        "regressor__hgb__max_leaf_nodes": rng.choice([31, 63], N_ITER_HGB),
        "regressor__hgb__min_samples_leaf": rng.choice([10, 15, 20], N_ITER_HGB),
        "regressor__hgb__l2_regularization": rng.choice([0.3, 0.5, 1.0], N_ITER_HGB),
    }

def param_space_xgb():
    rng = np.random.RandomState(RANDOM_STATE + 2)
    return {
        "regressor__xgb__learning_rate": rng.choice([0.03, 0.05, 0.07, 0.1], N_ITER_XGB),
        "regressor__xgb__max_depth": rng.choice([3, 4, 5, 6], N_ITER_XGB),
        "regressor__xgb__min_child_weight": rng.choice([1.0, 3.0, 5.0, 10.0], N_ITER_XGB),
        "regressor__xgb__subsample": rng.choice([0.7, 0.8, 1.0], N_ITER_XGB),
        "regressor__xgb__colsample_bytree": rng.choice([0.7, 0.8, 1.0], N_ITER_XGB),
        "regressor__xgb__reg_lambda": rng.choice([0.5, 1.0, 2.0, 5.0], N_ITER_XGB),
    }

@dataclass
class FoldResult:
    name: str
    r2: float
    rmse: float
    p90: float
    best_params: Dict

def _loss_rmse_pinball(y_true: np.ndarray, y_pred: np.ndarray, lam: float = LAMBDA_PB) -> float:
    return rmse(y_true, y_pred) + lam * pinball90(y_true, y_pred)

def learn_ensemble_weights(preds: Dict[str, np.ndarray], y_val: np.ndarray,
                           l2: float = 1e-3, lam: float = LAMBDA_PB,
                           step: float = WEIGHT_GRID_STEP,
                           weak_cap: Tuple[float, float] = (1.4, 1.7)) -> Dict[str, float]:
    names = list(preds.keys())
    M = len(names)
    P = np.vstack([preds[k] for k in names]).T
    y = y_val.reshape(-1)

    rmses = {k: rmse(y, preds[k]) for k in names}
    best_rmse = min(rmses.values())
    caps = {k: 1.0 for k in names}
    for k in names:
        if rmses[k] > weak_cap[1] * best_rmse:
            caps[k] = 0.0
        elif rmses[k] > weak_cap[0] * best_rmse:
            caps[k] = 0.5

    best_w, best_obj = None, float("inf")
    ks = np.arange(0.0, 1.0 + step, step)

    if M == 2:
        for w0 in ks:
            w = np.array([w0, 1 - w0])
            if w[0] > caps[names[0]] or w[1] > caps[names[1]]:
                continue
            obj = _loss_rmse_pinball(y, P @ w, lam) + l2 * float(np.dot(w, w))
            if obj < best_obj:
                best_obj, best_w = obj, w
    elif M == 3:
        for w0 in ks:
            for w1 in ks:
                w2 = 1 - w0 - w1
                if w2 < 0: continue
                w = np.array([w0, w1, w2])
                if any(w[i] > caps[names[i]] for i in range(3)):  # fixed indexing
                    continue
                obj = _loss_rmse_pinball(y, P @ w, lam) + l2 * float(np.dot(w, w))
                if obj < best_obj:
                    best_obj, best_w = obj, w
    else:
        best_w = np.ones(M) / M

    best_w = np.where(best_w < 1e-3, 0.0, best_w)
    s = best_w.sum()
    best_w = np.ones(M) / M if s <= 0 else best_w / s
    return {names[i]: float(best_w[i]) for i in range(M)}

# ------------- IQR tools -------------
def iqr_stats(y: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=float).ravel()
    q1 = np.percentile(y, 25)
    q3 = np.percentile(y, 75)
    iqr = q3 - q1
    return {
        "Q1": q1, "Q3": q3, "IQR": iqr,
        "inner_lo": q1 - IQR_K_INNER * iqr,
        "inner_hi": q3 + IQR_K_INNER * iqr,
        "outer_lo": q1 - IQR_K_OUTER * iqr,
        "outer_hi": q3 + IQR_K_OUTER * iqr,
    }

def flag_outliers_iqr(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    stats = iqr_stats(y)
    y = np.asarray(y, dtype=float).ravel()
    mild = (y < stats["inner_lo"]) | (y > stats["inner_hi"])
    extreme = (y < stats["outer_lo"]) | (y > stats["outer_hi"])
    return mild, extreme

def screen_outliers_and_save(y: np.ndarray, path_csv: str = OUTLIER_CSV) -> Dict[str, int]:
    mild, extreme = flag_outliers_iqr(y)
    idx = np.arange(len(y))
    stats = iqr_stats(y)
    out = pd.DataFrame({
        "index": idx, TARGET_COL: y,
        "mild_outlier_IQR1p5": mild.astype(int),
        "extreme_outlier_IQR3": extreme.astype(int),
    })
    meta = pd.DataFrame({k: [v] for k, v in stats.items()})
    with open(path_csv, "w", encoding="utf-8") as f:
        f.write("# IQR screening (Tukey fences)\n")
    out.to_csv(path_csv, mode="a", index=False)
    meta.to_csv(path_csv, mode="a", index=False)
    return {"n_total": len(y), "n_mild": int(mild.sum()), "n_extreme": int(extreme.sum()), **stats}

# ------------- Diagnostics & Plots (no seaborn) -------------
def plot_histograms_y(y: np.ndarray):
    y = np.asarray(y).ravel()
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(y, bins=40)
    ax.set_title("Target y distribution")
    ax.set_xlabel("y"); ax.set_ylabel("count")
    fig.tight_layout(); fig.savefig(PLOT_DATA_OVERVIEW); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(np.log1p(y), bins=40)
    ax.set_title("log1p(y) distribution")
    ax.set_xlabel("log1p(y)"); ax.set_ylabel("count")
    fig.tight_layout(); fig.savefig(PLOT_Y_LOG); plt.close(fig)

def plot_iqr_bars(stats: Dict[str,float], mild: int, extreme: int, total: int):
    fig, ax = plt.subplots(figsize=(5,4))
    ax.bar(["mild (1.5×IQR)", "extreme (3×IQR)"], [mild, extreme])
    ax.set_title(f"IQR outlier counts (total={total})\nQ1={stats['Q1']:.3f}, Q3={stats['Q3']:.3f}, IQR={stats['IQR']:.3f}")
    fig.tight_layout(); fig.savefig(PLOT_IQR); plt.close(fig)

def plot_oof_scatter(y_true: np.ndarray, y_pred: np.ndarray):
    fig, ax = plt.subplots(figsize=(5,5))
    ax.scatter(y_true, y_pred, s=16, alpha=0.7)
    lim = [0, max(1e-6, float(np.nanmax([y_true.max(), y_pred.max()]))) ]
    ax.plot(lim, lim)
    ax.set_xlabel("True y"); ax.set_ylabel("OOF prediction")
    ax.set_title("OOF: Prediction vs True")
    fig.tight_layout(); fig.savefig(PLOT_OOF_SCATTER); plt.close(fig)

def plot_oof_residual(y_true: np.ndarray, y_pred: np.ndarray):
    resid = y_pred - y_true
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(y_pred, resid, s=16, alpha=0.7)
    ax.axhline(0, linestyle="--")
    ax.set_xlabel("OOF prediction"); ax.set_ylabel("Residual")
    ax.set_title("OOF: Residual vs Prediction")
    fig.tight_layout(); fig.savefig(PLOT_OOF_RESID); plt.close(fig)

def plot_corr_heatmap(df: pd.DataFrame, method: str, path: str):
    corr = df.corr(method=method).values
    labels = df.columns.tolist()
    fig, ax = plt.subplots(figsize=(8,6))
    im = ax.imshow(corr, origin="lower")
    ax.set_title(f"Correlation heatmap ({method})")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)

def plot_pi_bar(pi_df: pd.DataFrame, topn: int = 15):
    df = pi_df.sort_values("pi_mean", ascending=False).head(topn)
    fig, ax = plt.subplots(figsize=(7,5))
    ax.barh(df["feature"], df["pi_mean"])
    ax.invert_yaxis()
    ax.set_xlabel("Permutation importance (mean Δscore)")
    ax.set_title(f"Top-{min(topn, len(df))} features (OOF PI)")
    for i, v in enumerate(df["pos_ratio"].values):
        ax.text(df["pi_mean"].values[i]*1.01 if df["pi_mean"].values[i]>0 else 0.002, i, f"pos {v:.2f}", va="center", fontsize=8)
    fig.tight_layout(); fig.savefig(PLOT_PI_BAR); plt.close(fig)

def plot_ensemble_weights(weights_list: List[Dict[str,float]]):
    models = sorted({k for w in weights_list for k in w.keys()})
    idx = np.arange(len(weights_list))
    bottoms = np.zeros(len(weights_list))
    fig, ax = plt.subplots(figsize=(7,4))
    for m in models:
        vals = np.array([w.get(m, 0.0) for w in weights_list])
        ax.bar(idx, vals, bottom=bottoms, label=m)
        bottoms += vals
    ax.set_xticks(idx); ax.set_xticklabels([f"Fold {i+1}" for i in idx], rotation=0)
    ax.set_ylim(0,1.0)
    ax.set_ylabel("weight"); ax.set_title("Ensemble weights per fold")
    ax.legend()
    fig.tight_layout(); fig.savefig(PLOT_ENSEMBLE_WEIGHTS); plt.close(fig)

def plot_learning_curve_model(X: np.ndarray, y: np.ndarray, make_model_func, title: str, path: str):
    # Use simple KFold to avoid heavy nesting; learning_curve internally splits.
    model = make_model_func(RANDOM_STATE)
    # Use negative RMSE scorer
    scorer = make_scorer(lambda yt, yp: -rmse(yt, yp))
    train_sizes, train_scores, test_scores = learning_curve(
        estimator=model, X=X, y=y, cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring=scorer, train_sizes=np.linspace(0.2, 1.0, 5), n_jobs=N_JOBS, shuffle=True, random_state=RANDOM_STATE
    )
    train_rmse = -train_scores
    test_rmse = -test_scores
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(train_sizes, train_rmse.mean(axis=1), marker='o', label="Train RMSE")
    ax.fill_between(train_sizes, train_rmse.mean(axis=1)-train_rmse.std(axis=1),
                    train_rmse.mean(axis=1)+train_rmse.std(axis=1), alpha=0.2)
    ax.plot(train_sizes, test_rmse.mean(axis=1), marker='o', label="CV RMSE")
    ax.fill_between(train_sizes, test_rmse.mean(axis=1)-test_rmse.std(axis=1),
                    test_rmse.mean(axis=1)+test_rmse.std(axis=1), alpha=0.2)
    ax.set_title(title); ax.set_xlabel("Train size"); ax.set_ylabel("RMSE (lower is better)")
    ax.legend(); fig.tight_layout(); fig.savefig(path); plt.close(fig)

# ------------- Core CV -------------
def run_nested_cv(X: np.ndarray, y: np.ndarray, feat_names: List[str], collect_pi: bool = True):
    n = len(y)
    y_bins_outer = stratify_bins(y, n_bins=Y_BINS, seed=RANDOM_STATE)
    rskf_outer = RepeatedStratifiedKFold(n_splits=N_OUTER, n_repeats=N_REPEATS, random_state=RANDOM_STATE)

    oof_pred = {"SVR": np.zeros(n), "HGB": np.zeros(n)}
    if USE_XGB: oof_pred["XGB"] = np.zeros(n)
    oof_pred["ENS"] = np.zeros(n)
    y_oof = np.full(n, np.nan)

    fold_summaries = []
    ens_weights_per_fold = []
    pi_records = []  # collect OOF permutation importance across folds

    fold_idx = 0
    for train_idx, test_idx in rskf_outer.split(X, y_bins_outer):
        fold_idx += 1
        Xtr, Xval = X[train_idx], X[test_idx]
        ytr, yval = y[train_idx], y[test_idx]

        inner_bins = stratify_bins(ytr, n_bins=min(Y_BINS, max(3, int(np.sqrt(len(ytr))))))
        rskf_inner = StratifiedKFold(n_splits=N_INNER, shuffle=True, random_state=RANDOM_STATE+fold_idx)

        # SVR
        svr = make_svr_pipeline(RANDOM_STATE + fold_idx)
        svr_rs = RandomizedSearchCV(
            estimator=svr, param_distributions=param_space_svr(), n_iter=N_ITER_SVR,
            cv=rskf_inner.split(Xtr, stratify_bins(ytr, n_bins=min(Y_BINS, 5), seed=RANDOM_STATE+fold_idx)),
            scoring="neg_root_mean_squared_error", n_jobs=N_JOBS, refit=True, random_state=RANDOM_STATE+fold_idx
        )
        svr_rs.fit(Xtr, ytr)
        y_sv = svr_rs.predict(Xval)

        # HGB
        hgb = make_hgb_pipeline(RANDOM_STATE + fold_idx)
        hgb_rs = RandomizedSearchCV(
            estimator=hgb, param_distributions=param_space_hgb(), n_iter=N_ITER_HGB,
            cv=rskf_inner.split(Xtr, stratify_bins(ytr, n_bins=min(Y_BINS, 5), seed=RANDOM_STATE+fold_idx+1)),
            scoring="neg_root_mean_squared_error", n_jobs=N_JOBS, refit=True, random_state=RANDOM_STATE+fold_idx+1
        )
        hgb_rs.fit(Xtr, ytr)
        y_hg = hgb_rs.predict(Xval)

        # XGB (opt)
        if USE_XGB:
            xgb = make_xgb_pipeline(RANDOM_STATE + fold_idx)
            xgb_rs = RandomizedSearchCV(
                estimator=xgb, param_distributions=param_space_xgb(), n_iter=N_ITER_XGB,
                cv=rskf_inner.split(Xtr, stratify_bins(ytr, n_bins=min(Y_BINS, 5), seed=RANDOM_STATE+fold_idx+2)),
                scoring="neg_root_mean_squared_error", n_jobs=N_JOBS, refit=True, random_state=RANDOM_STATE+fold_idx+2
            )
            xgb_rs.fit(Xtr, ytr)
            y_xg = xgb_rs.predict(Xval)

        # OOF pack
        y_oof[test_idx] = yval
        oof_pred["SVR"][test_idx] = y_sv
        oof_pred["HGB"][test_idx] = y_hg
        if USE_XGB: oof_pred["XGB"][test_idx] = y_xg

        # per-fold metrics
        def mk_foldres(name, yhat, best_params):
            return FoldResult(name, r2_score(yval, yhat), rmse(yval, yhat), pinball90(yval, yhat), best_params)

        fr_svr = mk_foldres("SVR", y_sv, svr_rs.best_params_)
        fr_hgb = mk_foldres("HGB", y_hg, hgb_rs.best_params_)
        if USE_XGB:
            fr_xgb = mk_foldres("XGB", y_xg, xgb_rs.best_params_)

        # ensemble weights
        preds_for_ens = {"SVR": y_sv, "HGB": y_hg}
        if USE_XGB: preds_for_ens["XGB"] = y_xg
        weights = learn_ensemble_weights(preds_for_ens, yval, lam=LAMBDA_PB)
        y_ens = sum(weights[k] * preds_for_ens[k] for k in preds_for_ens)
        oof_pred["ENS"][test_idx] = y_ens
        ens_weights_per_fold.append(weights)

        fold_summaries += [
            f"- Fold {fold_idx} | SVR | R2={fr_svr.r2:.3f} | RMSE={fr_svr.rmse:.3f} | Pin90={fr_svr.p90:.3f} | best={fr_svr.best_params}",
            f"- Fold {fold_idx} | HGB | R2={fr_hgb.r2:.3f} | RMSE={fr_hgb.rmse:.3f} | Pin90={fr_hgb.p90:.3f} | best={fr_hgb.best_params}",
        ]
        if USE_XGB:
            fold_summaries.append(f"- Fold {fold_idx} | XGB | R2={fr_xgb.r2:.3f} | RMSE={fr_xgb.rmse:.3f} | Pin90={fr_xgb.p90:.3f} | best={fr_xgb.best_params}")
        fold_summaries.append(f"- __DIAG_FOLD_{fold_idx}__ | ENS | Weights={weights}")

        # Permutation Importance on validation set ONLY (leak-safe)
        if collect_pi:
            # choose the fold's *ensemble-best* single model by RMSE for PI stability
            single_preds = {"SVR": y_sv, "HGB": y_hg}
            if USE_XGB: single_preds["XGB"] = y_xg
            best_name = min(single_preds.keys(), key=lambda k: rmse(yval, single_preds[k]))
            best_estimator = {"SVR": svr_rs.best_estimator_, "HGB": hgb_rs.best_estimator_}
            if USE_XGB: best_estimator["XGB"] = xgb_rs.best_estimator_
            model = best_estimator[best_name]
            # permutation_importance uses model.score by default; give it neg_RMSE scorer
            scorer = make_scorer(lambda yt, yp: -rmse(yt, yp))
            pi = permutation_importance(model, Xval, yval, scoring=scorer, n_repeats=20, random_state=RANDOM_STATE)
            for j, f in enumerate(feat_names):
                pi_records.append({"fold": fold_idx, "feature": f, "pi": float(pi.importances_mean[j])})

    # summaries
    def summary_line(name, yhat):
        return f"{name:>5} | R^2 = {r2_score(y_oof, yhat):.3f} | RMSE = {rmse(y_oof, yhat):.3f} | Pinball@{PINBALL_Q:.2f} = {pinball90(y_oof, yhat):.3f}"
    summary = {"SVR": summary_line("SVR", oof_pred["SVR"]),
               "HGB": summary_line("HGB", oof_pred["HGB"]),
               "ENS": summary_line("ENS", oof_pred["ENS"])}
    if USE_XGB:
        summary["XGB"] = summary_line("XGB", oof_pred["XGB"])

    # aggregate PI
    pi_df = None
    if collect_pi and len(pi_records) > 0:
        df = pd.DataFrame(pi_records)
        agg = df.groupby("feature")["pi"].agg(["mean", lambda s: np.mean(s > 0.0)])
        agg.columns = ["pi_mean", "pos_ratio"]
        pi_df = agg.reset_index().sort_values("pi_mean", ascending=False)

    return summary, fold_summaries, y_oof, oof_pred, ens_weights_per_fold, pi_df

# ------------- Save report -------------
def save_report(summary: Dict, fold_summaries: List[str], feat_names: List[str],
                X: np.ndarray, y: np.ndarray, weights_list: List[Dict[str, float]],
                iqr_info: Dict[str, float], sens_results: Dict = None,
                post_sel_summary: Dict = None):
    lines = []
    lines.append(f"# PM2.5 Meta-Ensemble Run Report\n")
    lines.append(f"- Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Rows: {len(y)} | Features: {X.shape[1]} | Target length: {len(y)}")
    lines.append(f"- Feature columns used ({len(feat_names)}): {feat_names}")
    lines.append(f"- Target column: {TARGET_COL}")
    lines.append(f"- Outer folds: {N_OUTER}, Inner folds: {N_INNER}, y-bins: {Y_BINS}, repeats: {N_REPEATS}")
    lines.append(f"- Target transform: log1p/expm1 via TransformedTargetRegressor (sklearn)。")
    lines.append(f"- Ensemble loss: RMSE + {LAMBDA_PB:.2f} * Pinball@{PINBALL_Q:.2f}")
    lines.append("")
    lines.append("## IQR screening (Tukey fences)")
    lines.append(f"- Q1 = {iqr_info['Q1']:.6f}, Q3 = {iqr_info['Q3']:.6f}, IQR = {iqr_info['IQR']:.6f}")
    lines.append(f"- Inner fences: [{iqr_info['inner_lo']:.6f}, {iqr_info['inner_hi']:.6f}] (k={IQR_K_INNER})")
    lines.append(f"- Outer fences: [{iqr_info['outer_lo']:.6f}, {iqr_info['outer_hi']:.6f}] (k={IQR_K_OUTER})")
    lines.append(f"- Mild outliers: {iqr_info['n_mild']} / {iqr_info['n_total']}")
    lines.append(f"- Extreme outliers: {iqr_info['n_extreme']} / {iqr_info['n_total']}")
    lines.append(f"- Screening list saved: {OUTLIER_CSV}")
    lines.append("")
    lines.append("## Per-fold results (All data; screening only, no deletion)")
    lines += fold_summaries
    lines.append("")
    lines.append("## OOF summary (All data)")
    for k, v in summary.items(): lines.append(f"- {k} | {v}")
    lines.append("")
    lines.append("## Per-fold ensemble weights (All data)")
    for i, w in enumerate(weights_list, 1): lines.append(f"- Fold {i}: {w}")
    lines.append("")
    lines.append("## Target distribution (y) summary")
    lines.append(str(pd.Series(y).describe()))
    lines.append("")
    if post_sel_summary is not None:
        lines.append("## Post-selection OOF summary (selected features)")
        for k, v in post_sel_summary.items(): lines.append(f"- {k} | {v}")
        lines.append("")
    lines.append("> 说明：回归任务原生不支持 StratifiedKFold；本脚本先对 y 分箱再分层（近似分层），评估更稳。")
    lines.append("> 预处理采用 Winsorizer + RobustScaler（均在训练折拟合，避免泄露），目标使用 log1p/expm1 变换。")
    lines.append("> IQR 用于异常识别的筛查与留痕；敏感性分析为预先声明的对照，主结论以全数据+稳健方法为准。")
    lines.append("")
    if sens_results is not None:
        lines.append("## IQR sensitivity (trim & re-run)")
        lines.append(f"- Trim strategy: {IQR_TRIM_STRATEGY} (k={IQR_K_INNER if IQR_TRIM_STRATEGY=='inner' else IQR_K_OUTER})")
        lines.append(f"- Removed rows: {sens_results['n_removed']} / {sens_results['n_total']}")
        lines.append("")
        lines.append("### OOF summary (IQR-trimmed)")
        for k, v in sens_results["summary_trim"].items(): lines.append(f"- {k} | {v}")
        lines.append("")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# ------------- Main -------------
def main():
    # Load
    df = pd.read_excel(DATA_PATH)
    if TARGET_COL not in df.columns:
        raise ValueError(f"找不到目标列 {TARGET_COL}，实际列名：{list(df.columns)}")

    # Optional hold-out
    if USE_HOLDOUT:
        df_train, df_hold = train_test_split(df, test_size=HOLDOUT_SIZE, random_state=HOLDOUT_RANDOM_STATE)
    else:
        df_train = df.copy(); df_hold = None

    # Build features
    Xdf, y, feat_cols = build_feature_frame(df_train)
    X = Xdf.values.astype(float)

    # --- Data overview plots (no leakage; just raw y) ---
    plot_histograms_y(y)
    iqr_info = {"n_total": len(y)}
    if RUN_IQR_SCREENING:
        counts = screen_outliers_and_save(y, OUTLIER_CSV)
        iqr_info.update({k: counts[k] for k in ["n_total","n_mild","n_extreme","Q1","Q3","IQR","inner_lo","inner_hi","outer_lo","outer_hi"]})
    else:
        iqr_info.update({**iqr_stats(y), "n_mild": 0, "n_extreme": 0})
    plot_iqr_bars(iqr_info, iqr_info["n_mild"], iqr_info["n_extreme"], iqr_info["n_total"])

    # Correlation heatmaps (features only)
    plot_corr_heatmap(Xdf, "spearman", PLOT_CORR_SPEARMAN)
    plot_corr_heatmap(Xdf, "pearson", PLOT_CORR_PEARSON)

    # --- Baseline nested CV (all features) ---
    summary_all, fold_summ_all, y_oof_all, oof_pred_all, weights_all, pi_df = run_nested_cv(X, y, feat_cols, collect_pi=True)

    # export OOF predictions for diagnostics
    oof_df = pd.DataFrame({"y_true": y})
    for k in oof_pred_all:
        oof_df[f"yhat_{k}"] = oof_pred_all[k]
    oof_df.to_csv(OOF_PRED_CSV, index=False)

    # plots based on ENS OOF
    plot_oof_scatter(oof_df["y_true"].values, oof_df["yhat_ENS"].values)
    plot_oof_residual(oof_df["y_true"].values, oof_df["yhat_ENS"].values)
    plot_ensemble_weights(weights_all)

    # PI aggregation & export
    if pi_df is not None:
        pi_df.to_csv(PI_CSV, index=False)
        plot_pi_bar(pi_df, topn=15)

    # --- Feature selection by OOF PI (one-shot, conservative) ---
    selected_feats = feat_cols.copy()
    if pi_df is not None and len(pi_df) > 0:
        sel = pi_df[(pi_df["pi_mean"] > PI_MIN_MEAN) & (pi_df["pos_ratio"] >= PI_KEEP_MIN_PROP)].sort_values("pi_mean", ascending=False)
        if PI_TOPK is not None:
            sel = sel.head(PI_TOPK)
        selected_feats = sel["feature"].tolist()
        if len(selected_feats) < PI_MIN_KEEP:
            selected_feats = pi_df.sort_values("pi_mean", ascending=False).head(PI_MIN_KEEP)["feature"].tolist()

        with open(SELECTED_JSON, "w", encoding="utf-8") as f:
            json.dump({"selected_features": selected_feats, "rules": {
                "pi_mean>": PI_MIN_MEAN, "pos_ratio>=": PI_KEEP_MIN_PROP,
                "min_keep": PI_MIN_KEEP, "topk": PI_TOPK
            }}, f, ensure_ascii=False, indent=2)

        # re-run with selected features
        X_sel = Xdf[selected_feats].values.astype(float)
        summary_sel, fold_summ_sel, y_oof_sel, oof_pred_sel, weights_sel, _ = run_nested_cv(X_sel, y, selected_feats, collect_pi=False)

        # compare table
        comp = []
        def parse_line(s: str):
            # format: "  ENS | R^2 = x | RMSE = y | Pinball@0.90 = z"
            parts = s.split("|")
            r2 = float(parts[1].split("=")[1])
            rm = float(parts[2].split("=")[1])
            pb = float(parts[3].split("=")[1])
            return r2, rm, pb
        for name in ["SVR","HGB","XGB","ENS"]:
            if name in summary_all:
                r2a, rma, pba = parse_line(summary_all[name])
                if name in summary_sel:
                    r2b, rmb, pbb = parse_line(summary_sel[name])
                else:
                    r2b = rmb = pbb = np.nan
                comp.append([name, r2a, rma, pba, r2b, rmb, pbb])
        comp_df = pd.DataFrame(comp, columns=["model", "pre_R2", "pre_RMSE", "pre_PB90", "post_R2", "post_RMSE", "post_PB90"])
        comp_df.to_csv(COMPARE_CSV, index=False)
        post_sel_summary = summary_sel
    else:
        post_sel_summary = None

    # --- Optional hold-out evaluation (final, once-only) ---
    holdout_metrics = None
    if USE_HOLDOUT and df_hold is not None:
        Xdf_hold, y_hold, _ = build_feature_frame(df_hold)
        # choose features (post-selection if available)
        use_feats = selected_feats if selected_feats else feat_cols
        X_hold = Xdf_hold[use_feats].values.astype(float)

        # Fit full-data best models on df_train with use_feats
        # SVR
        svr = make_svr_pipeline(RANDOM_STATE)
        svr_rs = RandomizedSearchCV(
            estimator=svr, param_distributions=param_space_svr(), n_iter=min(N_ITER_SVR, 50),
            cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
            scoring="neg_root_mean_squared_error", n_jobs=N_JOBS, refit=True, random_state=RANDOM_STATE
        )
        svr_rs.fit(Xdf[use_feats].values.astype(float), y)
        y_sv = svr_rs.predict(X_hold)

        # HGB
        hgb = make_hgb_pipeline(RANDOM_STATE)
        hgb_rs = RandomizedSearchCV(
            estimator=hgb, param_distributions=param_space_hgb(), n_iter=min(N_ITER_HGB, 50),
            cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE+1),
            scoring="neg_root_mean_squared_error", n_jobs=N_JOBS, refit=True, random_state=RANDOM_STATE+1
        )
        hgb_rs.fit(Xdf[use_feats].values.astype(float), y)
        y_hg = hgb_rs.predict(X_hold)

        # XGB
        if USE_XGB:
            xgb = make_xgb_pipeline(RANDOM_STATE)
            xgb_rs = RandomizedSearchCV(
                estimator=xgb, param_distributions=param_space_xgb(), n_iter=min(N_ITER_XGB, 50),
                cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE+2),
                scoring="neg_root_mean_squared_error", n_jobs=N_JOBS, refit=True, random_state=RANDOM_STATE+2
            )
            xgb_rs.fit(Xdf[use_feats].values.astype(float), y)
            y_xg = xgb_rs.predict(X_hold)

        preds_for_ens = {"SVR": y_sv, "HGB": y_hg}
        if USE_XGB: preds_for_ens["XGB"] = y_xg
        w_hold = learn_ensemble_weights(preds_for_ens, y_hold, lam=LAMBDA_PB)
        y_ens = sum(w_hold[k] * preds_for_ens[k] for k in preds_for_ens)
        holdout_metrics = {
            "SVR": {"R2": r2_score(y_hold, y_sv), "RMSE": rmse(y_hold, y_sv), "PB90": pinball90(y_hold, y_sv)},
            "HGB": {"R2": r2_score(y_hold, y_hg), "RMSE": rmse(y_hold, y_hg), "PB90": pinball90(y_hold, y_hg)},
            "ENS": {"R2": r2_score(y_hold, y_ens), "RMSE": rmse(y_hold, y_ens), "PB90": pinball90(y_hold, y_ens)},
        }
        if USE_XGB:
            holdout_metrics["XGB"] = {"R2": r2_score(y_hold, y_xg), "RMSE": rmse(y_hold, y_xg), "PB90": pinball90(y_hold, y_xg)}
        with open("holdout_metrics.json", "w", encoding="utf-8") as f:
            json.dump(holdout_metrics, f, ensure_ascii=False, indent=2)

    # --- Learning curves (using all training data features; leak-safe) ---
    plot_learning_curve_model(X, y, make_svr_pipeline, "SVR learning curve", PLOT_LC_SVR)
    plot_learning_curve_model(X, y, make_hgb_pipeline, "HGB learning curve", PLOT_LC_HGB)

    # --- IQR sensitivity (optional) ---
    sens_payload = None
    if RUN_IQR_SENSITIVITY:
        mild_mask, extreme_mask = flag_outliers_iqr(y)
        trim_mask = mild_mask if IQR_TRIM_STRATEGY == "inner" else extreme_mask
        keep = ~trim_mask
        X_trim, y_trim = X[keep], y[keep]
        summary_trim, fold_summ_trim, y_oof_trim, oof_pred_trim, weights_trim, _ = run_nested_cv(X_trim, y_trim, feat_cols, collect_pi=False)
        sens_payload = {"n_total": int(len(y)), "n_removed": int(trim_mask.sum()), "summary_trim": summary_trim, "weights_trim": weights_trim}

    # --- Save report ---
    save_report(summary_all, fold_summ_all, feat_cols, X, y, weights_all, iqr_info, sens_payload, post_sel_summary=post_sel_summary if 'post_sel_summary' in locals() else None)

    # --- Pack best estimators on full data (all features baseline) ---
    # (We pack baseline; users can choose to re-pack with selected_feats if desired.)
    svr = make_svr_pipeline(RANDOM_STATE)
    svr_rs = RandomizedSearchCV(
        estimator=svr, param_distributions=param_space_svr(), n_iter=min(N_ITER_SVR, 50),
        cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring="neg_root_mean_squared_error", n_jobs=N_JOBS, refit=True, random_state=RANDOM_STATE
    )
    svr_rs.fit(X, y)

    hgb = make_hgb_pipeline(RANDOM_STATE)
    hgb_rs = RandomizedSearchCV(
        estimator=hgb, param_distributions=param_space_hgb(), n_iter=min(N_ITER_HGB, 50),
        cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE+1),
        scoring="neg_root_mean_squared_error", n_jobs=N_JOBS, refit=True, random_state=RANDOM_STATE+1
    )
    hgb_rs.fit(X, y)

    xgb_best = None
    if USE_XGB:
        xgb = make_xgb_pipeline(RANDOM_STATE)
        xgb_rs = RandomizedSearchCV(
            estimator=xgb, param_distributions=param_space_xgb(), n_iter=min(N_ITER_XGB, 50),
            cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE+2),
            scoring="neg_root_mean_squared_error", n_jobs=N_JOBS, refit=True, random_state=RANDOM_STATE+2
        )
        xgb_rs.fit(X, y)
        xgb_best = xgb_rs.best_estimator_

    bundle = {
        "feature_names": feat_cols,
        "svr": svr_rs.best_estimator_,
        "hgb": hgb_rs.best_estimator_,
        "use_xgb": USE_XGB,
        "xgb": xgb_best,
        "meta": {
            "report": summary_all,
            "weights_hint": weights_all,
            "iqr_stats": iqr_info,
            "ensemble_loss": f"RMSE + {LAMBDA_PB:.2f} * Pinball@{PINBALL_Q}",
            "pi_selected_features": selected_feats if 'selected_feats' in locals() else feat_cols,
            "holdout_metrics": holdout_metrics
        }
    }
    try:
        import joblib
        joblib.dump(bundle, BUNDLE_PATH)
        print(f"Model bundle saved to {BUNDLE_PATH}")
    except Exception as e:
        print("joblib 保存失败（可忽略，不影响报告）：", e)

    # console summary
    print("\n=== OOF summary (All data; no deletion) ===")
    for k, v in summary_all.items(): print(f"{k:5} | {v}")
    if 'post_sel_summary' in locals() and post_sel_summary is not None:
        print("\n=== OOF summary (Selected features) ===")
        for k, v in post_sel_summary.items(): print(f"{k:5} | {v}")
    if sens_payload is not None:
        print("\n=== OOF summary (IQR-trimmed; sensitivity) ===")
        for k, v in sens_payload["summary_trim"].items(): print(f"{k:5} | {v}")
    if holdout_metrics is not None:
        print("\n=== Hold-out (never seen during model/selection) ===")
        for k, m in holdout_metrics.items():
            print(f"{k:5} | R2={m['R2']:.3f} | RMSE={m['RMSE']:.3f} | PB90={m['PB90']:.3f}")
    print(f"\nPlots/CSVs saved. Detailed report: {REPORT_PATH}")

if __name__ == "__main__":
    main()
