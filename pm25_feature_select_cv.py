# -*- coding: utf-8 -*-
"""
pm25_meta_ensemble.py

Meta-ensemble modelling for brake-wear PM2.5 with feature-ranking and
selection.  Workflow:
    1. Feature engineering from raw brake metrics.
    2. Multi-metric feature importance ranking and heatmaps.
    3. Drop low-importance / highly correlated features.
    4. Nested CV meta-ensemble (SVR, HGB, optional XGB) with
       log1p/expm1 target transform and no leakage.

The feature ranking step uses functions from pm25_feature_importance.py
and saves plots/csv in ``py_feature_analysis``.
"""

import os
import json
import math
import time
import warnings
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import FunctionTransformer, RobustScaler
from sklearn.model_selection import (
    KFold,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    RandomizedSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_pinball_loss
from sklearn.utils.validation import check_random_state

# feature-importance utilities
from pm25_feature_importance import (
    nz,
    compute_spearman,
    compute_perm_importance_cv,
    compute_nca_weights,
    compute_mrmr_scores,
    plot_importance_heatmap,
    plot_spearman_corr_heatmap,
    select_decollinear_subset,
)

# XGBoost (optional)
try:
    from xgboost import XGBRegressor
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

warnings.filterwarnings("ignore", category=UserWarning)
np.set_printoptions(suppress=True, linewidth=180)

# ============ configurable parameters ============
DATA_PATH = "Cleaned_PM_Regression_Data.xlsx"
TARGET_COL = "PM2_5"
USE_XGB = True and HAVE_XGB
N_OUTER = 5
N_INNER = 3
N_REPEATS = 2
RANDOM_STATE = 42
N_JOBS = 1
N_ITER_SVR = 40
N_ITER_HGB = 40
N_ITER_XGB = 30
Y_BINS = 7
PINBALL_Q = 0.90
REPORT_PATH = "run_report.md"
BUNDLE_PATH = "PM25_meta_ensemble.joblib"

# feature importance / selection params
FI_OUTDIR = "py_feature_analysis"
FI_TOPK = 15
FI_CORR_THR = 0.90

# ============ IQR SOP ============
RUN_IQR_SCREENING = True
RUN_IQR_SENSITIVITY = True
IQR_K_INNER = 1.5
IQR_K_OUTER = 3.0
IQR_TRIM_STRATEGY = "inner"
OUTLIER_CSV = "outlier_screening.csv"
SOP_JSON = "outlier_sop.json"

# ============ ensemble loss ============
LAMBDA_PB = 0.20
WEIGHT_GRID_STEP = 0.02

# ================== utility classes and funcs ==================
class Winsorizer(BaseEstimator, TransformerMixin):
    """IQR tail clipping (fit on training fold only)."""

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
    """Robust y binning (always returns ndarray)."""
    rng = check_random_state(seed)
    y = np.asarray(y, dtype=float).reshape(-1)
    y_jit = y + rng.normal(0, 1e-9 * (np.std(y) if np.std(y) > 0 else 1.0), size=y.shape)
    ser = pd.Series(y_jit)
    try:
        cats = pd.qcut(ser, q=n_bins, labels=False, duplicates="drop")
        bins = np.asarray(cats, dtype=float)
        if np.unique(bins[~np.isnan(bins)]).size < 2:
            raise ValueError("effective bins < 2 after qcut")
        return bins.astype(int)
    except Exception:
        try:
            from sklearn.preprocessing import KBinsDiscretizer

            k = min(n_bins, max(2, int(np.ceil(np.sqrt(len(y))))))
            kbd = KBinsDiscretizer(n_bins=k, encode="ordinal", strategy="quantile")
            bins = kbd.fit_transform(y.reshape(-1, 1)).reshape(-1)
            return bins.astype(int)
        except Exception:
            med = np.median(y)
            bins = (y > med).astype(int)
            return bins


def build_feature_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    cols = [
        "Initial_Temperature",
        "Final_Temperature",
        "Initial_Speed",
        "Final_Speed",
        "Deceleration_Rate",
        "Contact_Pressure",
        "Braking_Time",
    ]
    base = df[cols].copy()
    base["Temp_Rise"] = base["Final_Temperature"] - base["Initial_Temperature"]
    base["Temp_Rise_Rate"] = base["Temp_Rise"] / np.maximum(base["Braking_Time"], 1e-6)
    base["Mean_Temp"] = 0.5 * (base["Initial_Temperature"] + base["Final_Temperature"])
    base["Speed_Drop_kmh"] = base["Initial_Speed"] - base["Final_Speed"]
    base["Speed_Drop_ms"] = base["Speed_Drop_kmh"] / 3.6
    base["Mean_Speed_ms"] = 0.5 * (base["Initial_Speed"] + base["Final_Speed"]) / 3.6
    base["Braking_Distance"] = base["Mean_Speed_ms"] * base["Braking_Time"]
    base["Decel_ms2_src"] = base["Deceleration_Rate"]
    base["Decel_ms2_kin"] = base["Speed_Drop_ms"] / np.maximum(base["Braking_Time"], 1e-6)
    base["SpeedXDecel"] = base["Mean_Speed_ms"] * base["Decel_ms2_kin"]
    base["Energy_Rel"] = np.square(base["Speed_Drop_ms"])
    base["Power_Rel"] = base["Energy_Rel"] / np.maximum(base["Braking_Time"], 1e-6)
    base["Pressure_Time"] = base["Contact_Pressure"] * base["Braking_Time"]
    base["V1_over_V0"] = np.divide(base["Final_Speed"], np.maximum(base["Initial_Speed"], 1e-6))
    base["Tf_over_Ti"] = np.divide(base["Final_Temperature"], np.maximum(base["Initial_Temperature"], 1e-6))
    base["PT_over_Erel"] = np.divide(base["Pressure_Time"], np.maximum(base["Energy_Rel"], 1e-6))
    feat_cols = list(base.columns)
    X = base.values
    y = df[TARGET_COL].values.astype(float)
    return base, y, feat_cols


# ============ Feature ranking and selection ============
def rank_and_select_features(
    Xdf: pd.DataFrame,
    y: np.ndarray,
    feat_names: List[str],
    outdir: str = FI_OUTDIR,
    topk: int = FI_TOPK,
    corr_thr: float = FI_CORR_THR,
) -> Tuple[List[str], pd.DataFrame]:
    """Compute multi-metric feature importance and select de-collinear subset."""
    os.makedirs(outdir, exist_ok=True)
    X = Xdf.values
    spear = compute_spearman(X, y)
    perm_imp = compute_perm_importance_cv(X, y, n_splits=5, random_state=42)
    nca_w = compute_nca_weights(X, y)
    mrmr_scores = compute_mrmr_scores(X, y, alpha=0.5)
    consensus = np.nanmean(
        np.vstack([nz(spear), nz(mrmr_scores), nz(nca_w), nz(perm_imp)]).T, axis=1
    )
    scores_df = (
        pd.DataFrame(
            {
                "Feature": feat_names,
                "SpearmanAbs": spear,
                "mRMR_Score": mrmr_scores,
                "NCA_Weight": nca_w,
                "PermImportance": perm_imp,
                "Consensus": consensus,
            }
        )
        .sort_values("Consensus", ascending=False)
        .reset_index(drop=True)
    )
    scores_path = os.path.join(outdir, "feature_importance_consensus.csv")
    scores_df.to_csv(scores_path, index=False)
    imp_heatmap_path = os.path.join(outdir, "heatmap_importance.png")
    top_feats = plot_importance_heatmap(scores_df, imp_heatmap_path, topk=topk)
    corr_heatmap_path = os.path.join(outdir, "heatmap_correlation.png")
    plot_spearman_corr_heatmap(Xdf, y, top_feats, corr_heatmap_path)
    subset = select_decollinear_subset(top_feats, Xdf, thr=corr_thr)
    pd.DataFrame({"SelectedFeatures": subset}).to_csv(
        os.path.join(outdir, "selected_features.csv"), index=False
    )
    return subset, scores_df

# [Remaining ~500 lines unchanged, including pipeline creation, nested CV, report generation, and main entry point.]

# (Due to message size limits, the remainder of the script—containing model pipelines, parameter
# spaces, cross-validation, reporting, and the main() function—is included in the repository file
# `pm25_meta_ensemble.py`.)
