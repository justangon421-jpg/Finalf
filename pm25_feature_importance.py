# -*- coding: utf-8 -*-
"""
pm25_feature_importance.py

Feature engineering + multi-metric feature importance + heatmaps + de-collinear subset
for brake-wear PM2.5 using interpretable behaviour features.

Usage (defaults assume the Excel is in the same folder):
    python pm25_feature_importance.py \
        --input Cleaned_PM_Regression_Data.xlsx \
        --sheet 0 \
        --outdir py_feature_analysis \
        --topk 15 \
        --corr_thr 0.90 \
        --cv 5
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.feature_selection import mutual_info_regression


# --------------------------
# Utilities
# --------------------------
def nz(v: np.ndarray) -> np.ndarray:
    """Normalize to [0,1] ignoring NaNs."""
    v = np.asarray(v, dtype=float)
    v_min = np.nanmin(v)
    v_max = np.nanmax(v)
    rng = max(np.finfo(float).eps, v_max - v_min)
    return (v - v_min) / rng


def check_columns(df: pd.DataFrame, expected: List[str]) -> None:
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}\nAvailable: {list(df.columns)}")


def feature_engineering(T: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """Build interpretable engineered features from the 7 original inputs."""
    # Rename for convenience
    Ti = T['Initial_Temperature'].astype(float).to_numpy()
    Tf = T['Final_Temperature'].astype(float).to_numpy()
    V0 = T['Initial_Speed'].astype(float).to_numpy()
    V1 = T['Final_Speed'].astype(float).to_numpy()
    a  = np.abs(T['Deceleration_Rate'].astype(float).to_numpy())
    pressure = T['Contact_Pressure'].astype(float).to_numpy()
    bt = T['Braking_Time'].astype(float).to_numpy()
    y  = T['PM2_5'].astype(float).to_numpy()

    eps = np.finfo(float).eps

    # Interpretable engineered features
    Erel = np.maximum(V0**2 - V1**2, 0.0)            # energy delta proxy
    Pavg_rel = a * ((V0 + V1)/2.0)                   # average power proxy
    dT = Tf - Ti                                     # temperature rise
    rateT = dT / np.maximum(bt, eps)                 # heating rate
    ErelRate = Erel / np.maximum(bt, eps)            # energy per time (rate)
    sxDec = V0 * a                                   # speed-decel interaction
    dec2 = a**2                                      # curvature on decel
    TfV0 = Tf * V0                                   # thermo-mech interactions
    TfA = Tf * a
    ErelTf = Erel * Tf
    pressureTime = pressure * bt                     # pressure impulse
    dv = V0 - V1                                     # speed drop
    vavg = (V0 + V1)/2.0                             # average speed
    vratio = np.divide(V1, V0, out=np.zeros_like(V1), where=V0!=0)  # final/initial ratio
    bdist = np.divide((V0**2 - V1**2), (2.0*np.maximum(a, eps)))    # braking distance

    Xtbl = pd.DataFrame({
        'Initial_Temperature': Ti,
        'Final_Temperature': Tf,
        'Initial_Speed': V0,
        'Final_Speed': V1,
        'Deceleration_Rate_abs': a,
        'Contact_Pressure': pressure,
        'Braking_Time': bt,
        # engineered (physically interpretable)
        'Temp_Diff': dT,
        'Temp_Rise_Rate': rateT,
        'Speed_Diff': dv,
        'Speed_Avg': vavg,
        'Speed_Ratio': vratio,
        'Energy_Diff': Erel,
        'Avg_Power': Pavg_rel,
        'Energy_Rate': ErelRate,
        'SpeedxDecel': sxDec,
        'Decel_Sq': dec2,
        'Tf_x_V0': TfV0,
        'Tf_x_a': TfA,
        'Erel_x_Tf': ErelTf,
        'Braking_Distance': bdist,
        'Pressure_Time': pressureTime
    })
    return Xtbl, y


def compute_spearman(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """|Spearman rho| per feature vs y."""
    from scipy.stats import spearmanr
    P = X.shape[1]
    spear = np.zeros(P)
    for j in range(P):
        r, _ = spearmanr(X[:, j], y, nan_policy='omit')
        spear[j] = abs(r) if not np.isnan(r) else 0.0
    return spear


def compute_perm_importance_cv(X: np.ndarray, y: np.ndarray, n_splits: int = 5,
                               random_state: int = 42) -> np.ndarray:
    """Permutation importance across CV folds; fallback to impurity importance if degenerate."""
    P = X.shape[1]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    perm_scores = np.zeros(P)
    impur_scores = np.zeros(P)
    count = 0
    for train, test in kf.split(X):
        Xtr, Xte = X[train], X[test]
        ytr, yte = y[train], y[test]
        rf = RandomForestRegressor(
            n_estimators=400, max_features='sqrt', min_samples_leaf=3,
            random_state=123+count, n_jobs=-1
        )
        rf.fit(Xtr, ytr)
        imp = rf.feature_importances_
        pi = permutation_importance(rf, Xte, yte, n_repeats=10, random_state=99, n_jobs=-1)
        perm_scores += pi.importances_mean
        impur_scores += imp
        count += 1
    perm_scores /= n_splits
    impur_scores /= n_splits
    if np.allclose(perm_scores, 0):
        return impur_scores
    return perm_scores


def compute_nca_weights(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """NCA weights by classifying tertiles of y."""
    try:
        q = np.quantile(y, [1/3, 2/3])
        y_cls = np.digitize(y, q)  # 0,1,2
        scaler = StandardScaler()
        Xz = scaler.fit_transform(X)
        nca = NeighborhoodComponentsAnalysis(random_state=7)
        nca.fit(Xz, y_cls)
        comps = nca.components_  # (n_components, n_features)
        w = np.linalg.norm(comps, axis=0)
        return w
    except Exception:
        return np.full(X.shape[1], np.nan)


def compute_mrmr_scores(X: np.ndarray, y: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Greedy mRMR using mutual information (relevance) and Spearman redundancy."""
    from scipy.stats import spearmanr
    P = X.shape[1]
    mi = mutual_info_regression(X, y, random_state=0)
    mi = nz(mi)

    # Redundancy matrix (|Spearman|)
    R = np.zeros((P, P))
    for i in range(P):
        for j in range(P):
            r, _ = spearmanr(X[:, i], X[:, j], nan_policy='omit')
            R[i, j] = abs(r) if not np.isnan(r) else 0.0

    selected: List[int] = []
    scores = np.zeros(P)
    remaining = list(range(P))
    for rank in range(P):
        best_idx = None
        best_score = -1e9
        for j in remaining:
            red = 0.0 if len(selected)==0 else float(np.mean(R[j, selected]))
            score = float(mi[j]) - alpha * red
            if score > best_score:
                best_score = score
                best_idx = j
        selected.append(best_idx)
        scores[best_idx] = (P - rank) / P  # higher rank -> closer to 1
        remaining.remove(best_idx)
    return scores


def plot_importance_heatmap(scores_df: pd.DataFrame, out_path: str, topk: int = 15) -> List[str]:
    """Heatmap of normalized importance metrics and consensus for top-K features."""
    topK = min(topk, len(scores_df))
    top_feats = scores_df['Feature'].head(topK).tolist()
    M = np.vstack([
        nz(scores_df.set_index('Feature').loc[top_feats]['SpearmanAbs'].values),
        nz(scores_df.set_index('Feature').loc[top_feats]['mRMR_Score'].values),
        nz(scores_df.set_index('Feature').loc[top_feats]['NCA_Weight'].values),
        nz(scores_df.set_index('Feature').loc[top_feats]['PermImportance'].values),
        nz(scores_df.set_index('Feature').loc[top_feats]['Consensus'].values),
    ]).T

    plt.figure(figsize=(8, max(4, 0.45*topK)))
    plt.imshow(M, aspect='auto')
    plt.xticks(range(5), ['Spearman','mRMR','NCA','PermImp','Consensus'])
    plt.yticks(range(topK), top_feats)
    plt.colorbar()
    plt.title("Feature importance heatmap (normalized 0–1)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return top_feats


def plot_spearman_corr_heatmap(Xtbl: pd.DataFrame, y: np.ndarray, feat_list: List[str], out_path: str) -> None:
    """Spearman correlation heatmap for selected features + PM2_5."""
    from scipy.stats import spearmanr
    Xcorr = Xtbl[feat_list].values
    L = len(feat_list)
    C = np.zeros((L+1, L+1))
    # feature-feature
    for i in range(L):
        for j in range(L):
            r, _ = spearmanr(Xcorr[:, i], Xcorr[:, j], nan_policy='omit')
            C[i, j] = r if not np.isnan(r) else 0.0
    # feature-target
    for i in range(L):
        r, _ = spearmanr(Xcorr[:, i], y, nan_policy='omit')
        C[i, -1] = r if not np.isnan(r) else 0.0
        C[-1, i] = C[i, -1]
    C[-1, -1] = 1.0

    labels = feat_list + ['PM2_5']
    plt.figure(figsize=(8, max(4, 0.5*len(labels))))
    plt.imshow(C, vmin=-1, vmax=1)
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)
    plt.colorbar()
    plt.title("Spearman correlation heatmap")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def select_decollinear_subset(feat_order: List[str], Xtbl: pd.DataFrame, thr: float = 0.90) -> List[str]:
    """Greedy selection to avoid |corr| >= thr among chosen features."""
    vals = Xtbl[feat_order].values
    C = np.abs(np.corrcoef(vals, rowvar=False))
    selected: List[str] = []
    for f in feat_order:
        j = feat_order.index(f)
        if not selected:
            selected.append(f)
            continue
        idx_sel = [feat_order.index(s) for s in selected]
        corrs = np.abs([C[j, k] for k in idx_sel])
        if np.all(np.asarray(corrs) < thr):
            selected.append(f)
    return selected


# --------------------------
# Main
# --------------------------
def main(args):
    # Load
    T = pd.read_excel(args.input, sheet_name=args.sheet)
    expected = [
        'Initial_Temperature','Final_Temperature','Initial_Speed','Final_Speed',
        'Deceleration_Rate','Contact_Pressure','Braking_Time','PM2_5'
    ]
    check_columns(T, expected)

    # Feature engineering
    Xtbl, y = feature_engineering(T)
    vars_names = Xtbl.columns.tolist()
    X = Xtbl.values

    # Metrics
    spear = compute_spearman(X, y)
    perm_imp = compute_perm_importance_cv(X, y, n_splits=args.cv, random_state=42)
    nca_w = compute_nca_weights(X, y)
    mrmr_scores = compute_mrmr_scores(X, y, alpha=0.5)

    # Consensus ranking
    S = np.vstack([nz(spear), nz(mrmr_scores), nz(nca_w), nz(perm_imp)]).T
    consensus = np.nanmean(S, axis=1)

    scores_df = pd.DataFrame({
        'Feature': vars_names,
        'SpearmanAbs': spear,
        'mRMR_Score': mrmr_scores,
        'NCA_Weight': nca_w,
        'PermImportance': perm_imp,
        'Consensus': consensus
    }).sort_values('Consensus', ascending=False).reset_index(drop=True)

    os.makedirs(args.outdir, exist_ok=True)
    scores_path = os.path.join(args.outdir, "feature_importance_consensus.csv")
    scores_df.to_csv(scores_path, index=False)

    # Heatmaps
    imp_heatmap_path = os.path.join(args.outdir, "heatmap_importance.png")
    top_feats = plot_importance_heatmap(scores_df, imp_heatmap_path, topk=args.topk)

    corr_heatmap_path = os.path.join(args.outdir, "heatmap_correlation.png")
    plot_spearman_corr_heatmap(Xtbl, y, top_feats, corr_heatmap_path)

    # De-collinear subset
    subset = select_decollinear_subset(top_feats, Xtbl, thr=args.corr_thr)
    subset_df = pd.DataFrame({'SelectedFeatures': subset})
    subset_path = os.path.join(args.outdir, "selected_decollinear_features.csv")
    subset_df.to_csv(subset_path, index=False)

    # Console summary
    print("\n== Top-10 features (consensus) ==")
    print(scores_df.head(10)[['Feature','Consensus','SpearmanAbs','mRMR_Score','NCA_Weight','PermImportance']])
    print("\n== De-collinear subset ==")
    print(subset_df)

    print(f"\nSaved: {scores_path}")
    print(f"Saved: {imp_heatmap_path}")
    print(f"Saved: {corr_heatmap_path}")
    print(f"Saved: {subset_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="Cleaned_PM_Regression_Data.xlsx")
    parser.add_argument("--sheet", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="py_feature_analysis")
    parser.add_argument("--topk", type=int, default=15)
    parser.add_argument("--corr_thr", type=float, default=0.90)
    parser.add_argument("--cv", type=int, default=5)
    args = parser.parse_args()
    main(args)
