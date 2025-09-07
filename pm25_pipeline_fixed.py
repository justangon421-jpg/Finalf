#!/usr/bin/env python3
"""
pm25_complete_enhanced.py

Complete PM2.5 brake emissions prediction system with:
- Full feature engineering (23 features)
- OOF permutation importance feature selection
- Multiple models (SVR, HGB, XGBoost, CatBoost)
- Optimized weighted ensemble
- Outlier analysis
- Comprehensive reporting
"""

import os
import json
import time
import warnings
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import FunctionTransformer, RobustScaler
from sklearn.model_selection import (
    KFold, StratifiedKFold, GroupKFold,
    RandomizedSearchCV, cross_val_score
)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    mean_pinball_loss, median_absolute_error
)
from sklearn.inspection import permutation_importance
from sklearn.utils import check_random_state

# Optional imports
try:
    from xgboost import XGBRegressor

    HAVE_XGB = True
except ImportError:
    HAVE_XGB = False
    print("XGBoost not available - continuing without it")

try:
    from catboost import CatBoostRegressor

    HAVE_CAT = True
except ImportError:
    HAVE_CAT = False
    print("CatBoost not available - continuing without it")

warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True, precision=4)

# Try to set plotting style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('default')


# ============ Configuration ============
@dataclass
class Config:
    """Centralized configuration"""
    # Data
    data_path: str = "Cleaned_PM_Regression_Data.xlsx"
    output_dir: str = "output"
    target_col: str = "PM2_5"
    group_col: Optional[str] = None  # For GroupKFold if needed

    # Models
    use_xgb: bool = HAVE_XGB
    use_cat: bool = HAVE_CAT

    # Cross-validation
    n_outer_folds: int = 5
    n_inner_folds: int = 3
    random_state: int = 42
    n_jobs: int = -1

    # Hyperparameter search
    n_iter_svr: int = 30
    n_iter_hgb: int = 30
    n_iter_xgb: int = 25
    n_iter_cat: int = 20

    # Feature selection
    run_feature_selection: bool = True
    n_repeats_perm: int = 10
    min_features: int = 10
    importance_threshold: float = 0.0

    # Ensemble
    optimize_weights: bool = True
    ensemble_method: str = "optimize"  # "average" or "optimize"

    # Outlier analysis
    analyze_outliers: bool = True
    outlier_std_threshold: float = 3.0

    # Monotonic constraints (for XGB/CatBoost)
    enable_mono_constraints: bool = True
    mono_increasing_features: List[str] = field(default_factory=lambda: [
        "Energy_Rel", "Power_Rel", "Decel_ms2_kin",
        "Mean_Speed_ms", "Final_Temperature", "Mean_Temp",
        "Contact_Pressure", "Pressure_Time", "Temp_Rise"
    ])

    def __post_init__(self):
        Path(self.output_dir).mkdir(exist_ok=True)


# ============ Custom Transformers ============
class Winsorizer(BaseEstimator, TransformerMixin):
    """Robust outlier capping via winsorization"""

    def __init__(self, lower_q: float = 0.01, upper_q: float = 0.99):
        self.lower_q = lower_q
        self.upper_q = upper_q
        self.bounds_ = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.bounds_ = np.column_stack([
            np.percentile(X, self.lower_q * 100, axis=0),
            np.percentile(X, self.upper_q * 100, axis=0)
        ])
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        X_clipped = np.copy(X)
        for i in range(X.shape[1]):
            X_clipped[:, i] = np.clip(X[:, i], self.bounds_[i, 0], self.bounds_[i, 1])
        return X_clipped


# ============ Metrics ============
def rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error"""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def smape(y_true, y_pred, eps: float = 1e-8) -> float:
    """Symmetric Mean Absolute Percentage Error"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = np.maximum(eps, (np.abs(y_true) + np.abs(y_pred)) / 2.0)
    return float(np.mean(np.abs(y_pred - y_true) / denominator))


def q_abs_error(y_true, y_pred, q: float = 0.90) -> float:
    """Quantile absolute error"""
    return float(np.quantile(np.abs(y_pred - y_true), q))


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    """Compute comprehensive metrics"""
    return {
        'r2': r2_score(y_true, y_pred),
        'rmse': rmse(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'medae': median_absolute_error(y_true, y_pred),
        'smape': smape(y_true, y_pred),
        'q90ae': q_abs_error(y_true, y_pred, 0.90),
        'pinball90': mean_pinball_loss(y_true, y_pred, alpha=0.90)
    }


# ============ Feature Engineering ============
def build_features(df: pd.DataFrame, mode: str = "posthoc") -> Tuple[np.ndarray, List[str]]:
    """
    Complete feature engineering with all 23 features

    Args:
        df: Input dataframe
        mode: "posthoc" (all features) or "online" (no future features)
    """
    base_cols = [
        'Initial_Temperature', 'Final_Temperature',
        'Initial_Speed', 'Final_Speed',
        'Deceleration_Rate', 'Contact_Pressure', 'Braking_Time'
    ]

    X = df[base_cols].copy()

    # Temperature features
    X['Temp_Rise'] = X['Final_Temperature'] - X['Initial_Temperature']
    X['Temp_Rise_Rate'] = X['Temp_Rise'] / np.maximum(X['Braking_Time'], 1e-6)
    X['Mean_Temp'] = 0.5 * (X['Initial_Temperature'] + X['Final_Temperature'])

    # Speed and kinematics
    X['Speed_Drop_kmh'] = X['Initial_Speed'] - X['Final_Speed']
    X['Speed_Drop_ms'] = X['Speed_Drop_kmh'] / 3.6
    X['Mean_Speed_ms'] = 0.5 * (X['Initial_Speed'] + X['Final_Speed']) / 3.6
    X['Braking_Distance'] = X['Mean_Speed_ms'] * X['Braking_Time']

    # Two deceleration estimates
    X['Decel_ms2_src'] = X['Deceleration_Rate']  # From sensor
    X['Decel_ms2_kin'] = X['Speed_Drop_ms'] / np.maximum(X['Braking_Time'], 1e-6)  # Kinematic

    # Energy and power
    X['Energy_Rel'] = np.square(X['Speed_Drop_ms'])
    X['Power_Rel'] = X['Energy_Rel'] / np.maximum(X['Braking_Time'], 1e-6)

    # Pressure-time
    X['Pressure_Time'] = X['Contact_Pressure'] * X['Braking_Time']

    # Interaction terms
    X['SpeedXDecel'] = X['Mean_Speed_ms'] * X['Decel_ms2_kin']
    X['PT_over_Erel'] = X['Pressure_Time'] / np.maximum(X['Energy_Rel'], 1e-6)

    # Ratios (only in posthoc mode)
    if mode == "posthoc":
        X['V1_over_V0'] = X['Final_Speed'] / np.maximum(X['Initial_Speed'], 1e-6)
        X['Tf_over_Ti'] = X['Final_Temperature'] / np.maximum(X['Initial_Temperature'], 1e-6)

    # Remove unavailable features in online mode
    if mode == "online":
        online_drop = ['Final_Temperature', 'Final_Speed', 'Temp_Rise',
                       'Speed_Drop_kmh', 'Speed_Drop_ms', 'Mean_Temp']
        X = X.drop(columns=[c for c in online_drop if c in X.columns])

    return X.values.astype(float), list(X.columns)


def stratify_bins(y: np.ndarray, n_bins: int = 7, seed: int = 0) -> np.ndarray:
    """Create stratification bins for continuous target"""
    from sklearn.preprocessing import KBinsDiscretizer

    y = np.asarray(y, dtype=float).reshape(-1, 1)
    kbd = KBinsDiscretizer(n_bins=min(n_bins, 5), encode='ordinal', strategy='quantile')
    return kbd.fit_transform(y).ravel().astype(int)


# ============ Model Factories ============
def make_svr_pipeline(random_state: int) -> Pipeline:
    """SVR with preprocessing and log transform"""
    preprocessor = Pipeline([
        ('winsorize', Winsorizer(0.01, 0.99)),
        ('scale', RobustScaler())
    ])

    svr = SVR(kernel='rbf', cache_size=500)
    pipe = Pipeline([('preprocess', preprocessor), ('svr', svr)])

    return TransformedTargetRegressor(
        regressor=pipe,
        transformer=FunctionTransformer(
            func=np.log1p,
            inverse_func=np.expm1,
            validate=False
        )
    )


def make_hgb_pipeline(random_state: int) -> Pipeline:
    """Histogram Gradient Boosting"""
    preprocessor = Pipeline([
        ('winsorize', Winsorizer(0.01, 0.99)),
        ('scale', RobustScaler())
    ])

    hgb = HistGradientBoostingRegressor(
        loss='absolute_error',
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=10,
        random_state=random_state
    )

    pipe = Pipeline([('preprocess', preprocessor), ('hgb', hgb)])

    return TransformedTargetRegressor(
        regressor=pipe,
        transformer=FunctionTransformer(
            func=np.log1p,
            inverse_func=np.expm1,
            validate=False
        )
    )


def make_xgb_pipeline(random_state: int, feature_names: List[str],
                      config: Config) -> Optional[Pipeline]:
    """XGBoost with monotonic constraints"""
    if not HAVE_XGB:
        return None

    preprocessor = Pipeline([
        ('winsorize', Winsorizer(0.01, 0.99)),
        ('scale', RobustScaler())
    ])

    # Set up monotonic constraints
    mono_constraints = None
    if config.enable_mono_constraints:
        mono_map = {f: 1 for f in config.mono_increasing_features}
        mono_constraints = tuple(mono_map.get(f, 0) for f in feature_names)

    xgb = XGBRegressor(
        objective='reg:squarederror',
        tree_method='hist',
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        random_state=random_state,
        monotone_constraints=mono_constraints
    )

    pipe = Pipeline([('preprocess', preprocessor), ('xgb', xgb)])

    return TransformedTargetRegressor(
        regressor=pipe,
        transformer=FunctionTransformer(
            func=np.log1p,
            inverse_func=np.expm1,
            validate=False
        )
    )


def make_cat_pipeline(random_state: int, feature_names: List[str],
                      config: Config) -> Optional[Pipeline]:
    """CatBoost with monotonic constraints"""
    if not HAVE_CAT:
        return None

    # Set up monotonic constraints
    mono_constraints = {}
    if config.enable_mono_constraints:
        for i, f in enumerate(feature_names):
            if f in config.mono_increasing_features:
                mono_constraints[i] = 1

    cat = CatBoostRegressor(
        loss_function='RMSE',
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=3.0,
        iterations=500,
        random_seed=random_state,
        verbose=False,
        monotone_constraints=mono_constraints if mono_constraints else None
    )

    # Wrapper for sklearn compatibility
    class CatWrapper(BaseEstimator):
        def __init__(self, model):
            self.model = model

        def fit(self, X, y):
            self.model.fit(X, y)
            return self

        def predict(self, X):
            return self.model.predict(X)

    preprocessor = Pipeline([
        ('winsorize', Winsorizer(0.01, 0.99)),
        ('scale', RobustScaler())
    ])

    pipe = Pipeline([
        ('preprocess', preprocessor),
        ('cat', CatWrapper(cat))
    ])

    return TransformedTargetRegressor(
        regressor=pipe,
        transformer=FunctionTransformer(
            func=np.log1p,
            inverse_func=np.expm1,
            validate=False
        )
    )


# ============ Parameter Spaces ============
def get_param_space_svr(n_iter: int, random_state: int) -> Dict:
    """SVR hyperparameter space"""
    rng = np.random.RandomState(random_state)

    def log_uniform(low, high, size):
        return np.exp(rng.uniform(np.log(low), np.log(high), size))

    return {
        'regressor__svr__C': log_uniform(0.1, 100, n_iter),
        'regressor__svr__gamma': log_uniform(0.0001, 0.1, n_iter),
        'regressor__svr__epsilon': log_uniform(0.001, 0.5, n_iter)
    }


def get_param_space_hgb(n_iter: int, random_state: int) -> Dict:
    """HGB hyperparameter space"""
    rng = np.random.RandomState(random_state)
    return {
        'regressor__hgb__learning_rate': rng.choice([0.03, 0.05, 0.07, 0.1], n_iter),
        'regressor__hgb__max_depth': rng.choice([None, 3, 4, 5], n_iter),
        'regressor__hgb__max_leaf_nodes': rng.choice([31, 63, 127], n_iter),
        'regressor__hgb__min_samples_leaf': rng.choice([10, 20, 30], n_iter),
        'regressor__hgb__l2_regularization': rng.choice([0.1, 0.5, 1.0, 2.0], n_iter)
    }


def get_param_space_xgb(n_iter: int, random_state: int) -> Dict:
    """XGBoost hyperparameter space"""
    rng = np.random.RandomState(random_state)
    return {
        'regressor__xgb__learning_rate': rng.choice([0.03, 0.05, 0.07, 0.1], n_iter),
        'regressor__xgb__max_depth': rng.choice([3, 4, 5, 6], n_iter),
        'regressor__xgb__min_child_weight': rng.choice([1, 3, 5, 10], n_iter),
        'regressor__xgb__subsample': rng.choice([0.7, 0.8, 1.0], n_iter),
        'regressor__xgb__colsample_bytree': rng.choice([0.7, 0.8, 1.0], n_iter),
        'regressor__xgb__reg_lambda': rng.choice([0.5, 1.0, 2.0, 5.0], n_iter)
    }


# ============ Ensemble Optimization ============
def optimize_ensemble_weights(predictions: Dict[str, np.ndarray],
                              y_true: np.ndarray) -> Dict[str, float]:
    """
    Find optimal ensemble weights using scipy optimization
    """
    model_names = list(predictions.keys())
    n_models = len(model_names)

    # Stack predictions
    P = np.column_stack([predictions[name] for name in model_names])

    def loss_function(weights):
        """Combined loss: RMSE + 0.2 * Pinball@0.90"""
        weighted_pred = P @ weights
        rmse_loss = rmse(y_true, weighted_pred)
        pinball_loss = mean_pinball_loss(y_true, weighted_pred, alpha=0.90)
        return rmse_loss + 0.2 * pinball_loss

    # Initial weights (equal)
    initial_weights = np.ones(n_models) / n_models

    # Constraints: weights sum to 1, all >= 0
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, 1) for _ in range(n_models)]

    # Optimize
    result = minimize(
        loss_function,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000}
    )

    if result.success:
        weights = result.x
    else:
        # Fallback to equal weights if optimization fails
        weights = initial_weights

    return {name: float(weights[i]) for i, name in enumerate(model_names)}


# ============ Feature Selection ============
def compute_oof_feature_importance(X: np.ndarray, y: np.ndarray,
                                   feature_names: List[str],
                                   config: Config) -> Tuple[List[str], np.ndarray, pd.DataFrame]:
    """
    Compute feature importance using OOF permutation importance
    """
    print("\nComputing OOF feature importance...")

    importances_per_fold = []
    cv = KFold(n_splits=5, shuffle=True, random_state=config.random_state)

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X), 1):
        print(f"  Fold {fold_idx}/5...")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Use HGB as proxy (fast and robust)
        model = make_hgb_pipeline(config.random_state + fold_idx)
        model.fit(X_train, y_train)

        # Compute importance on test set
        result = permutation_importance(
            model, X_test, y_test,
            n_repeats=config.n_repeats_perm,
            random_state=config.random_state + fold_idx,
            scoring='neg_root_mean_squared_error',
            n_jobs=config.n_jobs
        )

        importances_per_fold.append(result.importances_mean)

    # Average across folds
    mean_importance = np.mean(importances_per_fold, axis=0)
    std_importance = np.std(importances_per_fold, axis=0)
    positive_ratio = np.mean(np.array(importances_per_fold) > 0, axis=0)

    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': mean_importance,
        'importance_std': std_importance,
        'positive_ratio': positive_ratio
    }).sort_values('importance_mean', ascending=False)

    # Select features
    # Criteria: positive importance and at least 60% positive across folds
    selected_mask = (mean_importance > config.importance_threshold) & (positive_ratio >= 0.6)

    # Ensure minimum features
    if np.sum(selected_mask) < config.min_features:
        # Keep top N by importance
        top_indices = np.argsort(mean_importance)[-config.min_features:]
        selected_mask[top_indices] = True

    selected_features = [f for i, f in enumerate(feature_names) if selected_mask[i]]

    print(f"  Selected {len(selected_features)}/{len(feature_names)} features")

    return selected_features, mean_importance, importance_df


# ============ Outlier Analysis ============
def analyze_fold_outliers(cv_results: Dict, y: np.ndarray, config: Config) -> Dict:
    """
    Analyze outliers in each fold
    """
    outlier_analysis = {}

    for fold_idx, fold_metrics in enumerate(cv_results['fold_results'], 1):
        # Find test indices for this fold
        fold_mask = np.zeros(len(y), dtype=bool)
        start_idx = (fold_idx - 1) * len(y) // config.n_outer_folds
        end_idx = fold_idx * len(y) // config.n_outer_folds
        fold_mask[start_idx:end_idx] = True

        fold_y = y[fold_mask]
        fold_pred = cv_results['oof_predictions']['ENS'][fold_mask]

        # Calculate residuals
        residuals = fold_pred - fold_y

        # Identify outliers
        outlier_mask = np.abs(residuals) > config.outlier_std_threshold * np.std(residuals)

        outlier_analysis[f'fold_{fold_idx}'] = {
            'rmse': fold_metrics['ENS']['rmse'],
            'n_samples': len(fold_y),
            'n_outliers': np.sum(outlier_mask),
            'outlier_ratio': np.mean(outlier_mask),
            'y_mean': np.mean(fold_y),
            'y_std': np.std(fold_y),
            'max_residual': np.max(np.abs(residuals))
        }

    return outlier_analysis


# ============ Main Pipeline ============
def run_nested_cv(X: np.ndarray, y: np.ndarray,
                  feature_names: List[str],
                  config: Config) -> Dict:
    """
    Run nested cross-validation with proper inner CV for regression
    """
    n_samples = len(y)

    # Initialize OOF predictions
    models_to_use = ['SVR', 'HGB']
    if config.use_xgb:
        models_to_use.append('XGB')
    if config.use_cat:
        models_to_use.append('CAT')

    oof_predictions = {model: np.zeros(n_samples) for model in models_to_use}
    oof_predictions['ENS'] = np.zeros(n_samples)

    y_oof = np.full(n_samples, np.nan)

    # Results storage
    fold_results = []
    ensemble_weights_per_fold = []
    best_params_per_fold = []

    # Outer CV - can use stratification
    y_bins = stratify_bins(y, n_bins=5, seed=config.random_state)
    cv_outer = StratifiedKFold(
        n_splits=config.n_outer_folds,
        shuffle=True,
        random_state=config.random_state
    )

    print(f"\nRunning nested CV ({config.n_outer_folds} outer folds)...")

    for fold_idx, (train_idx, test_idx) in enumerate(cv_outer.split(X, y_bins), 1):
        print(f"\n  Fold {fold_idx}/{config.n_outer_folds}")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # CRITICAL: Use regular KFold for inner CV (regression)
        cv_inner = KFold(
            n_splits=config.n_inner_folds,
            shuffle=True,
            random_state=config.random_state + fold_idx
        )

        fold_predictions = {}
        fold_params = {}

        # Train SVR
        print("    Training SVR...")
        svr_model = make_svr_pipeline(config.random_state + fold_idx)
        svr_search = RandomizedSearchCV(
            estimator=svr_model,
            param_distributions=get_param_space_svr(config.n_iter_svr, config.random_state + fold_idx),
            n_iter=config.n_iter_svr,
            cv=cv_inner,
            scoring='neg_root_mean_squared_error',
            n_jobs=config.n_jobs,
            random_state=config.random_state + fold_idx,
            verbose=0
        )
        svr_search.fit(X_train, y_train)
        fold_predictions['SVR'] = svr_search.predict(X_test)
        fold_params['SVR'] = svr_search.best_params_

        # Train HGB
        print("    Training HGB...")
        hgb_model = make_hgb_pipeline(config.random_state + fold_idx)
        hgb_search = RandomizedSearchCV(
            estimator=hgb_model,
            param_distributions=get_param_space_hgb(config.n_iter_hgb, config.random_state + fold_idx + 1),
            n_iter=config.n_iter_hgb,
            cv=cv_inner,
            scoring='neg_root_mean_squared_error',
            n_jobs=config.n_jobs,
            random_state=config.random_state + fold_idx + 1,
            verbose=0
        )
        hgb_search.fit(X_train, y_train)
        fold_predictions['HGB'] = hgb_search.predict(X_test)
        fold_params['HGB'] = hgb_search.best_params_

        # Train XGBoost if available
        if config.use_xgb:
            print("    Training XGBoost...")
            xgb_model = make_xgb_pipeline(config.random_state + fold_idx, feature_names, config)
            xgb_search = RandomizedSearchCV(
                estimator=xgb_model,
                param_distributions=get_param_space_xgb(config.n_iter_xgb, config.random_state + fold_idx + 2),
                n_iter=config.n_iter_xgb,
                cv=cv_inner,
                scoring='neg_root_mean_squared_error',
                n_jobs=config.n_jobs,
                random_state=config.random_state + fold_idx + 2,
                verbose=0
            )
            xgb_search.fit(X_train, y_train)
            fold_predictions['XGB'] = xgb_search.predict(X_test)
            fold_params['XGB'] = xgb_search.best_params_

        # Train CatBoost if available
        if config.use_cat:
            print("    Training CatBoost...")
            cat_model = make_cat_pipeline(config.random_state + fold_idx, feature_names, config)
            cat_model.fit(X_train, y_train)
            fold_predictions['CAT'] = cat_model.predict(X_test)
            fold_params['CAT'] = {'default': True}

        # Ensemble weights
        if config.optimize_weights:
            print("    Optimizing ensemble weights...")
            ensemble_weights = optimize_ensemble_weights(fold_predictions, y_test)
        else:
            # Simple average
            ensemble_weights = {name: 1.0 / len(fold_predictions) for name in fold_predictions}

        # Compute ensemble prediction
        fold_predictions['ENS'] = sum(
            w * fold_predictions[name]
            for name, w in ensemble_weights.items()
        )

        # Store OOF predictions
        y_oof[test_idx] = y_test
        for name, pred in fold_predictions.items():
            if name in oof_predictions:
                oof_predictions[name][test_idx] = pred

        # Store fold results
        fold_metrics = {}
        for name, pred in fold_predictions.items():
            fold_metrics[name] = compute_metrics(y_test, pred)

        fold_results.append(fold_metrics)
        ensemble_weights_per_fold.append(ensemble_weights)
        best_params_per_fold.append(fold_params)

        # Print fold summary
        print(f"    Fold {fold_idx} RMSE: " +
              ", ".join(f"{name}={m['rmse']:.4f}"
                        for name, m in fold_metrics.items()))
        print(f"    Ensemble weights: " +
              ", ".join(f"{name}={w:.3f}" for name, w in ensemble_weights.items()))

    # Compute overall OOF metrics
    overall_metrics = {}
    for name in oof_predictions:
        overall_metrics[name] = compute_metrics(y_oof, oof_predictions[name])

    return {
        'oof_predictions': oof_predictions,
        'y_oof': y_oof,
        'overall_metrics': overall_metrics,
        'fold_results': fold_results,
        'ensemble_weights': ensemble_weights_per_fold,
        'best_params': best_params_per_fold
    }


# ============ Visualization ============
def create_visualizations(results: Dict, feature_importance_df: Optional[pd.DataFrame],
                          outlier_analysis: Optional[Dict], config: Config):
    """Create comprehensive visualization plots"""

    # Set up figure
    n_rows = 3 if feature_importance_df is not None else 2
    fig = plt.figure(figsize=(16, 6 * n_rows))

    plot_idx = 1

    # 1. Feature importance (if available)
    if feature_importance_df is not None:
        ax1 = plt.subplot(n_rows, 3, plot_idx)
        plot_idx += 1

        top_features = feature_importance_df.head(15)
        colors = ['green' if imp > 0 else 'red' for imp in top_features['importance_mean']]

        ax1.barh(range(len(top_features)), top_features['importance_mean'], color=colors, alpha=0.7)
        ax1.set_yticks(range(len(top_features)))
        ax1.set_yticklabels(top_features['feature'], fontsize=9)
        ax1.set_xlabel('Permutation Importance')
        ax1.set_title('Top 15 Feature Importance')
        ax1.invert_yaxis()

    # 2. Model comparison
    ax2 = plt.subplot(n_rows, 3, plot_idx)
    plot_idx += 1

    models = list(results['overall_metrics'].keys())
    metrics_to_plot = ['r2', 'rmse', 'mae']

    x = np.arange(len(metrics_to_plot))
    width = 0.15

    for i, model in enumerate(models):
        values = [results['overall_metrics'][model][m] for m in metrics_to_plot]
        ax2.bar(x + i * width, values, width, label=model)

    ax2.set_xticks(x + width * (len(models) - 1) / 2)
    ax2.set_xticklabels(['R²', 'RMSE', 'MAE'])
    ax2.set_title('Model Performance Comparison')
    ax2.legend()

    # 3. Actual vs Predicted
    ax3 = plt.subplot(n_rows, 3, plot_idx)
    plot_idx += 1

    y_true = results['y_oof']
    y_pred = results['oof_predictions']['ENS']

    ax3.scatter(y_true, y_pred, alpha=0.5, s=20)
    ax3.plot([y_true.min(), y_true.max()],
             [y_true.min(), y_true.max()],
             'r--', lw=2)
    ax3.set_xlabel('Actual PM2.5')
    ax3.set_ylabel('Predicted PM2.5')
    ax3.set_title(f'Ensemble Predictions (R²={r2_score(y_true, y_pred):.3f})')

    # 4. Residual plot
    ax4 = plt.subplot(n_rows, 3, plot_idx)
    plot_idx += 1

    residuals = y_pred - y_true
    ax4.scatter(y_pred, residuals, alpha=0.5, s=20)
    ax4.axhline(y=0, color='r', linestyle='--')
    ax4.set_xlabel('Predicted PM2.5')
    ax4.set_ylabel('Residuals')
    ax4.set_title('Residual Plot')

    # 5. Fold performance
    ax5 = plt.subplot(n_rows, 3, plot_idx)
    plot_idx += 1

    fold_rmse = [fold['ENS']['rmse'] for fold in results['fold_results']]
    ax5.bar(range(1, len(fold_rmse) + 1), fold_rmse, alpha=0.7)
    ax5.axhline(y=np.mean(fold_rmse), color='r', linestyle='--', label=f'Mean: {np.mean(fold_rmse):.3f}')
    ax5.set_xlabel('Fold')
    ax5.set_ylabel('RMSE')
    ax5.set_title('RMSE by Fold')
    ax5.legend()

    # 6. Distribution comparison
    ax6 = plt.subplot(n_rows, 3, plot_idx)
    plot_idx += 1

    ax6.hist(y_true, bins=30, alpha=0.5, label='Actual', density=True)
    ax6.hist(y_pred, bins=30, alpha=0.5, label='Predicted', density=True)
    ax6.set_xlabel('PM2.5')
    ax6.set_ylabel('Density')
    ax6.set_title('Distribution Comparison')
    ax6.legend()

    # 7. Ensemble weights across folds
    if len(results['ensemble_weights']) > 0:
        ax7 = plt.subplot(n_rows, 3, plot_idx)
        plot_idx += 1

        # Extract weights for each model across folds
        model_names = list(results['ensemble_weights'][0].keys())
        weights_matrix = np.array([[w[m] for m in model_names]
                                   for w in results['ensemble_weights']])

        x = np.arange(len(results['ensemble_weights']))
        bottom = np.zeros(len(x))

        for i, model in enumerate(model_names):
            ax7.bar(x, weights_matrix[:, i], bottom=bottom, label=model, alpha=0.8)
            bottom += weights_matrix[:, i]

        ax7.set_xlabel('Fold')
        ax7.set_ylabel('Weight')
        ax7.set_title('Ensemble Weights by Fold')
        ax7.legend()
        ax7.set_xticks(x)
        ax7.set_xticklabels([f'F{i + 1}' for i in x])

    plt.tight_layout()

    # Save figure
    output_path = Path(config.output_dir) / 'complete_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nPlots saved to {output_path}")


# ============ Report Generation ============
def generate_report(results: Dict, feature_importance_df: Optional[pd.DataFrame],
                    selected_features: Optional[List[str]],
                    outlier_analysis: Optional[Dict],
                    config: Config) -> str:
    """Generate comprehensive markdown report"""

    lines = []
    lines.append("# PM2.5 Brake Emissions Prediction - Complete Analysis")
    lines.append(f"\nGenerated: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Configuration
    lines.append("\n## Configuration")
    lines.append(f"- Data: {config.data_path}")
    lines.append(f"- Models: SVR, HGB" +
                 (", XGBoost" if config.use_xgb else "") +
                 (", CatBoost" if config.use_cat else ""))
    lines.append(f"- CV: {config.n_outer_folds} outer folds, {config.n_inner_folds} inner folds")
    lines.append(f"- Feature selection: {'Enabled' if config.run_feature_selection else 'Disabled'}")
    lines.append(f"- Ensemble optimization: {'Enabled' if config.optimize_weights else 'Disabled'}")

    # Overall performance
    lines.append("\n## Overall OOF Performance")
    lines.append("| Model | R² | RMSE | MAE | MedAE | sMAPE | Q90AE | Pinball@90 |")
    lines.append("|-------|-----|------|-----|-------|-------|-------|------------|")

    for model_name, metrics in results['overall_metrics'].items():
        lines.append(f"| {model_name:5} | "
                     f"{metrics['r2']:.4f} | "
                     f"{metrics['rmse']:.4f} | "
                     f"{metrics['mae']:.4f} | "
                     f"{metrics['medae']:.4f} | "
                     f"{metrics['smape']:.4f} | "
                     f"{metrics['q90ae']:.4f} | "
                     f"{metrics['pinball90']:.4f} |")

    # Feature importance
    if feature_importance_df is not None:
        lines.append("\n## Feature Importance Analysis")
        lines.append(f"Selected {len(selected_features)} features\n")

        lines.append("### Top 10 Features")
        lines.append("| Rank | Feature | Importance | Std | Positive Ratio |")
        lines.append("|------|---------|------------|-----|----------------|")

        for idx, row in feature_importance_df.head(10).iterrows():
            lines.append(f"| {idx + 1} | {row['feature']} | "
                         f"{row['importance_mean']:.6f} | "
                         f"{row['importance_std']:.6f} | "
                         f"{row['positive_ratio']:.2f} |")

    # Fold-by-fold results
    lines.append("\n## Fold-by-Fold Performance")
    lines.append("| Fold | SVR | HGB | XGB | CAT | ENS |")
    lines.append("|------|-----|-----|-----|-----|-----|")

    for i, fold in enumerate(results['fold_results'], 1):
        line = f"| {i} | "
        for model in ['SVR', 'HGB', 'XGB', 'CAT', 'ENS']:
            if model in fold:
                line += f"{fold[model]['rmse']:.4f} | "
            else:
                line += "N/A | "
        lines.append(line)

    # Ensemble weights
    lines.append("\n## Ensemble Weights")
    avg_weights = {}
    for weights in results['ensemble_weights']:
        for name, w in weights.items():
            if name not in avg_weights:
                avg_weights[name] = []
            avg_weights[name].append(w)

    lines.append("| Model | Average Weight | Std |")
    lines.append("|-------|---------------|-----|")
    for name, weights in avg_weights.items():
        lines.append(f"| {name} | {np.mean(weights):.3f} | {np.std(weights):.3f} |")

    # Outlier analysis
    if outlier_analysis:
        lines.append("\n## Outlier Analysis")
        lines.append("| Fold | RMSE | Samples | Outliers | Outlier % | Max Residual |")
        lines.append("|------|------|---------|----------|-----------|--------------|")

        for fold_name, stats in outlier_analysis.items():
            lines.append(f"| {fold_name.replace('fold_', '')} | "
                         f"{stats['rmse']:.4f} | "
                         f"{stats['n_samples']} | "
                         f"{stats['n_outliers']} | "
                         f"{stats['outlier_ratio'] * 100:.1f}% | "
                         f"{stats['max_residual']:.2f} |")

    return "\n".join(lines)


# ============ Main Execution ============
def main():
    """Main execution function"""

    # Initialize configuration
    config = Config()

    # Load data
    print(f"Loading data from {config.data_path}...")
    df = pd.read_excel(config.data_path)

    if config.target_col not in df.columns:
        raise ValueError(f"Target column '{config.target_col}' not found")

    print(f"Loaded {len(df)} samples")

    # Build features
    X, feature_names = build_features(df, mode="posthoc")
    y = df[config.target_col].values.astype(float)

    print(f"Features ({len(feature_names)}): {feature_names}")

    # Feature selection (if enabled)
    if config.run_feature_selection:
        selected_features, importance_scores, importance_df = compute_oof_feature_importance(
            X, y, feature_names, config
        )

        # Use selected features
        selected_idx = [feature_names.index(f) for f in selected_features]
        X_selected = X[:, selected_idx]
        features_to_use = selected_features

        print(f"\nUsing {len(selected_features)} selected features")
    else:
        X_selected = X
        features_to_use = feature_names
        selected_features = None
        importance_df = None

    # Run nested CV
    results = run_nested_cv(X_selected, y, features_to_use, config)

    # Outlier analysis
    outlier_analysis = None
    if config.analyze_outliers:
        outlier_analysis = analyze_fold_outliers(results, y, config)

    # Print summary
    print("\n" + "=" * 60)
    print("OVERALL OOF PERFORMANCE:")
    print("=" * 60)

    for name, metrics in results['overall_metrics'].items():
        print(f"{name:5}: R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}, "
              f"MAE={metrics['mae']:.4f}")

    # Generate visualizations
    create_visualizations(results, importance_df, outlier_analysis, config)

    # Generate report
    report = generate_report(results, importance_df, selected_features,
                             outlier_analysis, config)

    # Save report
    report_path = Path(config.output_dir) / 'complete_analysis_report.md'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")

    # Save results
    results_df = pd.DataFrame({
        'y_true': results['y_oof'],
        **{f'{name}_pred': pred for name, pred in results['oof_predictions'].items()}
    })

    results_path = Path(config.output_dir) / 'oof_predictions_complete.csv'
    results_df.to_csv(results_path, index=False)
    print(f"Predictions saved to {results_path}")

    # Save model bundle
    try:
        import joblib
        bundle = {
            'config': config,
            'results': results,
            'feature_names': feature_names,
            'selected_features': selected_features,
            'importance_df': importance_df
        }
        bundle_path = Path(config.output_dir) / 'model_bundle_complete.joblib'
        joblib.dump(bundle, bundle_path)
        print(f"Model bundle saved to {bundle_path}")
    except ImportError:
        print("joblib not available - skipping model save")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()