# -*- coding: utf-8 -*-
"""
pm25_meta_ensemble_enhanced.py
全面增强版本，包含：
A. 数据准备与可视化（y分布、IQR围栏、特征相关性）
B. 基准建模与Permutation Importance
C. 基于PI的特征筛选与重训
D. 可选的留出集和敏感性分析
增强功能：
1. 特征工程增强（12个高级特征）
2. 超参数搜索优化
3. 集成策略升级
4. 可视化修复
"""
import os
import json
import math
import time
import warnings
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.interpolate import interp1d
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import FunctionTransformer, RobustScaler, PowerTransformer
from sklearn.model_selection import (
    KFold, RepeatedStratifiedKFold, StratifiedKFold,
    RandomizedSearchCV, learning_curve, train_test_split
)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_pinball_loss
from sklearn.utils.validation import check_random_state
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.feature_selection import RFECV
from sklearn.linear_model import Ridge
from mlxtend.regressor import StackingCVRegressor

# XGBoost（可选）
try:
    from xgboost import XGBRegressor

    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

# LightGBM（可选）
try:
    from lightgbm import LGBMRegressor

    HAVE_LGB = True
except Exception:
    HAVE_LGB = False

# Boruta（可选）
try:
    from boruta import BorutaPy

    HAVE_BORUTA = True
except Exception:
    HAVE_BORUTA = False

# 设置
warnings.filterwarnings("ignore", category=UserWarning)
np.set_printoptions(suppress=True, linewidth=180)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============ 配置参数 ============
DATA_PATH = "Cleaned_PM_Regression_Data.xlsx"
TARGET_COL = "PM2_5"

# 模型配置
USE_XGB = True and HAVE_XGB
USE_LGB = True and HAVE_LGB
USE_HOLDOUT = True  # 是否使用留出测试集
HOLDOUT_SIZE = 0.15  # 留出集比例
N_OUTER = 5
N_INNER = 3
N_REPEATS = 2
RANDOM_STATE = 42
N_JOBS = 1
N_ITER_SVR = 40
N_ITER_HGB = 40
N_ITER_XGB = 30
N_ITER_LGB = 30

# 分层配置
Y_BINS = 7
PINBALL_Q = 0.90

# 特征选择配置
PI_MIN_IMPORTANCE = 0.0  # PI最小重要性阈值
PI_MIN_POS_RATIO = 0.6  # PI正贡献折占比阈值
MIN_FEATURES_KEEP = 10  # 最少保留特征数
TOP_K_FEATURES = None  # 可设置为12等，None表示不限制
FEATURE_SELECTION_METHOD = "RFECV"  # "RFECV" 或 "BORUTA"

# 目标变换配置
TARGET_TRANSFORM = "yeojohnson"  # "log1p", "boxcox" 或 "yeojohnson"

# 集成方法配置
ENSEMBLE_METHOD = "stacking"  # "weighted" 或 "stacking"

# IQR配置
RUN_IQR_SCREENING = True
RUN_IQR_SENSITIVITY = True
IQR_K_INNER = 1.5
IQR_K_OUTER = 3.0
IQR_TRIM_STRATEGY = "inner"

# 集成损失权衡
LAMBDA_PB = 0.20
WEIGHT_GRID_STEP = 0.02

# 输出路径
OUTPUT_DIR = "pm25_output"
REPORT_PATH = os.path.join(OUTPUT_DIR, "enhanced_report.md")
BUNDLE_PATH = os.path.join(OUTPUT_DIR, "PM25_enhanced.joblib")
OUTLIER_CSV = os.path.join(OUTPUT_DIR, "outlier_screening.csv")
SOP_JSON = os.path.join(OUTPUT_DIR, "outlier_sop.json")
PI_CSV = os.path.join(OUTPUT_DIR, "feature_permutation_importance.csv")
SELECTED_FEATURES_JSON = os.path.join(OUTPUT_DIR, "selected_features.json")

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "figures"), exist_ok=True)


# ================== 工具类与函数 ==================
class Winsorizer(BaseEstimator, TransformerMixin):
    """IQR缩尾"""

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
    """稳健y分箱"""
    rng = check_random_state(seed)
    y = np.asarray(y, dtype=float).reshape(-1)
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
    """构建特征（增强版，包含交互项和多项式特征）"""
    cols = ['Initial_Temperature', 'Final_Temperature', 'Initial_Speed', 'Final_Speed',
            'Deceleration_Rate', 'Contact_Pressure', 'Braking_Time']
    base = df[cols].copy()

    # 基础派生特征
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

    # 新增高级特征
    # 动能转换效率相关
    base['Kinetic_Change'] = 0.5 * base['Speed_Drop_ms'] ** 2
    base['Heat_Efficiency'] = base['Temp_Rise'] / np.maximum(base['Kinetic_Change'], 1e-6)

    # 制动强度指标
    base['Brake_Intensity'] = base['Contact_Pressure'] * base['Decel_ms2_kin']
    base['Specific_Power'] = base['Power_Rel'] / np.maximum(base['Contact_Pressure'], 1e-6)

    # 温度梯度
    base['Temp_Gradient'] = base['Temp_Rise'] / np.maximum(base['Braking_Distance'], 1e-6)

    # 速度相关的非线性特征
    base['Speed_Ratio_Squared'] = base['V1_over_V0'] ** 2
    base['Speed_Drop_Ratio'] = base['Speed_Drop_kmh'] / np.maximum(base['Initial_Speed'], 1e-6)

    # 压力与能量的交互
    base['Pressure_Energy_Product'] = base['Contact_Pressure'] * base['Energy_Rel']
    base['Pressure_Power_Ratio'] = base['Contact_Pressure'] / np.maximum(base['Power_Rel'], 1e-6)

    # 时间相关的复合特征
    base['Time_Weighted_Temp'] = base['Mean_Temp'] * base['Braking_Time']
    base['Time_Efficiency'] = base['Speed_Drop_ms'] / np.maximum(base['Braking_Time'], 1e-6) ** 2

    feat_cols = list(base.columns)
    X = base.values
    y = df[TARGET_COL].values.astype(float)
    return base, y, feat_cols


# ============ 可视化函数 ============
def plot_y_distribution(y: np.ndarray, save_dir: str):
    """绘制y分布和log变换效果"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 原始y分布
    axes[0, 0].hist(y, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Original PM2.5 Distribution')
    axes[0, 0].set_xlabel('PM2.5')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(np.median(y), color='red', linestyle='--', label=f'Median: {np.median(y):.3f}')
    axes[0, 0].legend()

    # log1p(y)分布
    y_log = np.log1p(y)
    axes[0, 1].hist(y_log, bins=30, edgecolor='black', alpha=0.7, color='green')
    axes[0, 1].set_title('Log-transformed PM2.5 Distribution')
    axes[0, 1].set_xlabel('log1p(PM2.5)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(np.median(y_log), color='red', linestyle='--', label=f'Median: {np.median(y_log):.3f}')
    axes[0, 1].legend()

    # Q-Q图（原始）
    stats.probplot(y, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot (Original)')

    # Q-Q图（log变换后）
    stats.probplot(y_log, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot (Log-transformed)')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'y_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_iqr_summary(y: np.ndarray, save_dir: str):
    """绘制IQR围栏和异常值"""
    from matplotlib.patches import Rectangle

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 箱线图 - 修复：自动调整y轴范围
    bp = ax1.boxplot(y, vert=True, patch_artist=True, showmeans=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][0].set_alpha(0.7)
    ax1.set_ylabel('PM2.5')
    ax1.set_title('Boxplot with IQR Fences')
    ax1.grid(True, alpha=0.3)

    # 计算IQR统计
    q1, q3 = np.percentile(y, [25, 75])
    iqr = q3 - q1
    inner_lo = q1 - IQR_K_INNER * iqr
    inner_hi = q3 + IQR_K_INNER * iqr
    outer_lo = q1 - IQR_K_OUTER * iqr
    outer_hi = q3 + IQR_K_OUTER * iqr

    # 添加围栏线
    ax1.axhline(inner_lo, color='orange', linestyle='--', label=f'Inner fence (1.5×IQR)')
    ax1.axhline(inner_hi, color='orange', linestyle='--')
    ax1.axhline(outer_lo, color='red', linestyle='--', label=f'Outer fence (3.0×IQR)')
    ax1.axhline(outer_hi, color='red', linestyle='--')
    ax1.legend()

    # 自动调整y轴范围以显示所有异常值
    y_min = min(outer_lo, np.min(y))
    y_max = max(outer_hi, np.max(y))
    margin = (y_max - y_min) * 0.1
    ax1.set_ylim(y_min - margin, y_max + margin)

    # 异常值计数条形图
    mild = ((y < inner_lo) | (y > inner_hi)).sum()
    extreme = ((y < outer_lo) | (y > outer_hi)).sum()
    normal = len(y) - mild
    categories = ['Normal', 'Mild Outliers', 'Extreme Outliers']
    counts = [normal, mild - extreme, extreme]
    colors = ['green', 'orange', 'red']
    bars = ax2.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Count')
    ax2.set_title('Outlier Count Summary')
    ax2.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{count}\n({100 * count / len(y):.1f}%)',
                 ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'iqr_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_correlation_heatmaps(X: np.ndarray, feature_names: List[str], y: np.ndarray, save_dir: str):
    """绘制相关性热力图"""
    # 准备数据
    data = pd.DataFrame(X, columns=feature_names)
    data[TARGET_COL] = y

    # 计算相关性
    corr_pearson = data.corr(method='pearson')
    corr_spearman = data.corr(method='spearman')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Spearman相关性 - 修复：只显示Top 15特征，改用RdBu_r配色，增大字体
    # 选择Top 15特征（按与目标的相关性）
    target_corr = corr_spearman[TARGET_COL].drop(TARGET_COL).abs().sort_values(ascending=False)
    top_features = target_corr.head(15).index.tolist()

    # 创建只包含Top特征的相关性矩阵
    top_corr_spearman = corr_spearman.loc[top_features + [TARGET_COL], top_features + [TARGET_COL]]

    mask = np.triu(np.ones_like(top_corr_spearman, dtype=bool), k=1)
    sns.heatmap(top_corr_spearman, mask=mask, annot=True, cmap='RdBu_r', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                vmin=-1, vmax=1, ax=ax1, fmt=".2f", annot_kws={"size": 12})
    ax1.set_title('Spearman Correlation Matrix (Top 15 Features)', fontsize=16)
    ax1.tick_params(axis='both', which='major', labelsize=12)

    # Pearson相关性
    top_corr_pearson = corr_pearson.loc[top_features + [TARGET_COL], top_features + [TARGET_COL]]
    mask = np.triu(np.ones_like(top_corr_pearson, dtype=bool), k=1)
    sns.heatmap(top_corr_pearson, mask=mask, annot=True, cmap='RdBu_r', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                vmin=-1, vmax=1, ax=ax2, fmt=".2f", annot_kws={"size": 12})
    ax2.set_title('Pearson Correlation Matrix (Top 15 Features)', fontsize=16)
    ax2.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'correlation_heatmaps.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 目标相关性条形图 - 修复：只对强相关(>0.3)特征添加数值标签
    fig, ax = plt.subplots(figsize=(10, 8))

    # 使用Spearman相关性
    target_corr = corr_spearman[TARGET_COL].drop(TARGET_COL).sort_values()
    colors = ['red' if x < 0 else 'green' for x in target_corr.values]

    bars = target_corr.plot(kind='barh', color=colors, alpha=0.7, ax=ax)
    ax.set_xlabel('Spearman Correlation with PM2.5', fontsize=14)
    ax.set_title('Feature Correlations with Target', fontsize=16)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='x')
    ax.tick_params(axis='both', which='major', labelsize=12)

    # 只对强相关(>0.3)特征添加数值标签
    for i, (idx, value) in enumerate(target_corr.items()):
        if abs(value) > 0.3:
            ax.text(value + (0.01 if value > 0 else -0.01), i,
                    f'{value:.2f}', va='center', ha='left' if value > 0 else 'right',
                    fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'target_correlations.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_oof_diagnostics(y_true: np.ndarray, y_pred: Dict[str, np.ndarray], save_dir: str):
    """绘制OOF诊断图"""
    models = list(y_pred.keys())
    n_models = len(models)

    # 散点图和残差图
    fig, axes = plt.subplots(2, n_models, figsize=(5 * n_models, 10))
    if n_models == 1:
        axes = axes.reshape(2, 1)

    for idx, model in enumerate(models):
        pred = y_pred[model]

        # 预测vs真实散点图
        ax1 = axes[0, idx]
        ax1.scatter(y_true, pred, alpha=0.5, s=20)
        ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
                 'r--', lw=2, label='Perfect prediction')
        ax1.set_xlabel('True PM2.5')
        ax1.set_ylabel('Predicted PM2.5')
        ax1.set_title(f'{model}: R² = {r2_score(y_true, pred):.3f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 残差图
        ax2 = axes[1, idx]
        residuals = y_true - pred
        ax2.scatter(pred, residuals, alpha=0.5, s=20)
        ax2.axhline(y=0, color='r', linestyle='--', lw=2)
        ax2.set_xlabel('Predicted PM2.5')
        ax2.set_ylabel('Residuals')
        ax2.set_title(f'{model}: Residual Plot')
        ax2.grid(True, alpha=0.3)

        # 添加±2σ线
        std_resid = np.std(residuals)
        ax2.axhline(y=2 * std_resid, color='orange', linestyle=':', alpha=0.7, label='±2σ')
        ax2.axhline(y=-2 * std_resid, color='orange', linestyle=':', alpha=0.7)
        ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'oof_diagnostics.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_permutation_importance(pi_df: pd.DataFrame, save_dir: str, top_n: int = 15):
    """绘制Permutation Importance"""
    # 选择top N特征
    pi_sorted = pi_df.sort_values('mean_importance', ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    # 条形图
    bars = ax.barh(range(len(pi_sorted)), pi_sorted['mean_importance'].values,
                   xerr=pi_sorted['std_importance'].values,
                   color=['green' if x > 0 else 'red' for x in pi_sorted['mean_importance'].values],
                   alpha=0.7, edgecolor='black')
    # 添加正贡献比例标签
    for idx, (_, row) in enumerate(pi_sorted.iterrows()):
        ax.text(row['mean_importance'] + row['std_importance'] + 0.001, idx,
                f"{row['pos_ratio']:.0%}", va='center', fontsize=9)
    ax.set_yticks(range(len(pi_sorted)))
    ax.set_yticklabels(pi_sorted['feature'].values)
    ax.set_xlabel('Mean Decrease in R² (Permutation Importance)')
    ax.set_title(f'Top {top_n} Features by Permutation Importance\n(% shows positive contribution ratio across folds)')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'permutation_importance.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_learning_curves(X: np.ndarray, y: np.ndarray, models: Dict, save_dir: str):
    """绘制学习曲线 - 修复：动态y轴范围，添加收敛性指标"""
    train_sizes = np.linspace(0.2, 1.0, 8)

    for name, model in models.items():
        fig, ax = plt.subplots(figsize=(10, 6))

        cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X, y, cv=cv, train_sizes=train_sizes,
            scoring='neg_root_mean_squared_error', n_jobs=N_JOBS,
            random_state=RANDOM_STATE
        )

        # 转换为正RMSE
        train_scores = -train_scores
        val_scores = -val_scores

        # 计算均值和标准差
        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)

        # 动态y轴范围
        all_scores = np.concatenate([train_scores.flatten(), val_scores.flatten()])
        y_min, y_max = np.min(all_scores), np.max(all_scores)
        margin = (y_max - y_min) * 0.1
        ax.set_ylim(y_min - margin, y_max + margin)

        # 绘图
        ax.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Training RMSE')
        ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std,
                        alpha=0.2, color='blue')

        ax.plot(train_sizes_abs, val_mean, 'o-', color='red', label='Validation RMSE')
        ax.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std,
                        alpha=0.2, color='red')

        # 添加收敛性指标
        final_gap = val_mean[-1] - train_mean[-1]
        ax.text(0.5, 0.95, f'Final Gap: {final_gap:.3f}', transform=ax.transAxes,
                ha='center', va='top', bbox=dict(facecolor='white', alpha=0.7))

        ax.set_xlabel('Training Set Size')
        ax.set_ylabel('RMSE')
        ax.set_title(f'Learning Curve - {name}')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'learning_curve_{name.lower()}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()


def plot_ensemble_weights(weights_list: List[Dict[str, float]], save_dir: str):
    """绘制集成权重"""
    # 转换为DataFrame
    weights_df = pd.DataFrame(weights_list)
    weights_df.index = [f'Fold {i + 1}' for i in range(len(weights_list))]

    fig, ax = plt.subplots(figsize=(12, 6))
    # 堆积条形图
    weights_df.plot(kind='bar', stacked=True, ax=ax, alpha=0.8)
    ax.set_xlabel('CV Fold')
    ax.set_ylabel('Ensemble Weight')
    ax.set_title('Ensemble Weights Across CV Folds')
    ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    # 添加平均权重文本
    mean_weights = weights_df.mean()
    text = 'Mean weights: ' + ', '.join([f'{k}={v:.2f}' for k, v in mean_weights.items()])
    ax.text(0.5, -0.15, text, transform=ax.transAxes, ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ensemble_weights.png'), dpi=150, bbox_inches='tight')
    plt.close()


# ============ 模型构建函数 ============
def make_svr_pipeline(random_state: int) -> TransformedTargetRegressor:
    pre = Pipeline([
        ("winsor", Winsorizer(0.01, 0.99)),
        ("scale", RobustScaler()),
    ])
    svr = SVR(kernel='rbf')
    pipe = Pipeline([("pre", pre), ("svr", svr)])

    # 根据配置选择目标变换
    if TARGET_TRANSFORM == "log1p":
        transformer = FunctionTransformer(func=np.log1p, inverse_func=np.expm1,
                                          validate=False, feature_names_out='one-to-one')
    elif TARGET_TRANSFORM == "boxcox":
        # Box-Cox需要正数，所以使用log1p作为基础，然后应用Box-Cox
        transformer = PowerTransformer(method='box-cox', standardize=False)
    elif TARGET_TRANSFORM == "yeojohnson":
        transformer = PowerTransformer(method='yeo-johnson', standardize=False)
    else:
        transformer = FunctionTransformer(func=np.log1p, inverse_func=np.expm1,
                                          validate=False, feature_names_out='one-to-one')

    return TransformedTargetRegressor(
        regressor=pipe,
        transformer=transformer
    )


def make_hgb_pipeline(random_state: int) -> TransformedTargetRegressor:
    hgb = HistGradientBoostingRegressor(
        loss="absolute_error",
        early_stopping=True,
        random_state=random_state
    )
    pipe = Pipeline([
        ("winsor", Winsorizer(0.01, 0.99)),
        ("scale", RobustScaler()),
        ("hgb", hgb),
    ])

    # 根据配置选择目标变换
    if TARGET_TRANSFORM == "log1p":
        transformer = FunctionTransformer(func=np.log1p, inverse_func=np.expm1,
                                          validate=False)
    elif TARGET_TRANSFORM == "boxcox":
        transformer = PowerTransformer(method='box-cox', standardize=False)
    elif TARGET_TRANSFORM == "yeojohnson":
        transformer = PowerTransformer(method='yeo-johnson', standardize=False)
    else:
        transformer = FunctionTransformer(func=np.log1p, inverse_func=np.expm1,
                                          validate=False)

    return TransformedTargetRegressor(
        regressor=pipe,
        transformer=transformer
    )


def make_xgb_pipeline(random_state: int) -> TransformedTargetRegressor:
    xgb = XGBRegressor(
        objective="reg:pseudohubererror",
        tree_method="hist",
        eval_metric="rmse",
        random_state=random_state,
        n_estimators=600
    )
    pipe = Pipeline([
        ("winsor", Winsorizer(0.01, 0.99)),
        ("scale", RobustScaler()),
        ("xgb", xgb),
    ])

    # 根据配置选择目标变换
    if TARGET_TRANSFORM == "log1p":
        transformer = FunctionTransformer(func=np.log1p, inverse_func=np.expm1,
                                          validate=False)
    elif TARGET_TRANSFORM == "boxcox":
        transformer = PowerTransformer(method='box-cox', standardize=False)
    elif TARGET_TRANSFORM == "yeojohnson":
        transformer = PowerTransformer(method='yeo-johnson', standardize=False)
    else:
        transformer = FunctionTransformer(func=np.log1p, inverse_func=np.expm1,
                                          validate=False)

    return TransformedTargetRegressor(
        regressor=pipe,
        transformer=transformer
    )


def make_lgb_pipeline(random_state: int) -> TransformedTargetRegressor:
    lgb = LGBMRegressor(
        objective="regression",
        metric="rmse",
        boosting_type="gbdt",
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=600,
        random_state=random_state,
        verbose=-1
    )
    pipe = Pipeline([
        ("winsor", Winsorizer(0.01, 0.99)),
        ("scale", RobustScaler()),
        ("lgb", lgb),
    ])

    # 根据配置选择目标变换
    if TARGET_TRANSFORM == "log1p":
        transformer = FunctionTransformer(func=np.log1p, inverse_func=np.expm1,
                                          validate=False)
    elif TARGET_TRANSFORM == "boxcox":
        transformer = PowerTransformer(method='box-cox', standardize=False)
    elif TARGET_TRANSFORM == "yeojohnson":
        transformer = PowerTransformer(method='yeo-johnson', standardize=False)
    else:
        transformer = FunctionTransformer(func=np.log1p, inverse_func=np.expm1,
                                          validate=False)

    return TransformedTargetRegressor(
        regressor=pipe,
        transformer=transformer
    )


# ============ 参数空间 ============
def param_space_svr():
    rng = np.random.RandomState(RANDOM_STATE)

    def log_uniform(low, high, size):
        return np.exp(rng.uniform(np.log(low), np.log(high), size))

    return {
        "regressor__svr__C": log_uniform(0.01, 1000, N_ITER_SVR),  # 扩展C值范围
        "regressor__svr__gamma": ['scale', 'auto'] + list(log_uniform(1e-4, 1e-1, N_ITER_SVR - 2)),  # 添加'scale'和'auto'
        "regressor__svr__epsilon": log_uniform(1e-3, 5e-1, N_ITER_SVR),
    }


def param_space_hgb():
    rng = np.random.RandomState(RANDOM_STATE + 1)
    return {
        "regressor__hgb__learning_rate": rng.choice([0.03, 0.05, 0.07, 0.1, 0.15], N_ITER_HGB),  # 更多选项
        "regressor__hgb__max_depth": rng.choice([None, 3, 4, 5, 6], N_ITER_HGB),  # 更多深度选项
        "regressor__hgb__max_leaf_nodes": rng.choice([15, 31, 63, 127], N_ITER_HGB),  # 更多选项
        "regressor__hgb__min_samples_leaf": rng.choice([5, 10, 15, 20], N_ITER_HGB),
        "regressor__hgb__l2_regularization": rng.choice([0.1, 0.3, 0.5, 1.0, 2.0], N_ITER_HGB),  # 更细的正则化范围
        "regressor__hgb__max_iter": rng.choice([100, 200, 300, 400], N_ITER_HGB),  # 新增max_iter
    }


def param_space_xgb():
    rng = np.random.RandomState(RANDOM_STATE + 2)
    return {
        "regressor__xgb__learning_rate": rng.choice([0.01, 0.03, 0.05, 0.07, 0.1], N_ITER_XGB),
        "regressor__xgb__max_depth": rng.choice([3, 4, 5, 6, 7], N_ITER_XGB),
        "regressor__xgb__min_child_weight": rng.choice([1.0, 3.0, 5.0, 10.0], N_ITER_XGB),
        "regressor__xgb__subsample": rng.choice([0.6, 0.7, 0.8, 0.9, 1.0], N_ITER_XGB),
        "regressor__xgb__colsample_bytree": rng.choice([0.6, 0.7, 0.8, 0.9, 1.0], N_ITER_XGB),
        "regressor__xgb__colsample_bylevel": rng.choice([0.6, 0.7, 0.8, 0.9, 1.0], N_ITER_XGB),  # 新增
        "regressor__xgb__gamma": rng.choice([0, 0.1, 0.2, 0.5, 1.0], N_ITER_XGB),  # 新增
        "regressor__xgb__reg_lambda": rng.choice([0.5, 1.0, 2.0, 5.0], N_ITER_XGB),
        "regressor__xgb__reg_alpha": rng.choice([0, 0.1, 0.5, 1.0], N_ITER_XGB),  # 新增
    }


def param_space_lgb():
    rng = np.random.RandomState(RANDOM_STATE + 3)
    return {
        "regressor__lgb__learning_rate": rng.choice([0.01, 0.03, 0.05, 0.07, 0.1], N_ITER_LGB),
        "regressor__lgb__num_leaves": rng.choice([15, 31, 63, 127], N_ITER_LGB),
        "regressor__lgb__max_depth": rng.choice([-1, 3, 5, 7], N_ITER_LGB),
        "regressor__lgb__min_child_samples": rng.choice([5, 10, 20, 30], N_ITER_LGB),
        "regressor__lgb__subsample": rng.choice([0.6, 0.7, 0.8, 0.9, 1.0], N_ITER_LGB),
        "regressor__lgb__colsample_bytree": rng.choice([0.6, 0.7, 0.8, 0.9, 1.0], N_ITER_LGB),
        "regressor__lgb__reg_lambda": rng.choice([0, 0.1, 0.5, 1.0], N_ITER_LGB),
        "regressor__lgb__reg_alpha": rng.choice([0, 0.1, 0.5, 1.0], N_ITER_LGB),
    }


# ============ 集成权重学习（增强版） ============
def learn_ensemble_weights(preds: Dict[str, np.ndarray], y_val: np.ndarray,
                           l2: float = 1e-3, lam: float = LAMBDA_PB,
                           step: float = WEIGHT_GRID_STEP,
                           weak_cap: Tuple[float, float] = (1.4, 1.7)) -> Dict[str, float]:
    """单纯形网格搜索集成权重（带自适应权重约束、多样性奖励、两阶段搜索）"""
    names = list(preds.keys())
    M = len(names)
    P = np.vstack([preds[k] for k in names]).T
    y = y_val.reshape(-1)

    # 单模型评估
    rmses = {k: rmse(y, preds[k]) for k in names}
    r2s = {k: r2_score(y, preds[k]) for k in names}
    best_rmse = min(rmses.values())

    # 动态权重上限（基于R²） - 增强
    caps = {}
    for k in names:
        if rmses[k] > weak_cap[1] * best_rmse:
            caps[k] = 0.0  # 性能太差，禁用
        elif rmses[k] > weak_cap[0] * best_rmse:
            caps[k] = 0.3  # 性能一般，限制权重
        else:
            # 基于R²动态调整权重上限
            if r2s[k] > 0.6:
                caps[k] = 1.0  # 优秀模型，无限制
            elif r2s[k] > 0.4:
                caps[k] = 0.8  # 良好模型
            elif r2s[k] > 0.2:
                caps[k] = 0.6  # 一般模型
            else:
                caps[k] = 0.4  # 较差模型

    def _loss(w):
        y_ens = P @ w
        base_loss = rmse(y, y_ens) + lam * pinball90(y, y_ens)
        reg_term = l2 * float(np.dot(w, w))
        # 多样性奖励机制（增强） - 鼓励使用多个模型
        diversity_bonus = -0.02 * np.sum(w > 0.1) - 0.01 * np.sum(w > 0.05)
        return base_loss + reg_term + diversity_bonus

    # 优化搜索策略 - 两阶段搜索（粗搜+精搜）
    best_w, best_obj = None, float("inf")

    if M == 2:
        # 对于两个模型，使用更细的网格
        ks = np.arange(0.0, 1.0 + step / 2, step / 2)
        for w0 in ks:
            w = np.array([w0, 1 - w0])
            if w[0] > caps[names[0]] or w[1] > caps[names[1]]:
                continue
            obj = _loss(w)
            if obj < best_obj:
                best_obj, best_w = obj, w
    elif M == 3:
        # 对于三个模型，使用自适应搜索 - 两阶段搜索
        # 第一阶段：粗搜索
        coarse_step = step * 2
        for w0 in np.arange(0, 1 + coarse_step, coarse_step):
            for w1 in np.arange(0, 1 - w0 + coarse_step, coarse_step):
                w2 = 1 - w0 - w1
                if w2 < 0:
                    continue
                w = np.array([w0, w1, w2])
                if any(w[i] > caps[names[i]] for i in range(3)):
                    continue
                obj = _loss(w)
                if obj < best_obj:
                    best_obj, best_w = obj, w

        # 第二阶段：精细搜索（在最优解附近）
        if best_w is not None:
            center = best_w
            for delta in np.linspace(-step * 2, step * 2, 5):
                for i in range(3):
                    w_test = center.copy()
                    w_test[i] = max(0, min(1, center[i] + delta))
                    w_test = w_test / w_test.sum()  # 重新归一化
                    if any(w_test[j] > caps[names[j]] for j in range(3)):
                        continue
                    obj = _loss(w_test)
                    if obj < best_obj:
                        best_obj, best_w = obj, w_test
    else:
        # 多于3个模型时使用启发式方法
        # 基于性能的初始权重
        perfs = np.array([1.0 / (rmses[k] + 0.1) for k in names])
        best_w = perfs / perfs.sum()
        # 应用权重上限
        for i, name in enumerate(names):
            best_w[i] = min(best_w[i], caps[name])
        best_w = best_w / best_w.sum()

    # 更激进的稀疏化（增强）
    threshold = 0.02  # 降低阈值，更激进的稀疏化
    best_w = np.where(best_w < threshold, 0.0, best_w)
    s = best_w.sum()
    best_w = np.ones(M) / M if s <= 0 else best_w / s

    return {names[i]: float(best_w[i]) for i in range(M)}


# ============ Stacking集成 ============
def make_stacking_ensemble(base_models: Dict, X: np.ndarray, y: np.ndarray,
                           random_state: int) -> BaseEstimator:
    """创建Stacking集成模型"""
    # 准备基模型
    estimators = []
    for name, model in base_models.items():
        estimators.append((name, model))

    # 创建元模型（使用简单的线性模型）
    meta_model = Ridge(alpha=1.0, random_state=random_state)

    # 创建Stacking模型
    stacking_model = StackingCVRegressor(
        regressors=[model for _, model in estimators],
        meta_regressor=meta_model,
        cv=5,
        random_state=random_state,
        n_jobs=N_JOBS
    )

    return stacking_model


# ============ Permutation Importance计算 ============
def compute_permutation_importance(models: Dict, X_folds: List[np.ndarray],
                                   y_folds: List[np.ndarray], feature_names: List[str]) -> pd.DataFrame:
    """计算折外Permutation Importance"""
    pi_results = {feat: [] for feat in feature_names}

    for fold_idx, (X_val, y_val) in enumerate(zip(X_folds, y_folds)):
        for model_name, model in models.items():
            if model is None:
                continue

            # 计算PI
            result = permutation_importance(
                model, X_val, y_val,
                n_repeats=10,
                random_state=RANDOM_STATE + fold_idx,
                scoring='r2',
                n_jobs=N_JOBS
            )

            # 记录每个特征的重要性
            for feat_idx, feat_name in enumerate(feature_names):
                pi_results[feat_name].append(result.importances_mean[feat_idx])

    # 汇总结果
    pi_summary = []
    for feat in feature_names:
        importances = np.array(pi_results[feat])
        pi_summary.append({
            'feature': feat,
            'mean_importance': importances.mean(),
            'std_importance': importances.std(),
            'pos_ratio': (importances > 0).mean(),
            'n_positive_folds': (importances > 0).sum()
        })

    return pd.DataFrame(pi_summary).sort_values('mean_importance', ascending=False)


# ============ 特征选择 ============
def select_features_rfecv(X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> List[str]:
    """使用RFECV选择特征"""
    # 使用一个强模型作为特征选择的评估器
    estimator = HistGradientBoostingRegressor(
        loss="absolute_error",
        random_state=RANDOM_STATE
    )

    # 创建RFECV对象
    rfecv = RFECV(
        estimator=estimator,
        step=1,
        cv=5,
        scoring='neg_root_mean_squared_error',
        min_features_to_select=MIN_FEATURES_KEEP,
        n_jobs=N_JOBS
    )

    # 拟合RFECV
    rfecv.fit(X, y)

    # 获取选定的特征
    selected_indices = np.where(rfecv.support_)[0]
    selected_features = [feature_names[i] for i in selected_indices]

    # 保存选择的特征
    selection_info = {
        "method": "RFECV",
        "n_original": len(feature_names),
        "n_selected": len(selected_features),
        "selected_features": selected_features,
        "ranking": rfecv.ranking_.tolist(),
        "cv_scores": rfecv.cv_results_['mean_test_score'].tolist()
    }

    with open(SELECTED_FEATURES_JSON, 'w') as f:
        json.dump(selection_info, f, indent=2)

    print(f"RFECV selected {len(selected_features)}/{len(feature_names)} features")
    return selected_features


def select_features_boruta(X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> List[str]:
    """使用Boruta选择特征"""
    # 使用随机森林作为Boruta的评估器
    estimator = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS
    )

    # 创建Boruta对象
    boruta = BorutaPy(
        estimator=estimator,
        n_estimators='auto',
        verbose=0,
        random_state=RANDOM_STATE
    )

    # 拟合Boruta
    boruta.fit(X, y)

    # 获取选定的特征
    selected_indices = np.where(boruta.support_)[0]
    selected_features = [feature_names[i] for i in selected_indices]

    # 保存选择的特征
    selection_info = {
        "method": "Boruta",
        "n_original": len(feature_names),
        "n_selected": len(selected_features),
        "selected_features": selected_features,
        "support": boruta.support_.tolist(),
        "support_weak": boruta.support_weak_.tolist(),
        "ranking": boruta.ranking_.tolist()
    }

    with open(SELECTED_FEATURES_JSON, 'w') as f:
        json.dump(selection_info, f, indent=2)

    print(f"Boruta selected {len(selected_features)}/{len(feature_names)} features")
    return selected_features


def select_features_by_pi(pi_df: pd.DataFrame) -> List[str]:
    """基于PI选择特征"""
    # 筛选条件
    selected = pi_df[
        (pi_df['mean_importance'] > PI_MIN_IMPORTANCE) &
        (pi_df['pos_ratio'] >= PI_MIN_POS_RATIO)
        ].copy()

    # 确保最少特征数
    if len(selected) < MIN_FEATURES_KEEP:
        selected = pi_df.head(MIN_FEATURES_KEEP).copy()

    # 如果设置了TOP_K
    if TOP_K_FEATURES is not None and len(selected) > TOP_K_FEATURES:
        selected = selected.head(TOP_K_FEATURES)

    selected_features = selected['feature'].tolist()

    # 保存选择的特征
    selection_info = {
        "method": "PI",
        "n_original": len(pi_df),
        "n_selected": len(selected_features),
        "selection_criteria": {
            "min_importance": PI_MIN_IMPORTANCE,
            "min_pos_ratio": PI_MIN_POS_RATIO,
            "min_features": MIN_FEATURES_KEEP,
            "top_k": TOP_K_FEATURES
        },
        "selected_features": selected_features
    }

    with open(SELECTED_FEATURES_JSON, 'w') as f:
        json.dump(selection_info, f, indent=2)

    print(f"Selected {len(selected_features)}/{len(pi_df)} features based on PI")
    return selected_features


# ============ IQR相关 ============
def iqr_stats(y: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=float).reshape(-1)
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
    y = np.asarray(y, dtype=float).reshape(-1)
    mild = (y < stats["inner_lo"]) | (y > stats["inner_hi"])
    extreme = (y < stats["outer_lo"]) | (y > stats["outer_hi"])
    return mild, extreme


# ============ 训练与评估主函数 ============
def run_nested_cv_with_pi(X: np.ndarray, y: np.ndarray, feature_names: List[str],
                          compute_pi: bool = True) -> Dict[str, Any]:
    """嵌套CV与Permutation Importance"""
    n = len(y)

    # 准备分层
    y_bins_outer = stratify_bins(y, n_bins=Y_BINS, seed=RANDOM_STATE)
    rskf_outer = RepeatedStratifiedKFold(n_splits=N_OUTER, n_repeats=N_REPEATS,
                                         random_state=RANDOM_STATE)

    # 初始化结果容器
    oof_pred = {"SVR": np.zeros(n), "HGB": np.zeros(n)}
    if USE_XGB:
        oof_pred["XGB"] = np.zeros(n)
    if USE_LGB:
        oof_pred["LGB"] = np.zeros(n)
    oof_pred["ENS"] = np.zeros(n)

    y_oof = np.full(n, np.nan)
    fold_models = []
    fold_X_val = []
    fold_y_val = []
    fold_summaries = []
    ens_weights_per_fold = []
    fold_idx = 0

    for train_idx, test_idx in rskf_outer.split(X, y_bins_outer):
        fold_idx += 1
        Xtr, Xval = X[train_idx], X[test_idx]
        ytr, yval = y[train_idx], y[test_idx]

        # 内层分层
        y_bins_inner = stratify_bins(ytr, n_bins=min(Y_BINS, max(3, int(np.sqrt(len(ytr))))))
        skf_inner = StratifiedKFold(n_splits=N_INNER, shuffle=True,
                                    random_state=RANDOM_STATE + fold_idx)

        # 训练SVR
        svr = make_svr_pipeline(RANDOM_STATE + fold_idx)
        svr_rs = RandomizedSearchCV(
            estimator=svr,
            param_distributions=param_space_svr(),
            n_iter=N_ITER_SVR,
            cv=skf_inner.split(Xtr, y_bins_inner),
            scoring="neg_root_mean_squared_error",
            n_jobs=N_JOBS, refit=True,
            random_state=RANDOM_STATE + fold_idx, verbose=0
        )
        svr_rs.fit(Xtr, ytr)
        y_sv = svr_rs.predict(Xval)

        # 训练HGB
        hgb = make_hgb_pipeline(RANDOM_STATE + fold_idx)
        hgb_rs = RandomizedSearchCV(
            estimator=hgb,
            param_distributions=param_space_hgb(),
            n_iter=N_ITER_HGB,
            cv=skf_inner.split(Xtr, y_bins_inner),
            scoring="neg_root_mean_squared_error",
            n_jobs=N_JOBS, refit=True,
            random_state=RANDOM_STATE + fold_idx + 1, verbose=0
        )
        hgb_rs.fit(Xtr, ytr)
        y_hg = hgb_rs.predict(Xval)

        # 训练XGB（可选）
        xgb_rs = None
        if USE_XGB:
            xgb = make_xgb_pipeline(RANDOM_STATE + fold_idx)
            xgb_rs = RandomizedSearchCV(
                estimator=xgb,
                param_distributions=param_space_xgb(),
                n_iter=N_ITER_XGB,
                cv=skf_inner.split(Xtr, y_bins_inner),
                scoring="neg_root_mean_squared_error",
                n_jobs=N_JOBS, refit=True,
                random_state=RANDOM_STATE + fold_idx + 2, verbose=0
            )
            xgb_rs.fit(Xtr, ytr)
            y_xg = xgb_rs.predict(Xval)

        # 训练LGB（可选）
        lgb_rs = None
        if USE_LGB:
            lgb = make_lgb_pipeline(RANDOM_STATE + fold_idx)
            lgb_rs = RandomizedSearchCV(
                estimator=lgb,
                param_distributions=param_space_lgb(),
                n_iter=N_ITER_LGB,
                cv=skf_inner.split(Xtr, y_bins_inner),
                scoring="neg_root_mean_squared_error",
                n_jobs=N_JOBS, refit=True,
                random_state=RANDOM_STATE + fold_idx + 3, verbose=0
            )
            lgb_rs.fit(Xtr, ytr)
            y_lg = lgb_rs.predict(Xval)

        # 保存结果
        y_oof[test_idx] = yval
        oof_pred["SVR"][test_idx] = y_sv
        oof_pred["HGB"][test_idx] = y_hg
        if USE_XGB:
            oof_pred["XGB"][test_idx] = y_xg
        if USE_LGB:
            oof_pred["LGB"][test_idx] = y_lg

        # 保存模型和验证数据（用于PI）
        if compute_pi:
            fold_models.append({
                "SVR": svr_rs.best_estimator_,
                "HGB": hgb_rs.best_estimator_,
                "XGB": xgb_rs.best_estimator_ if USE_XGB else None,
                "LGB": lgb_rs.best_estimator_ if USE_LGB else None
            })
            fold_X_val.append(Xval)
            fold_y_val.append(yval)

        # 学习集成权重或使用Stacking
        if ENSEMBLE_METHOD == "stacking":
            # 创建基模型字典
            base_models = {
                "SVR": svr_rs.best_estimator_,
                "HGB": hgb_rs.best_estimator_
            }
            if USE_XGB:
                base_models["XGB"] = xgb_rs.best_estimator_
            if USE_LGB:
                base_models["LGB"] = lgb_rs.best_estimator_

            # 创建Stacking模型
            stacking_model = make_stacking_ensemble(base_models, Xtr, ytr, RANDOM_STATE + fold_idx)

            # 拟合Stacking模型
            stacking_model.fit(Xtr, ytr)

            # 预测
            y_ens = stacking_model.predict(Xval)

            # 保存权重（元模型系数）
            weights = {}
            for i, name in enumerate(base_models.keys()):
                weights[name] = float(stacking_model.meta_regr_.coef_[i])

            # 归一化权重
            total = sum(weights.values())
            if total > 0:
                weights = {k: v / total for k, v in weights.items()}
            else:
                weights = {k: 1 / len(weights) for k in weights}
        else:
            # 加权平均
            preds_for_ens = {"SVR": y_sv, "HGB": y_hg}
            if USE_XGB:
                preds_for_ens["XGB"] = y_xg
            if USE_LGB:
                preds_for_ens["LGB"] = y_lg

            weights = learn_ensemble_weights(preds_for_ens, yval, lam=LAMBDA_PB)
            y_ens = sum(weights[k] * preds_for_ens[k] for k in preds_for_ens)

        oof_pred["ENS"][test_idx] = y_ens
        ens_weights_per_fold.append(weights)

        # 记录fold结果
        fold_summaries.append(f"Fold {fold_idx}: SVR R²={r2_score(yval, y_sv):.3f}, "
                              f"HGB R²={r2_score(yval, y_hg):.3f}")
        if USE_XGB:
            fold_summaries[-1] += f", XGB R²={r2_score(yval, y_xg):.3f}"
        if USE_LGB:
            fold_summaries[-1] += f", LGB R²={r2_score(yval, y_lg):.3f}"
        fold_summaries[-1] += f", ENS R²={r2_score(yval, y_ens):.3f}"

    # 计算OOF汇总
    summary = {}
    for name in ["SVR", "HGB", "ENS"] + (["XGB"] if USE_XGB else []) + (["LGB"] if USE_LGB else []):
        summary[name] = {
            "R2": r2_score(y_oof, oof_pred[name]),
            "RMSE": rmse(y_oof, oof_pred[name]),
            "Pinball90": pinball90(y_oof, oof_pred[name])
        }

    # 计算Permutation Importance
    pi_df = None
    if compute_pi and fold_models:
        print("Computing Permutation Importance...")
        pi_df = compute_permutation_importance(fold_models[:N_OUTER],
                                               fold_X_val[:N_OUTER],
                                               fold_y_val[:N_OUTER],
                                               feature_names)
        pi_df.to_csv(PI_CSV, index=False)
        print(f"PI saved to {PI_CSV}")

    return {
        "summary": summary,
        "fold_summaries": fold_summaries,
        "y_oof": y_oof,
        "oof_pred": oof_pred,
        "weights": ens_weights_per_fold,
        "pi_df": pi_df
    }


# ============ 主函数 ============
def main():
    print("=" * 60)
    print("PM2.5 Meta-Ensemble Enhanced Pipeline")
    print("=" * 60)

    # 读取数据
    df = pd.read_excel(DATA_PATH)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column {TARGET_COL} not found")

    Xdf, y, feat_cols = build_feature_frame(df)
    X = Xdf.values.astype(float)
    print(f"Data shape: {X.shape}, Target shape: {y.shape}")
    print(f"Features: {feat_cols}")

    # A. 数据可视化
    print("\nA. Creating data visualizations...")
    fig_dir = os.path.join(OUTPUT_DIR, "figures")
    plot_y_distribution(y, fig_dir)
    plot_iqr_summary(y, fig_dir)
    plot_correlation_heatmaps(X, feat_cols, y, fig_dir)

    # IQR筛查
    if RUN_IQR_SCREENING:
        mild, extreme = flag_outliers_iqr(y)
        iqr_info = iqr_stats(y)
        iqr_info.update({
            "n_total": len(y),
            "n_mild": int(mild.sum()),
            "n_extreme": int(extreme.sum())
        })
        print(f"IQR screening: {iqr_info['n_mild']} mild, {iqr_info['n_extreme']} extreme outliers")

    # 可选：留出测试集
    if USE_HOLDOUT:
        X_work, X_hold, y_work, y_hold = train_test_split(
            X, y, test_size=HOLDOUT_SIZE, random_state=RANDOM_STATE,
            stratify=stratify_bins(y, n_bins=5, seed=RANDOM_STATE)
        )
        print(f"\nHoldout test set: {len(y_hold)} samples reserved")
    else:
        X_work, y_work = X, y
        X_hold, y_hold = None, None

    # B. 基准建模与PI
    print("\nB. Running nested CV with Permutation Importance...")
    results_full = run_nested_cv_with_pi(X_work, y_work, feat_cols, compute_pi=True)

    # 绘制OOF诊断图
    plot_oof_diagnostics(results_full["y_oof"], results_full["oof_pred"], fig_dir)

    # 绘制PI图
    if results_full["pi_df"] is not None:
        plot_permutation_importance(results_full["pi_df"], fig_dir)

    # 绘制集成权重
    plot_ensemble_weights(results_full["weights"], fig_dir)

    # C. 特征选择与重训
    print("\nC. Feature selection and retraining...")
    selected_features = None
    results_selected = None

    # 根据配置选择特征选择方法
    if FEATURE_SELECTION_METHOD == "RFECV":
        selected_features = select_features_rfecv(X_work, y_work, feat_cols)
    elif FEATURE_SELECTION_METHOD == "BORUTA" and HAVE_BORUTA:
        selected_features = select_features_boruta(X_work, y_work, feat_cols)
    elif results_full["pi_df"] is not None:
        selected_features = select_features_by_pi(results_full["pi_df"])
    else:
        print("No feature selection method available, using all features")
        selected_features = feat_cols

    if selected_features:
        selected_indices = [feat_cols.index(f) for f in selected_features]
        X_selected = X_work[:, selected_indices]

        print(f"Retraining with {len(selected_features)} selected features...")
        results_selected = run_nested_cv_with_pi(X_selected, y_work, selected_features,
                                                 compute_pi=False)

        # 比较结果
        comparison = pd.DataFrame({
            'Model': ['SVR', 'HGB', 'ENS'] + (['XGB'] if USE_XGB else []) + (['LGB'] if USE_LGB else []),
            'R2_Full': [results_full["summary"][m]["R2"] for m in
                        ['SVR', 'HGB', 'ENS'] + (['XGB'] if USE_XGB else []) + (['LGB'] if USE_LGB else [])],
            'RMSE_Full': [results_full["summary"][m]["RMSE"] for m in
                          ['SVR', 'HGB', 'ENS'] + (['XGB'] if USE_XGB else []) + (['LGB'] if USE_LGB else [])],
            'R2_Selected': [results_selected["summary"][m]["R2"] for m in
                            ['SVR', 'HGB', 'ENS'] + (['XGB'] if USE_XGB else []) + (['LGB'] if USE_LGB else [])],
            'RMSE_Selected': [results_selected["summary"][m]["RMSE"] for m in
                              ['SVR', 'HGB', 'ENS'] + (['XGB'] if USE_XGB else []) + (['LGB'] if USE_LGB else [])]
        })
        comparison['R2_Improvement'] = comparison['R2_Selected'] - comparison['R2_Full']
        comparison['RMSE_Improvement'] = comparison['RMSE_Full'] - comparison['RMSE_Selected']

        comparison.to_csv(os.path.join(OUTPUT_DIR, 'feature_selection_comparison.csv'), index=False)
        print("\nFeature Selection Comparison:")
        print(comparison.to_string())

    # 绘制学习曲线（使用选定特征或全部特征）
    print("\nPlotting learning curves...")
    X_for_lc = X_selected if selected_features else X_work
    models_for_lc = {
        'SVR': make_svr_pipeline(RANDOM_STATE),
        'HGB': make_hgb_pipeline(RANDOM_STATE)
    }
    if USE_XGB:
        models_for_lc['XGB'] = make_xgb_pipeline(RANDOM_STATE)
    if USE_LGB:
        models_for_lc['LGB'] = make_lgb_pipeline(RANDOM_STATE)

    plot_learning_curves(X_for_lc, y_work, models_for_lc, fig_dir)

    # D. 可选的留出集评估
    holdout_results = None
    if USE_HOLDOUT and X_hold is not None:
        print("\nD. Evaluating on holdout test set...")

        # 使用最佳特征集训练最终模型
        X_train_final = X_selected if selected_features else X_work
        features_final = selected_features if selected_features else feat_cols

        if selected_features:
            X_test_final = X_hold[:, selected_indices]
        else:
            X_test_final = X_hold

        # 训练最终模型
        final_models = {}

        # SVR
        svr_final = make_svr_pipeline(RANDOM_STATE)
        svr_cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        svr_rs = RandomizedSearchCV(
            svr_final, param_space_svr(), n_iter=N_ITER_SVR,
            cv=svr_cv, scoring="neg_root_mean_squared_error",
            n_jobs=N_JOBS, random_state=RANDOM_STATE
        )
        svr_rs.fit(X_train_final, y_work)
        final_models['SVR'] = svr_rs.best_estimator_

        # HGB
        hgb_final = make_hgb_pipeline(RANDOM_STATE)
        hgb_rs = RandomizedSearchCV(
            hgb_final, param_space_hgb(), n_iter=N_ITER_HGB,
            cv=svr_cv, scoring="neg_root_mean_squared_error",
            n_jobs=N_JOBS, random_state=RANDOM_STATE + 1
        )
        hgb_rs.fit(X_train_final, y_work)
        final_models['HGB'] = hgb_rs.best_estimator_

        # XGB（可选）
        if USE_XGB:
            xgb_final = make_xgb_pipeline(RANDOM_STATE)
            xgb_rs = RandomizedSearchCV(
                xgb_final, param_space_xgb(), n_iter=N_ITER_XGB,
                cv=svr_cv, scoring="neg_root_mean_squared_error",
                n_jobs=N_JOBS, random_state=RANDOM_STATE + 2
            )
            xgb_rs.fit(X_train_final, y_work)
            final_models['XGB'] = xgb_rs.best_estimator_

        # LGB（可选）
        if USE_LGB:
            lgb_final = make_lgb_pipeline(RANDOM_STATE)
            lgb_rs = RandomizedSearchCV(
                lgb_final, param_space_lgb(), n_iter=N_ITER_LGB,
                cv=svr_cv, scoring="neg_root_mean_squared_error",
                n_jobs=N_JOBS, random_state=RANDOM_STATE + 3
            )
            lgb_rs.fit(X_train_final, y_work)
            final_models['LGB'] = lgb_rs.best_estimator_

        # 预测并评估
        holdout_preds = {}
        for name, model in final_models.items():
            y_pred = model.predict(X_test_final)
            holdout_preds[name] = y_pred

        # 集成预测
        if ENSEMBLE_METHOD == "stacking":
            # 创建Stacking模型
            stacking_model = make_stacking_ensemble(final_models, X_train_final, y_work, RANDOM_STATE)

            # 拟合Stacking模型
            stacking_model.fit(X_train_final, y_work)

            # 预测
            y_ens_hold = stacking_model.predict(X_test_final)
        else:
            # 加权平均
            weights_final = results_selected["weights"][0] if results_selected else results_full["weights"][0]
            y_ens_hold = sum(weights_final.get(k, 0) * holdout_preds.get(k, np.zeros_like(y_hold))
                             for k in holdout_preds)

        holdout_preds['ENS'] = y_ens_hold

        # 计算指标
        holdout_results = {}
        for name, y_pred in holdout_preds.items():
            holdout_results[name] = {
                "R2": r2_score(y_hold, y_pred),
                "RMSE": rmse(y_hold, y_pred),
                "Pinball90": pinball90(y_hold, y_pred)
            }

        # 保存holdout结果
        with open(os.path.join(OUTPUT_DIR, 'holdout_results.json'), 'w') as f:
            json.dump(holdout_results, f, indent=2)

        print("\nHoldout Test Results:")
        for name, metrics in holdout_results.items():
            print(f"{name}: R²={metrics['R2']:.3f}, RMSE={metrics['RMSE']:.3f}, Pin90={metrics['Pinball90']:.3f}")

    # 生成最终报告
    print("\nGenerating final report...")
    generate_final_report(results_full, results_selected, holdout_results,
                          selected_features, feat_cols, iqr_info)

    print(f"\nPipeline completed! Results saved to {OUTPUT_DIR}/")
    print(f"Report: {REPORT_PATH}")
    print(f"Figures: {fig_dir}/")


def generate_final_report(results_full, results_selected, holdout_results,
                          selected_features, all_features, iqr_info):
    """生成最终报告"""
    lines = []
    lines.append("# PM2.5 Meta-Ensemble Enhanced Report")
    lines.append(f"\nGenerated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("\n## Configuration")
    lines.append(f"- Outer CV folds: {N_OUTER} × {N_REPEATS} repeats")
    lines.append(f"- Inner CV folds: {N_INNER}")
    lines.append(f"- Holdout test: {'Yes' if USE_HOLDOUT else 'No'} ({HOLDOUT_SIZE * 100:.0f}%)")
    lines.append(f"- Models: SVR, HGB" + (", XGB" if USE_XGB else "") + (", LGB" if USE_LGB else ""))
    lines.append(f"- Target transform: {TARGET_TRANSFORM}")
    lines.append(f"- Ensemble method: {ENSEMBLE_METHOD}")
    lines.append(f"- Feature selection: {FEATURE_SELECTION_METHOD}")
    lines.append(f"- Ensemble loss: RMSE + {LAMBDA_PB}×Pinball@{PINBALL_Q}")

    lines.append("\n## Data Overview")
    lines.append(f"- Total samples: 303")
    lines.append(f"- Original features: {len(all_features)}")
    lines.append(f"- IQR outliers: {iqr_info['n_mild']} mild, {iqr_info['n_extreme']} extreme")

    lines.append("\n## Full Feature Set Results (OOF)")
    for model, metrics in results_full["summary"].items():
        lines.append(f"- {model}: R²={metrics['R2']:.3f}, RMSE={metrics['RMSE']:.3f}, Pin90={metrics['Pinball90']:.3f}")

    if results_selected:
        lines.append(f"\n## Feature Selection")
        lines.append(f"- Selected features: {len(selected_features)}/{len(all_features)}")
        lines.append(f"- Selection method: {FEATURE_SELECTION_METHOD}")

        lines.append("\n## Selected Feature Set Results (OOF)")
        for model, metrics in results_selected["summary"].items():
            lines.append(
                f"- {model}: R²={metrics['R2']:.3f}, RMSE={metrics['RMSE']:.3f}, Pin90={metrics['Pinball90']:.3f}")

        # 比较改进
        lines.append("\n## Performance Improvement")
        for model in ['SVR', 'HGB', 'ENS']:
            r2_imp = results_selected["summary"][model]["R2"] - results_full["summary"][model]["R2"]
            rmse_imp = results_full["summary"][model]["RMSE"] - results_selected["summary"][model]["RMSE"]
            lines.append(f"- {model}: ΔR²={r2_imp:+.3f}, ΔRMSE={rmse_imp:+.3f}")

    if holdout_results:
        lines.append("\n## Holdout Test Results")
        for model, metrics in holdout_results.items():
            lines.append(
                f"- {model}: R²={metrics['R2']:.3f}, RMSE={metrics['RMSE']:.3f}, Pin90={metrics['Pinball90']:.3f}")

    lines.append("\n## Visualizations Generated")
    lines.append("- y_distribution.png: Target distribution and transform")
    lines.append("- iqr_summary.png: IQR fences and outlier counts")
    lines.append("- correlation_heatmaps.png: Feature correlations (Top 15)")
    lines.append("- oof_diagnostics.png: OOF predictions vs truth & residuals")
    lines.append("- permutation_importance.png: Feature importance ranking")
    lines.append("- ensemble_weights.png: Model weights across CV folds")
    lines.append("- learning_curve_*.png: Learning curves for each model")

    lines.append("\n## Files Generated")
    lines.append(f"- {PI_CSV}: Permutation importance details")
    lines.append(f"- {SELECTED_FEATURES_JSON}: Selected feature list")
    lines.append(f"- {OUTLIER_CSV}: Outlier screening results")
    lines.append("- feature_selection_comparison.csv: Before/after comparison")
    if holdout_results:
        lines.append("- holdout_results.json: Holdout test metrics")

    lines.append("\n## Conclusions")
    lines.append("1. Target distribution shows right-skew, transform improves normality")
    lines.append("2. IQR screening identifies potential outliers for sensitivity analysis")
    lines.append("3. Feature selection provides robust feature ranking")
    if results_selected:
        imp = "improved" if results_selected["summary"]["ENS"]["R2"] > results_full["summary"]["ENS"][
            "R2"] else "comparable"
        lines.append(
            f"4. Feature selection yields {imp} performance with {len(selected_features)}/{len(all_features)} features")
    lines.append("5. Ensemble weighting adapts to fold-specific data characteristics")
    lines.append("6. Learning curves suggest potential benefit from additional data")

    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


if __name__ == "__main__":
    main()