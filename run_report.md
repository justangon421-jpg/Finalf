# PM2.5 Meta-Ensemble Run Report

- Timestamp: 2025-09-08 00:17:55
- Rows: 303 | Features: 23 | Target length: 303
- Feature columns used (23): ['Initial_Temperature', 'Final_Temperature', 'Initial_Speed', 'Final_Speed', 'Deceleration_Rate', 'Contact_Pressure', 'Braking_Time', 'Temp_Rise', 'Temp_Rise_Rate', 'Mean_Temp', 'Speed_Drop_kmh', 'Speed_Drop_ms', 'Mean_Speed_ms', 'Braking_Distance', 'Decel_ms2_src', 'Decel_ms2_kin', 'SpeedXDecel', 'Energy_Rel', 'Power_Rel', 'Pressure_Time', 'V1_over_V0', 'Tf_over_Ti', 'PT_over_Erel']
- Target column: PM2_5
- Outer folds: 5, Inner folds: 3, y-bins: 7, repeats: 2
- Target transform: log1p/expm1 via TransformedTargetRegressor (sklearn)。
- Ensemble loss: RMSE + 0.20 * Pinball@0.90

## IQR screening (Tukey fences)
- Q1 = 0.051066, Q3 = 0.952853, IQR = 0.901787
- Inner fences: [-1.301615, 2.305533] (k=1.5)
- Outer fences: [-2.654295, 3.658214] (k=3.0)
- Mild outliers: 35 / 303
- Extreme outliers: 24 / 303
- Screening list saved: outlier_screening.csv

## Per-fold results (All data; screening only, no deletion)
- Fold 1 | SVR | R2=0.680 | RMSE=1.468 | Pin90=0.411 | best={'regressor__svr__gamma': 0.0006963114377829289, 'regressor__svr__epsilon': 0.28151055958334764, 'regressor__svr__C': 23.423849847112887}
- Fold 1 | HGB | R2=0.155 | RMSE=2.386 | Pin90=0.569 | best={'regressor__hgb__min_samples_leaf': 10, 'regressor__hgb__max_leaf_nodes': 31, 'regressor__hgb__max_depth': 4, 'regressor__hgb__learning_rate': 0.1, 'regressor__hgb__l2_regularization': 0.3}
- Fold 1 | XGB | R2=0.041 | RMSE=2.541 | Pin90=0.606 | best={'regressor__xgb__subsample': 0.8, 'regressor__xgb__reg_lambda': 2.0, 'regressor__xgb__min_child_weight': 5.0, 'regressor__xgb__max_depth': 3, 'regressor__xgb__learning_rate': 0.05, 'regressor__xgb__colsample_bytree': 1.0}
- __DIAG_FOLD_1__ | ENS | Weights={'SVR': 1.0, 'HGB': 0.0, 'XGB': 0.0}
- Fold 2 | SVR | R2=0.572 | RMSE=1.044 | Pin90=0.249 | best={'regressor__svr__gamma': 0.001465655388622534, 'regressor__svr__epsilon': 0.28151055958334764, 'regressor__svr__C': 84.71801418819973}
- Fold 2 | HGB | R2=0.814 | RMSE=0.688 | Pin90=0.268 | best={'regressor__hgb__min_samples_leaf': 10, 'regressor__hgb__max_leaf_nodes': 31, 'regressor__hgb__max_depth': None, 'regressor__hgb__learning_rate': 0.1, 'regressor__hgb__l2_regularization': 0.5}
- Fold 2 | XGB | R2=0.010 | RMSE=1.587 | Pin90=0.258 | best={'regressor__xgb__subsample': 1.0, 'regressor__xgb__reg_lambda': 0.5, 'regressor__xgb__min_child_weight': 1.0, 'regressor__xgb__max_depth': 5, 'regressor__xgb__learning_rate': 0.1, 'regressor__xgb__colsample_bytree': 1.0}
- __DIAG_FOLD_2__ | ENS | Weights={'SVR': 0.26, 'HGB': 0.74, 'XGB': 0.0}
- Fold 3 | SVR | R2=0.357 | RMSE=1.794 | Pin90=0.551 | best={'regressor__svr__gamma': 0.00022264204303769702, 'regressor__svr__epsilon': 0.22492927989859166, 'regressor__svr__C': 54.56725485601472}
- Fold 3 | HGB | R2=0.450 | RMSE=1.660 | Pin90=0.541 | best={'regressor__hgb__min_samples_leaf': 10, 'regressor__hgb__max_leaf_nodes': 63, 'regressor__hgb__max_depth': None, 'regressor__hgb__learning_rate': 0.1, 'regressor__hgb__l2_regularization': 0.3}
- Fold 3 | XGB | R2=0.495 | RMSE=1.589 | Pin90=0.482 | best={'regressor__xgb__subsample': 0.8, 'regressor__xgb__reg_lambda': 1.0, 'regressor__xgb__min_child_weight': 1.0, 'regressor__xgb__max_depth': 5, 'regressor__xgb__learning_rate': 0.05, 'regressor__xgb__colsample_bytree': 1.0}
- __DIAG_FOLD_3__ | ENS | Weights={'SVR': 0.0, 'HGB': 0.0, 'XGB': 1.0}
- Fold 4 | SVR | R2=0.562 | RMSE=1.753 | Pin90=0.481 | best={'regressor__svr__gamma': 0.00022264204303769702, 'regressor__svr__epsilon': 0.012811830949395147, 'regressor__svr__C': 23.423849847112887}
- Fold 4 | HGB | R2=0.379 | RMSE=2.088 | Pin90=0.540 | best={'regressor__hgb__min_samples_leaf': 10, 'regressor__hgb__max_leaf_nodes': 31, 'regressor__hgb__max_depth': None, 'regressor__hgb__learning_rate': 0.05, 'regressor__hgb__l2_regularization': 0.5}
- Fold 4 | XGB | R2=0.398 | RMSE=2.054 | Pin90=0.540 | best={'regressor__xgb__subsample': 1.0, 'regressor__xgb__reg_lambda': 5.0, 'regressor__xgb__min_child_weight': 10.0, 'regressor__xgb__max_depth': 5, 'regressor__xgb__learning_rate': 0.05, 'regressor__xgb__colsample_bytree': 1.0}
- __DIAG_FOLD_4__ | ENS | Weights={'SVR': 1.0, 'HGB': 0.0, 'XGB': 0.0}
- Fold 5 | SVR | R2=0.528 | RMSE=4.524 | Pin90=0.712 | best={'regressor__svr__gamma': 0.00035856126103454, 'regressor__svr__epsilon': 0.21354541789291287, 'regressor__svr__C': 67.96578090758145}
- Fold 5 | HGB | R2=-0.038 | RMSE=6.712 | Pin90=1.320 | best={'regressor__hgb__min_samples_leaf': 15, 'regressor__hgb__max_leaf_nodes': 63, 'regressor__hgb__max_depth': 4, 'regressor__hgb__learning_rate': 0.1, 'regressor__hgb__l2_regularization': 0.5}
- Fold 5 | XGB | R2=-0.033 | RMSE=6.695 | Pin90=1.298 | best={'regressor__xgb__subsample': 0.8, 'regressor__xgb__reg_lambda': 5.0, 'regressor__xgb__min_child_weight': 1.0, 'regressor__xgb__max_depth': 3, 'regressor__xgb__learning_rate': 0.1, 'regressor__xgb__colsample_bytree': 0.7}
- __DIAG_FOLD_5__ | ENS | Weights={'SVR': 0.7000000000000001, 'HGB': 0.0, 'XGB': 0.29999999999999993}
- Fold 6 | SVR | R2=0.398 | RMSE=1.290 | Pin90=0.348 | best={'regressor__svr__gamma': 0.0006963114377829289, 'regressor__svr__epsilon': 0.0021027192106506244, 'regressor__svr__C': 757.9479953347999}
- Fold 6 | HGB | R2=0.562 | RMSE=1.100 | Pin90=0.395 | best={'regressor__hgb__min_samples_leaf': 10, 'regressor__hgb__max_leaf_nodes': 31, 'regressor__hgb__max_depth': None, 'regressor__hgb__learning_rate': 0.05, 'regressor__hgb__l2_regularization': 0.3}
- Fold 6 | XGB | R2=0.341 | RMSE=1.350 | Pin90=0.366 | best={'regressor__xgb__subsample': 0.7, 'regressor__xgb__reg_lambda': 1.0, 'regressor__xgb__min_child_weight': 10.0, 'regressor__xgb__max_depth': 3, 'regressor__xgb__learning_rate': 0.1, 'regressor__xgb__colsample_bytree': 1.0}
- __DIAG_FOLD_6__ | ENS | Weights={'SVR': 0.38, 'HGB': 0.62, 'XGB': 0.0}
- Fold 7 | SVR | R2=0.690 | RMSE=1.400 | Pin90=0.328 | best={'regressor__svr__gamma': 0.001465655388622534, 'regressor__svr__epsilon': 0.10944851153312041, 'regressor__svr__C': 12.561043700013547}
- Fold 7 | HGB | R2=0.491 | RMSE=1.794 | Pin90=0.418 | best={'regressor__hgb__min_samples_leaf': 10, 'regressor__hgb__max_leaf_nodes': 63, 'regressor__hgb__max_depth': 4, 'regressor__hgb__learning_rate': 0.1, 'regressor__hgb__l2_regularization': 0.3}
- Fold 7 | XGB | R2=0.572 | RMSE=1.645 | Pin90=0.348 | best={'regressor__xgb__subsample': 1.0, 'regressor__xgb__reg_lambda': 0.5, 'regressor__xgb__min_child_weight': 1.0, 'regressor__xgb__max_depth': 6, 'regressor__xgb__learning_rate': 0.03, 'regressor__xgb__colsample_bytree': 1.0}
- __DIAG_FOLD_7__ | ENS | Weights={'SVR': 1.0, 'HGB': 0.0, 'XGB': 0.0}
- Fold 8 | SVR | R2=-0.010 | RMSE=3.967 | Pin90=0.829 | best={'regressor__svr__gamma': 0.0011895896737553553, 'regressor__svr__epsilon': 0.052571272727868656, 'regressor__svr__C': 24.81040974867808}
- Fold 8 | HGB | R2=-0.011 | RMSE=3.969 | Pin90=0.810 | best={'regressor__hgb__min_samples_leaf': 10, 'regressor__hgb__max_leaf_nodes': 31, 'regressor__hgb__max_depth': 3, 'regressor__hgb__learning_rate': 0.07, 'regressor__hgb__l2_regularization': 1.0}
- Fold 8 | XGB | R2=-0.021 | RMSE=3.988 | Pin90=0.759 | best={'regressor__xgb__subsample': 0.8, 'regressor__xgb__reg_lambda': 1.0, 'regressor__xgb__min_child_weight': 10.0, 'regressor__xgb__max_depth': 6, 'regressor__xgb__learning_rate': 0.1, 'regressor__xgb__colsample_bytree': 1.0}
- __DIAG_FOLD_8__ | ENS | Weights={'SVR': 0.26, 'HGB': 0.28, 'XGB': 0.45999999999999996}
- Fold 9 | SVR | R2=0.498 | RMSE=1.056 | Pin90=0.313 | best={'regressor__svr__gamma': 0.0001667761543019791, 'regressor__svr__epsilon': 0.21354541789291287, 'regressor__svr__C': 25.37815508265663}
- Fold 9 | HGB | R2=0.404 | RMSE=1.151 | Pin90=0.343 | best={'regressor__hgb__min_samples_leaf': 10, 'regressor__hgb__max_leaf_nodes': 63, 'regressor__hgb__max_depth': 4, 'regressor__hgb__learning_rate': 0.1, 'regressor__hgb__l2_regularization': 0.3}
- Fold 9 | XGB | R2=-0.167 | RMSE=1.611 | Pin90=0.295 | best={'regressor__xgb__subsample': 0.7, 'regressor__xgb__reg_lambda': 0.5, 'regressor__xgb__min_child_weight': 10.0, 'regressor__xgb__max_depth': 5, 'regressor__xgb__learning_rate': 0.03, 'regressor__xgb__colsample_bytree': 1.0}
- __DIAG_FOLD_9__ | ENS | Weights={'SVR': 0.76, 'HGB': 0.24, 'XGB': 0.0}
- Fold 10 | SVR | R2=0.931 | RMSE=1.613 | Pin90=0.438 | best={'regressor__svr__gamma': 0.0011756010900231862, 'regressor__svr__epsilon': 0.14760016020610808, 'regressor__svr__C': 138.26232179369853}
- Fold 10 | HGB | R2=-0.011 | RMSE=6.155 | Pin90=1.417 | best={'regressor__hgb__min_samples_leaf': 10, 'regressor__hgb__max_leaf_nodes': 31, 'regressor__hgb__max_depth': 4, 'regressor__hgb__learning_rate': 0.1, 'regressor__hgb__l2_regularization': 1.0}
- Fold 10 | XGB | R2=0.013 | RMSE=6.079 | Pin90=1.353 | best={'regressor__xgb__subsample': 0.8, 'regressor__xgb__reg_lambda': 0.5, 'regressor__xgb__min_child_weight': 3.0, 'regressor__xgb__max_depth': 6, 'regressor__xgb__learning_rate': 0.03, 'regressor__xgb__colsample_bytree': 0.7}
- __DIAG_FOLD_10__ | ENS | Weights={'SVR': 1.0, 'HGB': 0.0, 'XGB': 0.0}

## OOF summary (All data)
- SVR |   SVR | R^2 = 0.644 | RMSE = 2.153 | Pinball@0.90 = 0.452
- HGB |   HGB | R^2 = 0.091 | RMSE = 3.439 | Pinball@0.90 = 0.675
- ENS |   ENS | R^2 = 0.657 | RMSE = 2.111 | Pinball@0.90 = 0.439
- XGB |   XGB | R^2 = 0.082 | RMSE = 3.456 | Pinball@0.90 = 0.623

## Per-fold ensemble weights (All data)
- Fold 1: {'SVR': 1.0, 'HGB': 0.0, 'XGB': 0.0}
- Fold 2: {'SVR': 0.26, 'HGB': 0.74, 'XGB': 0.0}
- Fold 3: {'SVR': 0.0, 'HGB': 0.0, 'XGB': 1.0}
- Fold 4: {'SVR': 1.0, 'HGB': 0.0, 'XGB': 0.0}
- Fold 5: {'SVR': 0.7000000000000001, 'HGB': 0.0, 'XGB': 0.29999999999999993}
- Fold 6: {'SVR': 0.38, 'HGB': 0.62, 'XGB': 0.0}
- Fold 7: {'SVR': 1.0, 'HGB': 0.0, 'XGB': 0.0}
- Fold 8: {'SVR': 0.26, 'HGB': 0.28, 'XGB': 0.45999999999999996}
- Fold 9: {'SVR': 0.76, 'HGB': 0.24, 'XGB': 0.0}
- Fold 10: {'SVR': 1.0, 'HGB': 0.0, 'XGB': 0.0}

## Target distribution (y) summary
count    303.000000
mean       1.255358
std        3.612642
min        0.000574
25%        0.051066
50%        0.273292
75%        0.952853
max       42.486264
dtype: float64

## Post-selection OOF summary (selected features)
- SVR |   SVR | R^2 = 0.319 | RMSE = 2.977 | Pinball@0.90 = 0.588
- HGB |   HGB | R^2 = 0.099 | RMSE = 3.424 | Pinball@0.90 = 0.674
- ENS |   ENS | R^2 = 0.333 | RMSE = 2.946 | Pinball@0.90 = 0.568
- XGB |   XGB | R^2 = 0.095 | RMSE = 3.432 | Pinball@0.90 = 0.622

> 说明：回归任务原生不支持 StratifiedKFold；本脚本先对 y 分箱再分层（近似分层），评估更稳。
> 预处理采用 Winsorizer + RobustScaler（均在训练折拟合，避免泄露），目标使用 log1p/expm1 变换。
> IQR 用于异常识别的筛查与留痕；敏感性分析为预先声明的对照，主结论以全数据+稳健方法为准。

## IQR sensitivity (trim & re-run)
- Trim strategy: inner (k=1.5)
- Removed rows: 35 / 303

### OOF summary (IQR-trimmed)
- SVR |   SVR | R^2 = 0.410 | RMSE = 0.427 | Pinball@0.90 = 0.144
- HGB |   HGB | R^2 = 0.390 | RMSE = 0.434 | Pinball@0.90 = 0.168
- ENS |   ENS | R^2 = 0.501 | RMSE = 0.392 | Pinball@0.90 = 0.143
- XGB |   XGB | R^2 = 0.334 | RMSE = 0.453 | Pinball@0.90 = 0.157
