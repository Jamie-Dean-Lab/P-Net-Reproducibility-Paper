"""
Models compared for the radiosensitivity prediction task.

Single source of truth for which runs are compared, what they are called in figures,
and the order they appear in. Imported by the nested-CV and external-validation plots
and by significance_testing, so a display name cannot disagree between a figure and
the significance table.

Note: "dense" is the fully-connected P-NET (P-NET-FC), not a dense single layer.
"""

from architecture.plotting.model_comparison import ModelRegistry

# The model every other one is compared against.
REFERENCE = 'pnet'

MODEL_NAMES = [
    'pnet',
    'pnet_GO',
    'dense',
    'decision_tree',
    'adaboost',
    'linear_svm',
    'krr',
    'lgbm',
    'xgb',
    'random_forest',
    'rbf_svm',
]

DISPLAY_NAMES = {
    'pnet'         : 'P-NET',
    'pnet_GO'      : 'P-NET-GO',
    'dense'        : 'P-NET-FC',
    'decision_tree': 'Decision Tree',
    'adaboost'     : 'Ada. Boosting',
    'linear_svm'   : 'Linear SVR',
    'krr'          : 'Kernel Ridge Reg.',
    'lgbm'         : 'LightGBM',
    'xgb'          : 'XGBoost',
    'random_forest': 'Random Forest',
    'rbf_svm'      : 'RBF SVR',
}

# Left-to-right order on the x-axis; also fixes the colour each model gets.
PAPER_ORDER = [
    'Decision Tree',
    'Ada. Boosting',
    'Linear SVR',
    'Kernel Ridge Reg.',
    'LightGBM',
    'XGBoost',
    'Random Forest',
    'RBF SVR',
    'P-NET-FC',
    'P-NET',
    'P-NET-GO',
]

MODELS = ModelRegistry(
    names=MODEL_NAMES,
    display=DISPLAY_NAMES,
    order=PAPER_ORDER,
    col_prefix='AUC_log1p_',
)

# Same models, but the external-validation metrics carry that dataset's label prefix.
EXTERNAL = ModelRegistry(
    names=MODEL_NAMES,
    display=DISPLAY_NAMES,
    order=PAPER_ORDER,
    col_prefix='auc_dose_range_1_10_log1p_',
)

METRICS = {
    'r2'                : 'R²',
    'explained_variance': 'Explained Variance',
    'pearson_r'         : 'Pearson r',
    'spearman_r'        : 'Spearman r',
    'concordance_index' : 'Concordance Index',
    'mae'               : 'MAE',
    'rmse'              : 'RMSE',
}

BOUNDED_METRICS = {
    'pearson_r',
    'spearman_r',
    'concordance_index',
}
