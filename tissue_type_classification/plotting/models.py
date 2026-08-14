"""
Models compared for the tissue type classification task.

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
    'dense',
    'decision_tree',
    'adaboost',
    'sgd_logistic_regression',
    'svc',
    'rbf_svm',
    'lgbm',
    'xgb',
    'random_forest',
]

DISPLAY_NAMES = {
    'pnet'                   : 'P-NET',
    'dense'                  : 'P-NET-FC',
    'decision_tree'          : 'Decision Tree',
    'adaboost'               : 'Ada. Boosting',
    'sgd_logistic_regression': 'Logistic Reg.',
    'svc'                    : 'Linear SVM',
    'rbf_svm'                : 'RBF SVM',
    'lgbm'                   : 'LightGBM',
    'xgb'                    : 'XGBoost',
    'random_forest'          : 'Random Forest',
}

# Left-to-right order on the x-axis; also fixes the colour each model gets.
PAPER_ORDER = [
    'Decision Tree',
    'Ada. Boosting',
    'Logistic Reg.',
    'Linear SVM',
    'RBF SVM',
    'LightGBM',
    'XGBoost',
    'Random Forest',
    'P-NET-FC',
    'P-NET',
]

MODELS = ModelRegistry(
    names=MODEL_NAMES,
    display=DISPLAY_NAMES,
    order=PAPER_ORDER,
    col_prefix='',
)

# Same models, but the external-validation metrics carry that dataset's label prefix.
EXTERNAL = ModelRegistry(
    names=MODEL_NAMES,
    display=DISPLAY_NAMES,
    order=PAPER_ORDER,
    col_prefix='',
)

METRICS = {
    'auc'     : 'AUROC',
    'auprc'   : 'AUPRC',
    'f1'      : 'Weighted F1',
    'accuracy': 'Accuracy',
}

BOUNDED_METRICS = {
    'auc',
    'auprc',
    'f1',
    'accuracy',
}
