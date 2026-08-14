"""
Models compared for the glioma prediction task.

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
    'svc',
    'random_forest',
    'rbf_svm',
    'xgb',
    'lgbm',
    'sgd_logistic_regression',
]

DISPLAY_NAMES = {
    'pnet'                   : 'P-NET',
    'pnet_GO'                : 'P-NET-GO',
    'dense'                  : 'P-NET-FC',
    'decision_tree'          : 'Decision Tree',
    'adaboost'               : 'Ada. Boosting',
    'svc'                    : 'Linear SVM',
    'random_forest'          : 'Random Forest',
    'rbf_svm'                : 'RBF SVM',
    'xgb'                    : 'XGBoost',
    'lgbm'                   : 'LightGBM',
    'sgd_logistic_regression': 'Logistic Regression',
}

# Left-to-right order on the x-axis; also fixes the colour each model gets.
PAPER_ORDER = [
    'Decision Tree',
    'Ada. Boosting',
    'Logistic Regression',
    'Linear SVM',
    'P-NET-FC',
    'LightGBM',
    'XGBoost',
    'Random Forest',
    'RBF SVM',
    'P-NET',
    'P-NET-GO',
]

MODELS = ModelRegistry(
    names=MODEL_NAMES,
    display=DISPLAY_NAMES,
    order=PAPER_ORDER,
    col_prefix='response_',
)

# Same models, but the external-validation metrics carry that dataset's label prefix.
EXTERNAL = ModelRegistry(
    names=MODEL_NAMES,
    display=DISPLAY_NAMES,
    order=PAPER_ORDER,
    col_prefix='response_',
)

METRICS = {
    'auc'      : 'AUROC',
    'auprc'    : 'AUPRC',
    'f1'       : 'F1',
    'accuracy' : 'Accuracy',
    'precision': 'Precision',
    'recall'   : 'Recall',
}

# Every metric here is a proportion, so all are capped at 1 on the y-axis.
BOUNDED_METRICS = None
