"""
Models compared for the prostate cancer prediction task.

Single source of truth for which runs are compared, what they are called in figures,
and the order they appear in. Imported by the nested-CV and external-validation plots
and by significance_testing, so a display name cannot disagree between a figure and
the significance table.

Note: "dense_single_layer" is a genuine dense single layer, distinct from the
fully-connected P-NET ("pnetfc" -> P-NET-FC).
"""

from architecture.plotting.model_comparison import ModelRegistry

# The model every other one is compared against.
REFERENCE = 'pnet_nested_CV'

MODEL_NAMES = [
    'pnet_nested_CV',
    'pnet_GO_nested_CV',
    'pnetfc_nested_CV',
    'dense_single_layer_nested_CV',
    'decision_tree_nested_CV',
    'adaboost_nested_CV',
    'linear_svm_nested_CV',
    'random_forest_nested_CV',
    'rbf_svm_nested_CV',
    'sgd_logistic_regression_nested_CV',
]

DISPLAY_NAMES = {
    'pnet_nested_CV'                   : 'P-NET',
    'pnet_GO_nested_CV'                : 'P-NET-GO',
    'pnetfc_nested_CV'                 : 'P-NET-FC',
    'dense_single_layer_nested_CV'     : 'Dense Single Layer',
    'decision_tree_nested_CV'          : 'Decision Tree',
    'adaboost_nested_CV'               : 'Ada. Boosting',
    'linear_svm_nested_CV'             : 'Linear SVM',
    'random_forest_nested_CV'          : 'Random Forest',
    'rbf_svm_nested_CV'                : 'RBF SVM',
    'sgd_logistic_regression_nested_CV': 'Logistic Regression',
}

# Left-to-right order on the x-axis; also fixes the colour each model gets.
PAPER_ORDER = [
    'Decision Tree',
    'Ada. Boosting',
    'Logistic Regression',
    'Linear SVM',
    'Dense Single Layer',
    'P-NET-FC',
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
    col_prefix='metastatic_',
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
