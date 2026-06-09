import copy
from sklearn.linear_model import SGDClassifier

from architecture.pipeline import MLPipeline
from .base_config import (base_config, f1_selection, auprc_selection, auc_selection, save_processor)

sgd_logistic_regression_single_split_elmarakeby_config = {
    **copy.deepcopy(base_config),
    "run_id":                 "sgd_logistic_regression_single_split_elmarakeby",
    "model":                  SGDClassifier,
    "task":                   "binary classification",
    "pipeline_class":         MLPipeline,
    "results_processors":     [save_processor],
    "model_params":           {"alpha": 0.01, "penalty": "l2", "class_weight": {0: 0.75, 1: 1.5},
                               "loss": "log_loss"},
    "val_metric":             {},
    "grid_search":            [],
    "run_method":             "run_single_split"
}
