import copy
from sklearn.ensemble import AdaBoostClassifier

from architecture.pipeline import MLPipeline
from .base_config import (base_config, f1_selection, auprc_selection, auc_selection, save_processor)

adaboost_single_split_elmarakeby_config = {
    **copy.deepcopy(base_config),
    "run_id":                 "adaboost_single_split_elmarakeby",
    "model":                  AdaBoostClassifier,
    "task":                   "binary classification",
    "pipeline_class":         MLPipeline,
    "results_processors":     [save_processor],
    "model_params":           {"learning_rate": 0.1, "n_estimators": 50},
    "val_metric":             {},
    "grid_search":            [],
    "run_method":             "run_single_split"
}
