import copy
from sklearn.ensemble import RandomForestClassifier

from architecture.pipeline import MLPipeline
from .base_config import (base_config, f1_selection, auprc_selection, auc_selection, save_processor)

random_forest_single_split_elmarakeby_config = {
    **copy.deepcopy(base_config),
    "run_id":                 "random_forest_single_split_elmarakeby",
    "model":                  RandomForestClassifier,
    "task":                   "binary classification",
    "pipeline_class":         MLPipeline,
    "results_processors":     [save_processor],
    "model_params":           {"bootstrap": False, "max_depth": None, "n_estimators": 50,
                               "class_weight": {0: 0.75, 1: 1.5}},
    "val_metric":             {},
    "grid_search":            [],
    "run_method":             "run_single_split"
}
