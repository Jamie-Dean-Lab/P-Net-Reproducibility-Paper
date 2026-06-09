import copy
from sklearn.tree import DecisionTreeClassifier

from architecture.pipeline import MLPipeline
from .base_config import (base_config, f1_selection, auprc_selection, auc_selection, save_processor)

decision_tree_single_split_elmarakeby_config = {
    **copy.deepcopy(base_config),
    "run_id":                 "decision_tree_single_split_elmarakeby",
    "model":                  DecisionTreeClassifier,
    "task":                   "binary classification",
    "pipeline_class":         MLPipeline,
    "results_processors":     [save_processor],
    "model_params":           {"min_samples_split": 10, "max_depth": 10, "class_weight": {0: 0.75, 1: 1.5}},
    "val_metric":             {},
    "grid_search":            [],
    "run_method":             "run_single_split"
}
