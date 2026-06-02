import copy
from sklearn.tree import DecisionTreeClassifier

from architecture.pipeline import MLPipeline
from .base_config import (base_config, f1_selection, auprc_selection, auc_selection, save_processor)

decision_tree_single_split_config = {
    **copy.deepcopy(base_config),
    "run_id":                 "decision_tree_single_split",
    "model":                  DecisionTreeClassifier,
    "task":                   "binary classification",
    "pipeline_class":         MLPipeline,
    "results_processors":     [save_processor],
    "val_metric":             {"f1": f1_selection, "auprc": auprc_selection, "auc": auc_selection},
    "grid_search":            {"model_params": {
        f"ssplit_{s}_depth_{d}": {"min_samples_split": s, "max_depth": d, "class_weight": {0: 0.75, 1: 1.5}}
        for s in range(10, 500, 20) for d in range(1, 20, 2)
    }},
    "run_method":             "run_single_split"
}