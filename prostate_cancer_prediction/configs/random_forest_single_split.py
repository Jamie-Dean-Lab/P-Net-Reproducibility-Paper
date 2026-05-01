import copy
from sklearn.ensemble import RandomForestClassifier

from architecture.pipeline import MLPipeline
from .base_config import (base_config, f1_selection, auprc_selection, auc_selection, save_processor)

random_forest_single_split_config = {
    **copy.deepcopy(base_config),
    "run_id":                 "random_forest_single_split",
    "model":                  RandomForestClassifier,
    "task":                   "binary classification",
    "pipeline_class":         MLPipeline,
    "results_processors":     [save_processor],
    "use_validation_on_test": True,
    "val_metric":             {"f1": f1_selection, "auprc": auprc_selection, "auc": auc_selection},
    "grid_search":            {"model_params": {
        f"bootstrap_{b}_depth_{d}_estimators_{n}": {
            "bootstrap": b, "max_depth": d, "n_estimators": n, "class_weight": {0: 0.75, 1: 1.5}
        }
        for b in [True, False] for d in [10, 30, 50, 70, None] for n in [10, 50, 100, 200]
    }},
    "run_method":             "run_single_split"
}
