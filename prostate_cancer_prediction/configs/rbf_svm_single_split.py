import copy
from sklearn.svm import SVC

from architecture.pipeline import MLPipeline
from .base_config import (base_config, f1_selection, auprc_selection, auc_selection, save_processor)

rbf_svm_single_split_config = {
    **copy.deepcopy(base_config),
    "run_id":                 "rbf_svm_single_split",
    "model":                  SVC,
    "task":                   "binary classification",
    "pipeline_class":         MLPipeline,
    "results_processors":     [save_processor],
    "val_metric":             {"f1": f1_selection, "auprc": auprc_selection, "auc": auc_selection},
    "grid_search":            {"model_params": {
        f"c_{c}_g_{g}": {"kernel": "rbf", "probability": True, "C": c, "class_weight": {0: 0.75, 1: 1.5}, "gamma": g}
        for c in [0.001, 0.01, 0.1, 1, 10, 100, 1000] for g in [0.001, 0.01, 0.1, 1]
    }},
    "run_method":             "run_single_split"
}