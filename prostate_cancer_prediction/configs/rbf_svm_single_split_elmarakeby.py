import copy
from sklearn.svm import SVC

from architecture.pipeline import MLPipeline
from .base_config import (base_config, f1_selection, auprc_selection, auc_selection, save_processor)

rbf_svm_single_split_elmarakeby_config = {
    **copy.deepcopy(base_config),
    "run_id":                 "rbf_svm_single_split_elmarakeby",
    "model":                  SVC,
    "task":                   "binary classification",
    "pipeline_class":         MLPipeline,
    "results_processors":     [save_processor],
    "use_validation_on_test": True,
    "val_metric":             {"auc": auc_selection},
    "grid_search":            {"model_params": {
        f"c_{c}_g_{g}": {"kernel": "rbf", "probability": True, "C": c, "class_weight": {0: 0.75, 1: 1.5}, "gamma": g}
        for c in [100] for g in [0.001]
    }},
    "run_method":             "run_single_split"
}