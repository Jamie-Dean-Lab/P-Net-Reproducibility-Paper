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
    "model_params":           {"kernel": "rbf", "probability": True, "C": 100,
                               "class_weight": {0: 0.75, 1: 1.5}, "gamma": 0.001},
    "val_metric":             {},
    "grid_search":            [],
    "run_method":             "run_single_split"
}
