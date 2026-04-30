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
    "use_validation_on_test": True,
    "val_metric":             {"f1": f1_selection},
    "grid_search":            {"model_params": {
        f"alpha_{a}_penalty_{p}": {
            "alpha": a, "penalty": p, "class_weight": {0: 0.75, 1: 1.5}, "loss": "log_loss"
        }
        for a in [0.01] for p in ["l2"]
    }},
    "run_method":             "run_single_split"
}