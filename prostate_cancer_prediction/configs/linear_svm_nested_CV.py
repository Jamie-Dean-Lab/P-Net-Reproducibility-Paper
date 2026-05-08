import copy

from sklearn.svm import SVC

from pipeline import MLPipeline
from .base_config import (save_processor, base_config, f1_selection, auprc_selection, auc_selection)

linear_svm_nested_CV_config = {
    **copy.deepcopy(base_config),
    "run_id":                    "linear_svm_nested_CV",
    "model":                     SVC,
    "pipeline_class":            MLPipeline,
    "task":                      "binary classification",
    "results_processors":        [save_processor],
    "val_metric":                {"f1": f1_selection, "auprc": auprc_selection, "auc": auc_selection},
    "inner_kfolds":              5,
    "outer_kfolds":              10,
    "stratified":                True,
    "run_method":                "run_crossvalidation",
    "grid_search":               {"model_params": {
        f"c_{c}": {"kernel": "linear", "probability": True, "C": c, "class_weight": {0: 0.75, 1: 1.5}}
        for c in [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    }},
}
del linear_svm_nested_CV_config["train_samples"]
del linear_svm_nested_CV_config["val_samples"]
del linear_svm_nested_CV_config["test_samples"]


