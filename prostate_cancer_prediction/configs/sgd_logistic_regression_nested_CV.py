import copy

from sklearn.linear_model import SGDClassifier

from architecture.pipeline import MLPipeline
from .base_config import (save_processor, base_config, f1_selection, auprc_selection, auc_selection)

sgd_logistic_regression_nested_CV_config = {
    **copy.deepcopy(base_config),
    "run_id":                    "sgd_logistic_regression_nested_CV",
    "model":                     SGDClassifier,
    "pipeline_class":            MLPipeline,
    "task":                      "binary classification",
    "results_processors":        [save_processor],
    "val_metric":                {"f1": f1_selection, "auprc": auprc_selection, "auc": auc_selection},
    "inner_kfolds":              5,
    "outer_kfolds":              5,
    "stratified":                True,
    "run_method":                "run_crossvalidation",
    "grid_search":            {"model_params": {
        f"alpha_{a}_penalty_{p}": {
            "alpha": a, "penalty": p, "class_weight": {0: 0.75, 1: 1.5}, "loss": "log_loss"
        }
        for a in [0.0001, 0.001, 0.009, 0.01, 0.09, 1, 5, 10] for p in ["l1", "l2"]
    }},
}
del sgd_logistic_regression_nested_CV_config["train_samples"]
del sgd_logistic_regression_nested_CV_config["val_samples"]
del sgd_logistic_regression_nested_CV_config["test_samples"]


