import copy

from sklearn.ensemble import RandomForestClassifier

from pipeline import MLPipeline
from .base_config import (save_processor, base_config, f1_selection, auprc_selection, auc_selection)

random_forest_nested_CV_config = {
    **copy.deepcopy(base_config),
    "run_id":                    "random_forest_nested_CV",
    "model":                     RandomForestClassifier,
    "pipeline_class":            MLPipeline,
    "task":                      "binary classification",
    "results_processors":        [save_processor],
    "val_metric":                {"f1": f1_selection, "auprc": auprc_selection, "auc": auc_selection},
    "inner_kfolds":              5,
    "outer_kfolds":              10,
    "stratified":                True,
    "run_method":                "run_crossvalidation",
    "grid_search":            {"model_params": {
        f"bootstrap_{b}_depth_{d}_estimators_{n}": {
            "bootstrap": b, "max_depth": d, "n_estimators": n, "class_weight": {0: 0.75, 1: 1.5}
        }
        for b in [True, False] for d in [10, 30, 50, 70, None] for n in [10, 50, 100, 200]
    }},
}
del random_forest_nested_CV_config["train_samples"]
del random_forest_nested_CV_config["val_samples"]
del random_forest_nested_CV_config["test_samples"]


