import copy

from sklearn.ensemble import AdaBoostClassifier
from pipeline import MLPipeline
from .base_config import (save_processor, base_config, f1_selection, auprc_selection, auc_selection)

adaboost_nested_CV_config = {
    **copy.deepcopy(base_config),
    "run_id":                    "adaboost_nested_CV",
    "model":                     AdaBoostClassifier,
    "pipeline_class":            MLPipeline,
    "task":                      "binary classification",
    "results_processors":        [save_processor],
    "val_metric":                {"f1": f1_selection, "auprc": auprc_selection, "auc": auc_selection},
    "inner_kfolds":              5,
    "outer_kfolds":              10,
    "stratified":                True,
    "run_method":                "run_crossvalidation",
    "grid_search":               {"model_params": {
        f"lr_{l}_estimators_{n}": {"learning_rate": l, "n_estimators": n}
        for l in [0.01, 0.05, 0.1, 0.3, 1] for n in [50, 100]
    }},
}
del adaboost_nested_CV_config["train_samples"]
del adaboost_nested_CV_config["val_samples"]
del adaboost_nested_CV_config["test_samples"]


