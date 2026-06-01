import copy

from sklearn.tree import DecisionTreeClassifier

from architecture.pipeline import MLPipeline
from .base_config import (save_processor, base_config, f1_selection, auprc_selection, auc_selection)

decision_tree_nested_CV_config = {
    **copy.deepcopy(base_config),
    "run_id":                    "decision_tree_nested_CV",
    "model":                     DecisionTreeClassifier,
    "pipeline_class":            MLPipeline,
    "task":                      "binary classification",
    "results_processors":        [save_processor],
    "val_metric":                {"f1": f1_selection, "auprc": auprc_selection, "auc": auc_selection},
    "inner_kfolds":              5,
    "outer_kfolds":              5,
    "stratified":                True,
    "run_method":                "run_crossvalidation",
    "grid_search":            {"model_params": {
        f"ssplit_{s}_depth_{d}": {"min_samples_split": s, "max_depth": d, "class_weight": {0: 0.75, 1: 1.5}}
        for s in range(10, 500, 20) for d in range(1, 20, 2)
    }},
}
del decision_tree_nested_CV_config["train_samples"]
del decision_tree_nested_CV_config["val_samples"]
del decision_tree_nested_CV_config["test_samples"]


