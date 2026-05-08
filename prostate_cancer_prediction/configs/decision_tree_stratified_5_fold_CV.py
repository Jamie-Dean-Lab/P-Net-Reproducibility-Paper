import copy

from sklearn.tree import DecisionTreeClassifier

from architecture.evaluation import collate_folds
from architecture.pipeline import MLPipeline
from .base_config import (save_processor, train_samples,
                          val_samples, base_config)

decision_tree_stratified_5_fold_CV_config = {
    **copy.deepcopy(base_config),
    "run_id":                    "decision_tree_stratified_5_fold_CV",
    "model":                     DecisionTreeClassifier,
    "pipeline_class":            MLPipeline,
    "task":                      "binary classification",
    "results_processors":        [save_processor],
    "val_metric":                {},
    "grid_search":               [],
    "grid_search_collators":     [],
    "inner_kfolds":              5,
    "fold_collators":            [collate_folds],
    "stratified":                True,
    "tt_split_seed":             123,
    "tv_split_seed":             123,
    "run_method":                "run_crossvalidation",
    "train_samples":             train_samples + val_samples,
    "val_samples":               [],
    "model_params": {"min_samples_split": 10, "max_depth": 10, "class_weight": {0: 0.75, 1: 1.5}}
}

