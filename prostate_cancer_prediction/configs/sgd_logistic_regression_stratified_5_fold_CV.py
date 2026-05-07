import copy

from sklearn.linear_model import SGDClassifier

from evaluation import collate_folds
from pipeline import MLPipeline
from .base_config import (save_processor, train_samples,
                          val_samples, base_config)

sgd_logistic_regression_stratified_5_fold_CV_config = {
    **copy.deepcopy(base_config),
    "run_id":                    "sgd_logistic_regression_stratified_5_fold_CV",
    "model":                     SGDClassifier,
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
    "model_params": {"alpha": 0.01, "penalty": "l2", "class_weight": {0: 0.75, 1: 1.5}, "loss": "log_loss"}
}
