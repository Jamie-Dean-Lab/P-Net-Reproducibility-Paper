import copy

from sklearn.ensemble import AdaBoostClassifier

from architecture.evaluation import collate_folds
from architecture.pipeline import MLPipeline
from .base_config import (save_processor, train_samples,
                          val_samples, base_config)

adaboost_stratified_5_fold_CV_config = {
    **copy.deepcopy(base_config),
    "run_id":                    "adaboost_stratified_5_fold_CV",
    "model":                     AdaBoostClassifier,
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
    "model_params": {"learning_rate": 0.1, "n_estimators": 50}
}


