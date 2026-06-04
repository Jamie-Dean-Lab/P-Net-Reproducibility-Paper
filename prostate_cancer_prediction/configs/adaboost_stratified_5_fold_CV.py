import copy

from sklearn.ensemble import AdaBoostClassifier

from architecture.pipeline import MLPipeline
from .base_config import save_processor, base_config
from .stratified_5_fold_CV_base_config import stratified_5_fold_CV_base_config

adaboost_stratified_5_fold_CV_config = {
    **copy.deepcopy(base_config),
    **copy.deepcopy(stratified_5_fold_CV_base_config),
    "run_id":                    "adaboost_stratified_5_fold_CV",
    "model":                     AdaBoostClassifier,
    "pipeline_class":            MLPipeline,
    "task":                      "binary classification",
    "results_processors":        [save_processor],
    "model_params": {"learning_rate": 0.1, "n_estimators": 50}
}
