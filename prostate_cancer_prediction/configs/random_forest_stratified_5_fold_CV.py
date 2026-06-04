import copy

from sklearn.ensemble import RandomForestClassifier

from architecture.pipeline import MLPipeline
from .base_config import save_processor, base_config
from .stratified_5_fold_CV_base_config import stratified_5_fold_CV_base_config

random_forest_stratified_5_fold_CV_config = {
    **copy.deepcopy(base_config),
    **copy.deepcopy(stratified_5_fold_CV_base_config),
    "run_id":                    "random_forest_stratified_5_fold_CV",
    "model":                     RandomForestClassifier,
    "pipeline_class":            MLPipeline,
    "task":                      "binary classification",
    "results_processors":        [save_processor],
    "model_params": {"bootstrap": False, "max_depth": None, "n_estimators": 50, "class_weight": {0: 0.75, 1: 1.5}}
}
