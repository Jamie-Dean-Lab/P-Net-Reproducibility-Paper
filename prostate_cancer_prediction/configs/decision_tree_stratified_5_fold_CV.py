import copy

from sklearn.tree import DecisionTreeClassifier

from architecture.pipeline import MLPipeline
from .base_config import save_processor, base_config
from .stratified_5_fold_CV_base_config import stratified_5_fold_CV_base_config

decision_tree_stratified_5_fold_CV_config = {
    **copy.deepcopy(base_config),
    **copy.deepcopy(stratified_5_fold_CV_base_config),
    "run_id":                    "decision_tree_stratified_5_fold_CV",
    "model":                     DecisionTreeClassifier,
    "pipeline_class":            MLPipeline,
    "task":                      "binary classification",
    "results_processors":        [save_processor],
    "model_params": {"min_samples_split": 10, "max_depth": 10, "class_weight": {0: 0.75, 1: 1.5}}
}
