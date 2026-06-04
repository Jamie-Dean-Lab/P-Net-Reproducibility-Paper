import copy

from sklearn.svm import SVC

from architecture.pipeline import MLPipeline
from .base_config import save_processor, base_config
from .stratified_5_fold_CV_base_config import stratified_5_fold_CV_base_config

rbf_svm_stratified_5_fold_CV_config = {
    **copy.deepcopy(base_config),
    **copy.deepcopy(stratified_5_fold_CV_base_config),
    "run_id":                    "rbf_svm_stratified_5_fold_CV",
    "model":                     SVC,
    "pipeline_class":            MLPipeline,
    "task":                      "binary classification",
    "results_processors":        [save_processor],
    "model_params": {"kernel": "rbf", "probability": True, "C": 100, "class_weight": {0: 0.75, 1: 1.5}, "gamma": 0.001}
}
