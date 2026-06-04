import copy

from sklearn.linear_model import SGDClassifier

from architecture.pipeline import MLPipeline
from .base_config import save_processor, base_config
from .stratified_5_fold_CV_base_config import stratified_5_fold_CV_base_config

sgd_logistic_regression_stratified_5_fold_CV_config = {
    **copy.deepcopy(base_config),
    **copy.deepcopy(stratified_5_fold_CV_base_config),
    "run_id":                    "sgd_logistic_regression_stratified_5_fold_CV",
    "model":                     SGDClassifier,
    "pipeline_class":            MLPipeline,
    "task":                      "binary classification",
    "results_processors":        [save_processor],
    "model_params": {"alpha": 0.01, "penalty": "l2", "class_weight": {0: 0.75, 1: 1.5}, "loss": "log_loss"}
}
