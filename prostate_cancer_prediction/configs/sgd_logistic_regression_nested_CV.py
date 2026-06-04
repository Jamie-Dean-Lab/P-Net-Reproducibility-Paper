import copy

from sklearn.linear_model import SGDClassifier

from architecture.pipeline import MLPipeline
from .base_config import save_processor
from .nested_CV_base_config import nested_CV_base_config

sgd_logistic_regression_nested_CV_config = {
    **copy.deepcopy(nested_CV_base_config),
    "run_id":                    "sgd_logistic_regression_nested_CV",
    "model":                     SGDClassifier,
    "pipeline_class":            MLPipeline,
    "results_processors":        [save_processor],
    "grid_search":            {"model_params": {
        f"alpha_{a}_penalty_{p}": {
            "alpha": a, "penalty": p, "class_weight": {0: 0.75, 1: 1.5}, "loss": "log_loss"
        }
        for a in [0.0001, 0.001, 0.009, 0.01, 0.09, 1, 5, 10] for p in ["l1", "l2"]
    }},
}
