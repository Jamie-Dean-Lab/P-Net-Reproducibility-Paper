import copy
import numpy as np
from sklearn.linear_model import SGDClassifier

from architecture.pipeline import MLPipeline
from .base_config import base_config, save_processor
from ..wrapper import ProbaWrapper

sgd_logistic_regression_config = {
    **copy.deepcopy(base_config),
    "run_id":             "sgd_logistic_regression",
    "model":              ProbaWrapper,
    "task":               "multiclass",
    "results_processors": [save_processor],
    "pipeline_class":     MLPipeline,
    "run_method":         "run_crossvalidation",
    "grid_search": {
        "model_params": {
            f"alpha_{a}_penalty_{p}": {
                "estimator": SGDClassifier,
                "args": {
                    "loss": "log_loss",
                    "alpha": a,
                    "penalty": p,
                    "max_iter": 1000,
                    "random_state": 42,
                },
            }
            for a in np.logspace(-6, 3, 7).tolist()
            for p in ["l1", "l2"]
        }
    },
}
