import copy

import numpy as np
from sklearn.linear_model import SGDRegressor

from architecture.pipeline import MLPipeline
from .base_config import base_config

sgd_logistic_regression_config = {
    **copy.deepcopy(base_config),
    "run_id":         "sgd_logistic_regression",
    "model":          SGDRegressor,
    "task":           "regression",
    "pipeline_class": MLPipeline,
    "run_method":     "run_crossvalidation",
    "grid_search": {
        "model_params": {
            f"alpha_{a}_penalty_{p}": {
                "loss": "squared_error",
                "alpha": a,
                "penalty": p,
                "max_iter": 1000,
                "random_state": 42,
            }
            for a in np.logspace(-6, 3, 7).tolist()
            for p in ["l1", "l2"]
        }
    },
}