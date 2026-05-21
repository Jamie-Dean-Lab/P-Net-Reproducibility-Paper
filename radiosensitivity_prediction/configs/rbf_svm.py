import copy

import numpy as np
from sklearn.svm import SVR

from architecture.pipeline import MLPipeline
from .base_config import base_config

rbf_svm_config = {
    **copy.deepcopy(base_config),
    "run_id":         "rbf_svm",
    "model":          SVR,
    "task":           "regression",
    "pipeline_class": MLPipeline,
    "run_method":     "run_crossvalidation",
    "grid_search": {
        "model_params": {
            f"C_{c}_gamma_{g}": {
                "kernel": "rbf",
                "C": c,
                "gamma": g,
            }
            for c in np.logspace(-6, 6, 7).tolist()
            for g in np.logspace(-6, 6, 7).tolist()
        }
    },
}