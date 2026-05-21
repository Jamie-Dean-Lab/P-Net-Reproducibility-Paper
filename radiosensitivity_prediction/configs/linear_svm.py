import copy

import numpy as np
from sklearn.svm import SVR

from architecture.pipeline import MLPipeline
from .base_config import base_config

linear_svm_config = {
    **copy.deepcopy(base_config),
    "run_id":         "linear_svm",
    "model":          SVR,
    "task":           "regression",
    "pipeline_class": MLPipeline,
    "run_method":     "run_crossvalidation",
    "grid_search": {
        "model_params": {
            f"C_{c}": {
                "kernel": "linear",
                "C": c,
            }
            for c in np.logspace(-6, 6, 10).tolist()
        }
    },
}