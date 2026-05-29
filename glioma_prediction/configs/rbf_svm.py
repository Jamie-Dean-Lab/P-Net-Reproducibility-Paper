import copy

import numpy as np
from sklearn.svm import SVC

from architecture.pipeline import MLPipeline
from .base_config import base_config, save_processor

rbf_svm_config = {
    **copy.deepcopy(base_config),
    "run_id":             "rbf_svm",
    "model":              SVC,
    "task":               "binary classification",
    "results_processors": [save_processor],
    "pipeline_class":     MLPipeline,
    "run_method":         "run_crossvalidation",
    "grid_search": {
        "model_params": {
            f"C_{c}_gamma_{g}": {"kernel": "rbf", "C": c, "gamma": g, "probability": True}
            for c in np.logspace(-6, 6, 7).tolist()
            for g in np.logspace(-6, 6, 7).tolist()
        }
    },
}
