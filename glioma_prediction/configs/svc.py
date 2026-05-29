import copy

import numpy as np
from sklearn.svm import SVC

from architecture.pipeline import MLPipeline
from .base_config import base_config, save_processor

svc_config = {
    **copy.deepcopy(base_config),
    "run_id":             "svc",
    "model":              SVC,
    "task":               "binary classification",
    "results_processors": [save_processor],
    "pipeline_class":     MLPipeline,
    "run_method":         "run_crossvalidation",
    "grid_search": {
        "model_params": {
            f"c_{c}": {"kernel": "linear", "C": c, "probability": True}
            for c in np.logspace(-6, 6, 10).tolist()
        }
    },
}
