import copy

import numpy as np
from sklearn.svm import LinearSVC

from architecture.pipeline import MLPipeline
from .base_config import base_config, save_processor
from ..wrapper import ProbaWrapper

svc_config = {
    **copy.deepcopy(base_config),
    "run_id":             "svc",
    "model":              ProbaWrapper,
    "task":               "multiclass",
    "results_processors": [save_processor],
    "pipeline_class":     MLPipeline,
    "run_method":         "run_crossvalidation",
    "grid_search": {
        "model_params": {
            f"c_{c}": {"estimator": LinearSVC, "args": {"C": c}}
            for c in np.logspace(-6, 6, 10).tolist()
        }
    },
}