import copy

from sklearn.ensemble import AdaBoostClassifier

from architecture.pipeline import MLPipeline
from .base_config import base_config, save_processor

adaboost_config = {
    **copy.deepcopy(base_config),
    "run_id":             "adaboost",
    "model":              AdaBoostClassifier,
    "task":               "binary classification",
    "results_processors": [save_processor],
    "pipeline_class":     MLPipeline,
    "run_method":         "run_crossvalidation",
    "grid_search": {
        "model_params": {
            f"n_est_{n}": {"n_estimators": n}
            for n in [5]
        }
    },
}
