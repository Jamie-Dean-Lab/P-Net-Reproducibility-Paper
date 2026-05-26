import copy
from sklearn.ensemble import AdaBoostClassifier

from architecture.pipeline import MLPipeline
from .base_config import base_config, save_processor
from ..ovr import ProbaWrapper

adaboost_config = {
    **copy.deepcopy(base_config),
    "run_id":             "adaboost",
    "model":              ProbaWrapper,
    "task":               "multiclass",
    "results_processors": [save_processor],
    "pipeline_class":     MLPipeline,
    "run_method":         "run_crossvalidation",
    "grid_search": {
        "model_params": {
            f"n_est_{n}": {"estimator": AdaBoostClassifier, "args": {"n_estimators": n, "algorithm": "SAMME"}}
            for n in [50, 100, 200]
        }
    },
}
