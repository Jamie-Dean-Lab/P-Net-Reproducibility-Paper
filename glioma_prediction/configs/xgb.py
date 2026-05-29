import copy

from xgboost import XGBClassifier

from architecture.pipeline import MLPipeline
from .base_config import base_config, save_processor

xgb_config = {
    **copy.deepcopy(base_config),
    "run_id":             "xgb",
    "model":              XGBClassifier,
    "task":               "binary classification",
    "results_processors": [save_processor],
    "pipeline_class":     MLPipeline,
    "run_method":         "run_crossvalidation",
    "grid_search": {
        "model_params": {
            f"lr_{lr}_depth_{d}_mcw_{mcw}": {
                "n_estimators": 500,
                "learning_rate": lr,
                "max_depth": d,
                "min_child_weight": mcw,
                "random_state": 42,
                "verbosity": 0,
            }
            for lr in [0.01, 0.05, 0.1]
            for d in [3, 6, 9]
            for mcw in [1, 10]
        }
    },
}
