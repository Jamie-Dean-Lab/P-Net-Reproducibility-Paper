import copy
from xgboost import XGBRegressor

from architecture.pipeline import MLPipeline
from .base_config import base_config

xgb_config = {
    **copy.deepcopy(base_config),
    "run_id":         "xgb",
    "model":          XGBRegressor,
    "task":           "regression",
    "pipeline_class": MLPipeline,
    "run_method":     "run_crossvalidation",
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
