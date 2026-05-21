import copy
from sklearn.ensemble import AdaBoostRegressor

from architecture.pipeline import MLPipeline
from .base_config import base_config

adaboost_config = {
    **copy.deepcopy(base_config),
    "run_id":         "adaboost",
    "model":          AdaBoostRegressor,
    "task":           "regression",
    "pipeline_class": MLPipeline,
    "run_method":     "run_crossvalidation",
    "grid_search": {
        "model_params": {
            f"lr_{lr}_est_{n}": {
                "n_estimators": n,
                "learning_rate": lr,
                "random_state": 42,
            }
            for lr in [0.01, 0.05, 0.1, 0.3, 1]
            for n in [50, 100]
        }
    },
}
