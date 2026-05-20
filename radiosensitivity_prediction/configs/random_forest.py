import copy
from sklearn.ensemble import RandomForestRegressor

from architecture.pipeline import MLPipeline
from .base_config import base_config

random_forest_config = {
    **copy.deepcopy(base_config),
    "run_id": "random_forest",
    "model": RandomForestRegressor,
    "task": "regression",
    "pipeline_class": MLPipeline,
    "run_method": "run_crossvalidation",
    "grid_search": {
        "model_params": {
            f"bootstrap_{b}_depth_{d}_estimators_{n}": {"bootstrap": b, "max_depth": d, "n_estimators": n}
            for b in [True, False] for d in [10, 30, 50, 70, None] for n in [10, 50, 100, 200]
        }
    }
}
