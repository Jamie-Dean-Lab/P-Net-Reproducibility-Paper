import copy

from lightgbm import LGBMClassifier

from architecture.pipeline import MLPipeline
from .base_config import base_config, save_processor

lgbm_config = {
    **copy.deepcopy(base_config),
    "run_id":             "lgbm",
    "model":              LGBMClassifier,
    "task":               "binary classification",
    "results_processors": [save_processor],
    "pipeline_class":     MLPipeline,
    "run_method":         "run_crossvalidation",
    "grid_search": {
        "model_params": {
            f"lr_{lr}_leaves_{l}_mcs_{mcs}": {
                "n_estimators": 500,
                "learning_rate": lr,
                "num_leaves": l,
                "min_child_samples": mcs,
                "random_state": 42,
                "verbosity": -1,
            }
            for lr in [0.01, 0.05, 0.1]
            for l in [15, 31, 63]
            for mcs in [10, 30]
        }
    },
}
