import copy
from lightgbm import LGBMClassifier

from architecture.pipeline import MLPipeline
from .base_config import base_config, save_processor
from ..wrapper import ProbaWrapper

lgbm_config = {
    **copy.deepcopy(base_config),
    "run_id":             "lgbm",
    "model":              ProbaWrapper,
    "task":               "multiclass",
    "results_processors": [save_processor],
    "pipeline_class":     MLPipeline,
    "run_method":         "run_crossvalidation",
    "grid_search": {
        "model_params": {
            f"lr_{lr}_leaves_{l}_mcs_{mcs}": {
                "estimator": LGBMClassifier,
                "args": {
                    "n_estimators": 100,
                    "learning_rate": lr,
                    "num_leaves": l,
                    "min_child_samples": mcs,
                    "random_state": 42,
                    "verbosity": -1,
                    "n_jobs": -1,
                    "feature_fraction": 0.1,
                    "feature_fraction_seed": 42,
                    "bagging_fraction": 0.8,
                    "bagging_freq": 5,
                    "bagging_seed": 42,
                    "max_bin": 63,
                    "min_data_in_bin": 5,
                },
            }
            for lr in [0.01, 0.05, 0.1]
            for l in [15, 31, 63]
            for mcs in [10, 30]
        }
    },
}