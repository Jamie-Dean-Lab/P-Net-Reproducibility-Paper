import copy

from sklearn.ensemble import RandomForestClassifier

from architecture.pipeline import MLPipeline
from .base_config import save_processor
from .nested_CV_base_config import nested_CV_base_config

random_forest_nested_CV_config = {
    **copy.deepcopy(nested_CV_base_config),
    "run_id":                    "random_forest_nested_CV",
    "model":                     RandomForestClassifier,
    "pipeline_class":            MLPipeline,
    "results_processors":        [save_processor],
    "grid_search":            {"model_params": {
        f"bootstrap_{b}_depth_{d}_estimators_{n}": {
            "bootstrap": b, "max_depth": d, "n_estimators": n, "class_weight": {0: 0.75, 1: 1.5}
        }
        for b in [True, False] for d in [10, 30, 50, 70, None] for n in [10, 50, 100, 200]
    }},
}
