import copy

from sklearn.ensemble import AdaBoostClassifier
from architecture.pipeline import MLPipeline
from .base_config import save_processor
from .nested_CV_base_config import nested_CV_base_config

adaboost_nested_CV_config = {
    **copy.deepcopy(nested_CV_base_config),
    "run_id":                    "adaboost_nested_CV",
    "model":                     AdaBoostClassifier,
    "pipeline_class":            MLPipeline,
    "results_processors":        [save_processor],
    "grid_search":               {"model_params": {
        f"lr_{l}_estimators_{n}": {"learning_rate": l, "n_estimators": n}
        for l in [0.01, 0.05, 0.1, 0.3, 1] for n in [50, 100]
    }},
}
