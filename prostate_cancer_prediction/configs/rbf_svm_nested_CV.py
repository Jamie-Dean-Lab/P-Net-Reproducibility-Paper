import copy

from sklearn.svm import SVC

from architecture.pipeline import MLPipeline
from .base_config import save_processor
from .nested_CV_base_config import nested_CV_base_config

rbf_svm_nested_CV_config = {
    **copy.deepcopy(nested_CV_base_config),
    "run_id":                    "rbf_svm_nested_CV",
    "model":                     SVC,
    "pipeline_class":            MLPipeline,
    "results_processors":        [save_processor],
    "grid_search": {"model_params": {
        f"c_{c}_g_{g}": {"kernel": "rbf", "probability": True, "C": c, "class_weight": {0: 0.75, 1: 1.5}, "gamma": g}
        for c in [0.001, 0.01, 0.1, 1, 10, 100, 1000] for g in [0.001, 0.01, 0.1, 1]
    }},
}
