import copy

from sklearn.tree import DecisionTreeClassifier

from architecture.pipeline import MLPipeline
from .base_config import save_processor
from .nested_CV_base_config import nested_CV_base_config

decision_tree_nested_CV_config = {
    **copy.deepcopy(nested_CV_base_config),
    "run_id":                    "decision_tree_nested_CV",
    "model":                     DecisionTreeClassifier,
    "pipeline_class":            MLPipeline,
    "results_processors":        [save_processor],
    "grid_search":            {"model_params": {
        f"ssplit_{s}_depth_{d}": {"min_samples_split": s, "max_depth": d, "class_weight": {0: 0.75, 1: 1.5}}
        for s in range(10, 500, 20) for d in range(1, 20, 2)
    }},
}
