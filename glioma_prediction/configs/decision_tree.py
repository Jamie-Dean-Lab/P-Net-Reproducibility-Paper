import copy

from sklearn.tree import DecisionTreeClassifier

from architecture.pipeline import MLPipeline
from .base_config import base_config

decision_tree_config = {
    **copy.deepcopy(base_config),
    "run_id": "decision_tree",
    "model": DecisionTreeClassifier,
    "pipeline_class": MLPipeline,
    "task": "binary classification",
    "run_method": "run_crossvalidation",
    "grid_search": {
        "model_params": {
            f"ssplit_{s}_depth_{d}": {"min_samples_split": s, "max_depth": d}
            for s in range(10, 500, 20) for d in range(1, 20, 2)
        }},
}
