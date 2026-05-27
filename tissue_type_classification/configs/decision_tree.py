import copy
from sklearn.tree import DecisionTreeClassifier

from architecture.pipeline import MLPipeline
from .base_config import base_config, save_processor
from ..wrapper import ProbaWrapper

decision_tree_config = {
    **copy.deepcopy(base_config),
    "run_id":             "decision_tree",
    "model":              ProbaWrapper,
    "task":               "multiclass",
    "results_processors": [save_processor],
    "pipeline_class":     MLPipeline,
    "run_method":         "run_crossvalidation",
    "grid_search": {
        "model_params": {
            f"depth_{d}_mss_{mss}": {
                "estimator": DecisionTreeClassifier,
                "args": {"max_depth": d, "min_samples_split": mss, "random_state": 42},
            }
            for d in range(1, 20, 2)
            for mss in range(10, 500, 20)
        }
    },
}
