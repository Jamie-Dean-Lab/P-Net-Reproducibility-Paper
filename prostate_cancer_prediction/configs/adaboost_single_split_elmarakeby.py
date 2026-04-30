import copy
from sklearn.ensemble import AdaBoostClassifier

from architecture.pipeline import MLPipeline
from .base_config import (base_config, f1_selection, auprc_selection, auc_selection, save_processor)

adaboost_single_split_elmarakeby_config = {
    **copy.deepcopy(base_config),
    "run_id":                 "adaboost_single_split_elmarakeby",
    "model":                  AdaBoostClassifier,
    "task":                   "binary classification",
    "pipeline_class":         MLPipeline,
    "results_processors":     [save_processor],
    "use_validation_on_test": True,
    "val_metric":             {"f1": f1_selection},
    "grid_search":            {"model_params": {
        f"lr_{l}_estimators_{n}": {"learning_rate": l, "n_estimators": n}
        for l in [0.1] for n in [50]
    }},
    "run_method":             "run_single_split"
}