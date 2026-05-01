import copy
from sklearn.kernel_ridge import KernelRidge

from architecture.pipeline import MLPipeline
from .base_config import base_config

krr_config = {
    **copy.deepcopy(base_config),
    "run_id":         "krr",
    "model":          KernelRidge,
    "task":           "regression",
    "pipeline_class": MLPipeline,
    "run_method":     "run_crossvalidation",
    "grid_search": {
        "model_params": {
            f"degree_{d}_alpha_{a}": {"kernel": "poly", "degree": d, "alpha": a}
            for d in [1, 2, 3]
            for a in [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
        }
    },
}