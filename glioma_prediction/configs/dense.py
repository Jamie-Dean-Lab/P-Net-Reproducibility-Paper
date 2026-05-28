import copy

from architecture.pipeline import TFPipeline
from architecture.pnet_model import compile_pnet
from architecture.evaluation import plot_history
from .base_config import base_config, auc_selection, save_processor
from .pnet import _model_params, n_hidden_layers

dense_config = {
    **copy.deepcopy(base_config),
    "run_id":         "dense",
    "model":          compile_pnet,
    "model_params":   {**_model_params, "sparse": False},
    "fitting_params": {
        "epochs":            200,
        "batch":             100,
        "LRScheduler":       None,
        "early_stopping":    None,
        "prediction_output": "average",
        "shuffle_samples":   True,
        "class_weight":      None,
    },
    "results_processors": [save_processor, plot_history],
    "val_metric":     {"auc": auc_selection},
    "pipeline_class": TFPipeline,
    "run_method":     "run_crossvalidation",
}
