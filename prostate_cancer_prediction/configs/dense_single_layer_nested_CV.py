import copy
from functools import partial

from keras.regularizers import L2
from keras.callbacks import LearningRateScheduler

from architecture.pipeline import TFPipeline
from architecture.callbacks_custom import step_decay
from architecture.dense_model import compile_dense
from .base_config import save_processor
from .nested_CV_base_config import nested_CV_base_config

n_hidden_layers = 0

learning_rate = 1e-3
step_decay_part = partial(step_decay, init_lr=learning_rate, drop=0.25, epochs_drop=50)

_dense_model_params_base = {
    "h_activation":    "selu",
    "o_activation":    "sigmoid",
    "n_hidden_layers": n_hidden_layers,
    "n_weights":       71009,
    "loss":            {"class_name": "BinaryCrossentropy", "config": {"from_logits": False}},
    "optimizer":       {"class_name": "Adam", "config": {"learning_rate": learning_rate}},
}

dense_single_layer_nested_CV_config = {
    **copy.deepcopy(nested_CV_base_config),
    "run_id":             "dense_single_layer_nested_CV",
    "model":              compile_dense,
    "fitting_params": {
        "epochs":             300,
        "batch":              50,
        "LRScheduler":        LearningRateScheduler(step_decay_part, verbose=0),
        "early_stopping":     None,
        "prediction_output":  "average",
        "shuffle_samples":    True,
        "class_weight":       [[0.75, 1.5]] * (n_hidden_layers + 1),
    },
    "results_processors": [save_processor],
    "pipeline_class":     TFPipeline,
    "grid_search": {
        "model_params": {
            f"h_reg_{h}": {**copy.deepcopy(_dense_model_params_base), "h_reg": (L2, {"l2": h})}
            for h in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
        },
    }
}
