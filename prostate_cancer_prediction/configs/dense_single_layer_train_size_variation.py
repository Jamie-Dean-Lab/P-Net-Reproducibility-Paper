import copy
from keras.regularizers import L2
from keras.callbacks import LearningRateScheduler

from architecture.pipeline import TFPipeline
from architecture.callbacks_custom import step_decay_part
from architecture.evaluation import collate_folds
from dense_model import compile_dense
from .base_config import (base_config, val_samples, save_processor)
from .pnet_train_size_variation import train_size_samples

n_hidden_layers = 5

_dense_fitting_params = {
    "epochs":             300,
    "batch":              50,
    "LRScheduler":        LearningRateScheduler(step_decay_part, verbose=0),
    "early_stopping":     None,
    "prediction_output":  "average",
    "shuffle_samples":    True,
    "class_weight":       [[0.75, 1.5]] * (n_hidden_layers + 1),
}

_dense_model_params = {
    "h_activation":   "selu",
    "o_activation":   "sigmoid",
    "n_hidden_layers": 0,
    "h_reg":          (L2, {"l2": 1e-3}),
    "n_weights":      71009,
    "loss":           {"class_name": "BinaryCrossentropy", "config": {"from_logits": False}},
    "optimizer":      {"class_name": "Adam", "config": {"learning_rate": 1e-3}},
}

dense_single_layer_train_size_variation_configs = [
    {
        **copy.deepcopy(base_config),
        "run_id":             f"dense_single_layer_train_size_variation_{i}",
        "train_samples":      ts,
        "model":              compile_dense,
        "model_params":       copy.deepcopy(_dense_model_params),
        "fitting_params":     copy.deepcopy(_dense_fitting_params),
        "results_processors": [save_processor],
        "use_validation_on_test": False,
        "val_metric":         {},
        "stratified":         True,
        "inner_kfolds":       5,
        "fold_collators":     [collate_folds],
        "grid_search_collators": [],
        "pipeline_class":     TFPipeline,
        "run_method": "run_crossvalidation"
    }
    for i, ts in enumerate(train_size_samples)
]