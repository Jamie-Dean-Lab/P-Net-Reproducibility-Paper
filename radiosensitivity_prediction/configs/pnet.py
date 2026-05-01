import copy
from keras.regularizers import L2
from keras.callbacks import LearningRateScheduler

from architecture.pipeline import TFPipeline
from architecture.pnet_model import compile_pnet
from .base_config import base_config, step_decay_part

n_hidden_layers = 5

_model_params_base = {
    "pp_relations":           "architecture/Reactome/ReactomePathwaysRelation.txt",
    "gp_relations":           "architecture/Reactome/ReactomePathways.gmt",
    "n_hidden_layers":        n_hidden_layers,
    "h_dropout":              [0.5] + [0.1] * n_hidden_layers,
    "h_activation":           ["tanh"] * (n_hidden_layers + 1),
    "o_activation":           ["linear"] * (n_hidden_layers + 1),
    "h_kernel_initializer":   ["lecun_uniform"] * (n_hidden_layers + 1),
    "h_kernel_constraints":   [None] * (n_hidden_layers + 1),
    "h_bias_initializer":     ["lecun_uniform"] * (n_hidden_layers + 1),
    "h_bias_constraints":     [None] * (n_hidden_layers + 1),
    "batch_normal":           False,
    "sparse":                 True,
    "dropout_testing":        False,
    "loss":                   ["MeanSquaredError"] * (n_hidden_layers + 1),
    "loss_weights":           [2, 7, 20, 54, 148, 400],
    "optimizer":              {"class_name": "Adam", "config": {"learning_rate": 1e-3}},
}

_fitting_params = {
    "epochs":             200,
    "batch":              50,
    "LRScheduler":        LearningRateScheduler(step_decay_part, verbose=0),
    "early_stopping":     None,
    "prediction_output":  "average",
    "shuffle_samples":    True,
    "class_weight":       None,
}

pnet_config = {
    **copy.deepcopy(base_config),
    "run_id":        "pnet",
    "model":         compile_pnet,
    "fitting_params": _fitting_params,
    "pipeline_class": TFPipeline,
    "run_method":    "run_crossvalidation",
    "grid_search": {
        "model_params": {
            f"reg_{l}": {**_model_params_base, "sparse": True,
                         "h_reg": [(L2, {"l2": 10 ** l})] * (n_hidden_layers + 1),
                         "o_reg": [(L2, {"l2": 10 ** l})] * (n_hidden_layers + 1)}
            for l in [-1, -2, -3, -4]
        }
    },
}