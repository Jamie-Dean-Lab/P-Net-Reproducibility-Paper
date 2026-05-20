import copy
from keras.regularizers import L2
from keras.callbacks import LearningRateScheduler

from architecture.pipeline import TFPipeline
from architecture.pnet_model import compile_pnet
from .base_config import base_config, step_decay_part

n_hidden_layers = 5

_model_params_base = {
    "pathway_dataset": "reactome",
    "pp_relations": "architecture/Reactome/ReactomePathwaysRelation.txt",
    "gp_relations": "architecture/Reactome/ReactomePathways.gmt",
    "n_hidden_layers": n_hidden_layers,
    "h_dropout": [0.5] + [0.1] * n_hidden_layers,
    "h_activation": ["tanh"] * (n_hidden_layers + 1),
    "o_activation": ["linear"] * (n_hidden_layers + 1),
    "h_kernel_initializer": ["lecun_uniform"] * (n_hidden_layers + 1),
    "h_kernel_constraints": [None] * (n_hidden_layers + 1),
    "h_bias_initializer": ["lecun_uniform"] * (n_hidden_layers + 1),
    "h_bias_constraints": [None] * (n_hidden_layers + 1),
    "batch_normal": False,
    "sparse": True,
    "dropout_testing": False,
    "loss": ["MeanSquaredError"] * (n_hidden_layers + 1),
    "loss_weights": [2, 7, 20, 54, 148, 400],
    "optimizer": {"class_name": "Adam", "config": {"learning_rate": 1e-3}},
    "map_seed": 42
}

_fitting_params = {
    "epochs": 300,
    "batch": 50,
    "LRScheduler": LearningRateScheduler(step_decay_part, verbose=0),
    "early_stopping": None,
    "prediction_output": "average",
    "shuffle_samples": True,
    "class_weight": None,
}

pnet_config = {
    **copy.deepcopy(base_config),
    "run_id": "pnet",
    "model": compile_pnet,
    "fitting_params": _fitting_params,
    "pipeline_class": TFPipeline,
    "run_method": "run_crossvalidation",
    "grid_search": {
        "model_params": {
            f"h_reg_{h}_o_reg_{o}": {**_model_params_base,
                                     "h_reg": [(L2, {"l2": h})] * (n_hidden_layers + 1),
                                     "o_reg": [(L2, {"l2": o})] * (n_hidden_layers + 1)}
            for h in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
            for o in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
        },
    },
}
