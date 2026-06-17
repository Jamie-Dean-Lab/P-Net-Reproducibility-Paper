import copy
from functools import partial

from keras.regularizers import L2
from keras.callbacks import LearningRateScheduler

from architecture.pipeline import TFPipeline
from architecture.pnet_model import compile_pnet
from architecture.callbacks_custom import step_decay
from architecture.evaluation import plot_history, get_deeplift_global
from .base_config import (base_config, data_dir, f1_selection, auprc_selection,
                          auc_selection, save_processor, train_samples, val_samples, test_samples, selected_genes)

n_hidden_layers = 5

learning_rate = 1e-3
step_decay_part = partial(step_decay, init_lr=learning_rate, drop=0.25, epochs_drop=50)

_model_params = {
    "pathway_dataset": "reactome",
    "pp_relations": "architecture/Reactome/ReactomePathwaysRelation.txt",
    "gp_relations": "architecture/Reactome/ReactomePathways.gmt",
    "n_hidden_layers": n_hidden_layers,
    "h_dropout": [0.5] + [0.1] * n_hidden_layers,
    "h_activation": ["tanh"] * (n_hidden_layers + 1),
    "o_activation": ["sigmoid"] * (n_hidden_layers + 1),
    "h_kernel_initializer": ["lecun_uniform"] * (n_hidden_layers + 1),
    "h_kernel_constraints": [None] * (n_hidden_layers + 1),
    "h_bias_initializer": ["lecun_uniform"] * (n_hidden_layers + 1),
    "h_bias_constraints": [None] * (n_hidden_layers + 1),
    "batch_normal": False,
    "sparse": True,
    "h_reg": [(L2, {"l2": 1e-3})] * (n_hidden_layers + 1),
    "o_reg": [(L2, {"l2": 1e-2})] * (n_hidden_layers + 1),
    "dropout_testing": False,
    "loss": [{"class_name": "BinaryCrossentropy", "config": {"from_logits": False}}] * (n_hidden_layers + 1),
    "loss_weights": [2, 7, 20, 54, 148, 400],
    "optimizer": {"class_name": "Adam", "config": {"learning_rate": learning_rate}},
    "map_seed": 42
}

pnet_single_split_elmarakeby_config = {
    **copy.deepcopy(base_config),
    "run_id": "pnet_single_split_elmarakeby",
    "model": compile_pnet,
    "model_params": _model_params,
    "fitting_params": {
        "epochs": 300,
        "batch": 50,
        "LRScheduler": LearningRateScheduler(step_decay_part, verbose=0),
        "early_stopping": None,
        "prediction_output": "average",
        "shuffle_samples": True,
        "class_weight": [{0: 0.75, 1: 1.5}] * (n_hidden_layers + 1),
    },
    "results_processors": [
        save_processor,
        plot_history,
        partial(get_deeplift_global,
                n_hidden_layers=n_hidden_layers,
                pathway_dataset=_model_params["pathway_dataset"],
                pp_relations=_model_params["pp_relations"],
                gp_relations=_model_params["gp_relations"],
                always_run=True)
    ],
    "val_metric": {},
    "pipeline_class": TFPipeline,
    "run_method": "run_single_split",
    "grid_search": []
}
