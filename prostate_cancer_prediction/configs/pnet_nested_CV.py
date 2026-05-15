import copy
from keras.regularizers import L2
from keras.callbacks import LearningRateScheduler

from architecture.pipeline import TFPipeline
from architecture.pnet_model import compile_pnet
from architecture.callbacks_custom import step_decay_part
from architecture.evaluation import plot_history
from .base_config import (base_config, f1_selection, auprc_selection,
                          auc_selection, save_processor)

n_hidden_layers = 5


_model_params_base = {
        "pathway_dataset":        "reactome",
        "pp_relations":           "architecture/Reactome/ReactomePathwaysRelation.txt",
        "gp_relations":           "architecture/Reactome/ReactomePathways.gmt",
        "n_hidden_layers":        n_hidden_layers,
        "h_dropout":              [0.5] + [0.1] * n_hidden_layers,
        "h_activation":           ["tanh"] * (n_hidden_layers + 1),
        "o_activation":           ["sigmoid"] * (n_hidden_layers + 1),
        "h_kernel_initializer":   ["lecun_uniform"] * (n_hidden_layers + 1),
        "h_kernel_constraints":   [None] * (n_hidden_layers + 1),
        "h_bias_initializer":     ["lecun_uniform"] * (n_hidden_layers + 1),
        "h_bias_constraints":     [None] * (n_hidden_layers + 1),
        "batch_normal":           False,
        "sparse":                 True,
        "dropout_testing":        False,
        "loss":                   [{"class_name": "BinaryCrossentropy", "config": {"from_logits": False}}] * (n_hidden_layers + 1),
        "loss_weights":           [2, 7, 20, 54, 148, 400],
        "optimizer":              {"class_name": "Adam", "config": {"learning_rate": 1e-3}},
        "map_seed":               42
}

pnet_nested_CV_config = {
    **copy.deepcopy(base_config),
    "run_id":                 "pnet_nested_CV",
    "model":                  compile_pnet,
    "task":                   "binary classification",
    "fitting_params": {
        "epochs":             300,
        "batch":              50,
        "LRScheduler":        LearningRateScheduler(step_decay_part, verbose=0),
        "early_stopping":     None,
        "prediction_output":  "average",
        "shuffle_samples":    True,
        "class_weight":       [[0.75, 1.5]] * (n_hidden_layers + 1),
    },
    "results_processors":     [save_processor, plot_history],
    "val_metric":             {"f1": f1_selection, "auprc": auprc_selection, "auc": auc_selection},
    "pipeline_class":         TFPipeline,
    "run_method":             "run_crossvalidation",
    "stratified":              True,
    "inner_kfolds":            5,
    "outer_kfolds":            10,
    "grid_search": {
        "model_params": {
            f"reg_{l}": {**_model_params_base,
                         "h_reg": [(L2, {"l2": 10 ** l})] * (n_hidden_layers + 1),
                         "o_reg": [(L2, {"l2": 10 ** l})] * (n_hidden_layers + 1)}
            for l in [-1, -2, -3, -4]
        }
    },
}
del pnet_nested_CV_config["train_samples"]
del pnet_nested_CV_config["val_samples"]
del pnet_nested_CV_config["test_samples"]

