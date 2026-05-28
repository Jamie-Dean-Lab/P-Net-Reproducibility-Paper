import copy

from keras.regularizers import L2

from architecture.pipeline import TFPipeline
from architecture.pnet_model import compile_pnet
from architecture.evaluation import plot_history
from .base_config import base_config, auc_selection, save_processor

n_hidden_layers = 5

_model_params = {
    "pathway_dataset":       "reactome",
    "pp_relations":          "architecture/Reactome/ReactomePathwaysRelation.txt",
    "gp_relations":          "architecture/Reactome/ReactomePathways.gmt",
    "n_hidden_layers":       n_hidden_layers,
    "h_dropout":             [0.6] * (n_hidden_layers + 1),
    "h_activation":          ["relu", "tanh", "leaky_relu", "tanh", "leaky_relu", "tanh"],
    "o_activation":          ["sigmoid"] * (n_hidden_layers + 1),
    "h_reg":                 [(L2, {"l2": 1})] * (n_hidden_layers + 1),
    "o_reg":                 [(L2, {"l2": 1e-1})] * (n_hidden_layers + 1),
    "h_kernel_initializer":  ["lecun_uniform"] * (n_hidden_layers + 1),
    "h_kernel_constraints":  [None] * (n_hidden_layers + 1),
    "h_bias_initializer":    ["lecun_uniform"] * (n_hidden_layers + 1),
    "h_bias_constraints":    [None] * (n_hidden_layers + 1),
    "batch_normal":          False,
    "sparse":                True,
    "dropout_testing":       False,
    "loss":                  [{"class_name": "BinaryCrossentropy", "config": {"from_logits": False}}] * (n_hidden_layers + 1),
    "loss_weights":          [2, 7, 20, 54, 148, 400],
    "optimizer":             {"class_name": "Adam", "config": {"learning_rate": 1e-3}},
    "map_seed":              42,
}

pnet_config = {
    **copy.deepcopy(base_config),
    "run_id":          "pnet",
    "model":           compile_pnet,
    "model_params":    _model_params,
    "fitting_params":  {
        "epochs":              200,
        "batch":               100,
        "LRScheduler":         None,
        "early_stopping":      None,
        "prediction_output":   "average",
        "shuffle_samples":     True,
        "class_weight":        None,
    },
    "results_processors": [save_processor, plot_history],
    "val_metric":      {"auc": auc_selection},
    "pipeline_class":  TFPipeline,
    "run_method":      "run_crossvalidation",
}
