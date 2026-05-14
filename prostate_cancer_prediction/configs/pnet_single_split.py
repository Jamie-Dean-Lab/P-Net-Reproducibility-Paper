import copy
from keras.regularizers import L2
from keras.callbacks import LearningRateScheduler

from architecture.pipeline import TFPipeline
from architecture.pnet_model import compile_pnet
from architecture.callbacks_custom import step_decay_part
from architecture.evaluation import plot_history, get_deeplift_global
from .base_config import (base_config, data_dir, f1_selection, auprc_selection,
                          auc_selection, save_processor, train_samples, val_samples, test_samples, selected_genes)

n_hidden_layers = 5

pnet_single_split_config = {
    **copy.deepcopy(base_config),
    "run_id":                 "pnet_single_split",
    "model":                  compile_pnet,
    "model_params": {
        "pathway_dataset":        "reactome",
        "pp_relations":           "architecture/Reactome/ReactomePathwaysRelation.txt",
        "gp_relations":           "architecture/Reactome/ReactomePathways.gmt",
        "n_hidden_layers":        n_hidden_layers,
        "h_dropout":              [0.5] + [0.1] * n_hidden_layers,
        "h_activation":           ["tanh"] * (n_hidden_layers + 1),
        "o_activation":           ["sigmoid"] * (n_hidden_layers + 1),
        "h_reg":                  [(L2, {"l2": 1e-3})] * (n_hidden_layers + 1),
        "o_reg":                  [(L2, {"l2": 1e-2})] * (n_hidden_layers + 1),
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
        "map_seed":              42
    },
    "fitting_params": {
        "epochs":             300,
        "batch":              50,
        "LRScheduler":        LearningRateScheduler(step_decay_part, verbose=0),
        "early_stopping":     None,
        "prediction_output":  "average",
        "shuffle_samples":    True,
        "class_weight":       [[0.75, 1.5]] * (n_hidden_layers + 1),
    },
    "results_processors":     [save_processor, plot_history, lambda results: get_deeplift_global(results, selected_genes, n_hidden_layers)],
    "use_validation_on_test": True,
    "val_metric":             {"f1": f1_selection, "auprc": auprc_selection, "auc": auc_selection},
    "pipeline_class":         TFPipeline,
    "run_method":             "run_single_split"
}