import copy
from functools import partial

from keras.regularizers import L2

from architecture.evaluation import plot_history, get_deeplift_global
from .base_config import save_processor
from .pnet_10_fold_CV_stability import pnet_10_fold_CV_stability_config
from .pnet_single_split_elmarakeby import learning_rate

n_hidden_layers = 5

# Identical to pnet_10_fold_CV_stability, but using the GO pathway hierarchy
# instead of Reactome (matching pnet_GO_single_split's pathway dataset).
_go_model_params = {
    "pathway_dataset":        "go",
    "pp_relations":           "architecture/GO/go_hierarchy.tsv",
    "gp_relations":           "architecture/GO/go_gene_sets.gmt",
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
    "optimizer":              {"class_name": "Adam", "config": {"learning_rate": learning_rate}},
    "map_seed":               42
}

pnet_GO_10_fold_CV_stability_config = {
    **copy.deepcopy(pnet_10_fold_CV_stability_config),
    "run_id":                 "pnet_GO_10_fold_CV_stability",
    "model_params":           _go_model_params,
    "results_processors": [
        save_processor,
        plot_history,
        partial(get_deeplift_global,
                n_hidden_layers=n_hidden_layers,
                pathway_dataset=_go_model_params["pathway_dataset"],
                pp_relations=_go_model_params["pp_relations"],
                gp_relations=_go_model_params["gp_relations"])
    ],
    "grid_search": {
        "model_params": {
            f"h_reg_{h}_o_reg_{o}": {**_go_model_params,
                                     "h_reg": [(L2, {"l2": h})] * (n_hidden_layers + 1),
                                     "o_reg": [(L2, {"l2": o})] * (n_hidden_layers + 1)}
            for h in [1e-3]
            for o in [1e-5]
        },
    }
}