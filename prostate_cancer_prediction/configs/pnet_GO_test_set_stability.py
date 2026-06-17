import copy
from functools import partial

from keras.regularizers import L2

from architecture.evaluation import plot_history, get_deeplift_global
from .base_config import save_processor
from .pnet_test_set_stability import pnet_test_set_stability_configs
from .pnet_single_split_elmarakeby import _model_params, n_hidden_layers

# Identical to pnet_test_set_stability_configs in every respect (same 100 test-set
# seeds, split proportions, fitting params and run method) — the ONLY difference is
# the model config: the GO pathway hierarchy instead of Reactome, with GO's optimal
# regularization (h_reg=1e-3, o_reg=1e-5).
_go_model_params = {
    **copy.deepcopy(_model_params),
    "pathway_dataset": "go",
    "pp_relations":    "architecture/GO/go_hierarchy.tsv",
    "gp_relations":    "architecture/GO/go_gene_sets.gmt",
    "h_reg":           [(L2, {"l2": 1e-3})] * (n_hidden_layers + 1),
    "o_reg":           [(L2, {"l2": 1e-5})] * (n_hidden_layers + 1),
}

pnet_GO_test_set_stability_configs = [
    {
        **copy.deepcopy(config),
        "run_id":             f"pnet_GO_test_set_stability_{i}",
        "model_params":       copy.deepcopy(_go_model_params),
        # DeepLIFT importance must be mapped through the GO hierarchy, not Reactome.
        "results_processors": [
            save_processor,
            plot_history,
            partial(get_deeplift_global,
                    n_hidden_layers=n_hidden_layers,
                    pathway_dataset=_go_model_params["pathway_dataset"],
                    pp_relations=_go_model_params["pp_relations"],
                    gp_relations=_go_model_params["gp_relations"],
                    always_run=True),
        ],
    }
    for i, config in enumerate(pnet_test_set_stability_configs)
]
