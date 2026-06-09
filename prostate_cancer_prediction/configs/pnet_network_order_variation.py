import copy
from functools import partial

from architecture.evaluation import plot_history, get_deeplift_global
from .base_config import save_processor
from .pnet_single_split_elmarakeby import pnet_single_split_elmarakeby_config, n_hidden_layers, _model_params

# Network-order variation: keep the train/val/test split and all hyperparameters
# fixed, and vary ONLY the network construction seed (map_seed) so any change in
# performance is attributable to the pathway/gene ordering of the network alone.
seeds = [203928, 84954, 603492, 1023924, 72832934, 55464, 123454, 99854, 456134, 115549,
         781233, 9123, 4456721, 33812, 670091, 2210984, 58123, 99012, 1500321, 7788123]

pnet_network_order_variation_configs = [
    {
        **copy.deepcopy(pnet_single_split_elmarakeby_config),
        "run_id": f"pnet_network_order_variation_{i}",
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
        "grid_search": [],
        "val_metric": {},
        "run_method": "run_single_split",
        "model_params": {
            **copy.deepcopy(pnet_single_split_elmarakeby_config["model_params"]),
            "map_seed": seed,
        },
        "shuffle_seed": seed
    }
    for i, seed in enumerate(seeds)
]
