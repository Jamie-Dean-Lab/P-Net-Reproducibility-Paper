import copy

from architecture.evaluation import plot_history
from .base_config import save_processor
from .pnet_single_split_elmarakeby import pnet_single_split_elmarakeby_config

# Network-order variation: keep the train/val/test split and all hyperparameters
# fixed, and fix the network construction seed (map_seed) and input variable order (shuffle_seed) to verify
# that there is no variance in the results.
seeds = [42, 42, 42, 42, 42]

pnet_network_order_fixed_seed_configs = [
    {
        **copy.deepcopy(pnet_single_split_elmarakeby_config),
        "run_id":             f"pnet_network_order_fixed_seed_{i}",
        "results_processors": [save_processor, plot_history],
        "grid_search":        [],
        "val_metric":         {},
        "run_method":         "run_single_split",
        "model_params": {
            **copy.deepcopy(pnet_single_split_elmarakeby_config["model_params"]),
            "map_seed": seed,
        },
        "shuffle_seed": seed
    }
    for i, seed in enumerate(seeds)
]
