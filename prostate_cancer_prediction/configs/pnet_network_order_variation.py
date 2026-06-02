import copy
import pandas as pd
from architecture.pipeline import TFPipeline
from architecture.evaluation import plot_history, collate_folds
from .base_config import (base_config, data_dir, val_samples, save_processor)
from .pnet_single_split import pnet_single_split_config
from .pnet_single_split_elmarakeby import pnet_single_split_elmarakeby_config

shuffle_seeds = [203928, 84954, 603492, 1023924, 72832934]

_train_size_base = {
    "results_processors":    [save_processor, plot_history],
    "val_metric":            {},
    "stratified":            True,
    "inner_kfolds":          5,
    "fold_collators":        [collate_folds],
    "grid_search_collators": [],
    "pipeline_class":        TFPipeline,
}

pnet_network_order_variation_configs = [
    {
        **copy.deepcopy(pnet_single_split_elmarakeby_config),
        **copy.deepcopy(_train_size_base),
        "run_id":                        f"pnet_network_order_variation_{i}",
        "shuffle_seed" : seed,
        "val_samples": [],
        "model_params": {
            **copy.deepcopy(pnet_single_split_elmarakeby_config["model_params"]),
            "sparse": True,
            "map_seed" : seed
        },
        "run_method": "run_crossvalidation"
    }
    for i, seed in enumerate(shuffle_seeds)
]

