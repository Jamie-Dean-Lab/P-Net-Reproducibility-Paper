import copy
import pandas as pd
from keras.regularizers import L2
from architecture.pipeline import TFPipeline
from architecture.evaluation import plot_history, collate_folds
from .base_config import (base_config, data_dir, val_samples, save_processor)
from .pnet_single_split import pnet_single_split_config
from .pnet_single_split_elmarakeby import pnet_single_split_elmarakeby_config, n_hidden_layers

train_size_samples = [
    pd.read_csv(f"{data_dir}/prostate/splits/training_set_{s}.csv")["id"].to_list() + val_samples
    for s in range(0, 20, 3)
]

_train_size_base = {
    "results_processors":    [save_processor, plot_history],
    "val_metric":            {},
    "stratified":            True,
    "inner_kfolds":          5,
    "fold_collators":        [collate_folds],
    "grid_search_collators": [],
    "pipeline_class":        TFPipeline,
}

pnetfc_train_size_variation_configs = [
    {
        **copy.deepcopy(pnet_single_split_elmarakeby_config),
        **copy.deepcopy(_train_size_base),
        "run_id":        f"pnetfc_train_size_variation_{i}",
        "train_samples": ts,
        "val_samples": [],
        "model_params": {
            **copy.deepcopy(pnet_single_split_elmarakeby_config["model_params"]),
            "sparse": False,
            "h_reg": [(L2, {"l2": 1e-3})] * (n_hidden_layers + 1),
            "o_reg": [(L2, {"l2": 1e-2})] * (n_hidden_layers + 1),
        },
    "run_method": "run_crossvalidation"
    }
    for i, ts in enumerate(train_size_samples)
]