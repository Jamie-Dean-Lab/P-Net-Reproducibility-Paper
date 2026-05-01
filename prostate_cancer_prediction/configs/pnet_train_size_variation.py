import copy
import pandas as pd
from architecture.pipeline import TFPipeline
from architecture.evaluation import plot_history, collate_folds
from .base_config import (base_config, data_dir, val_samples, save_processor)
from .pnet_single_split import pnet_single_split_config

train_size_samples = [
    pd.read_csv(f"{data_dir}/prostate/splits/training_set_{s}.csv")["id"].to_list() + val_samples
    for s in range(0, 20, 3)
]

_train_size_base = {
    "results_processors":    [save_processor, plot_history],
    "use_validation_on_test": False,
    "val_metric":            {},
    "stratified":            True,
    "inner_kfolds":          5,
    "fold_collators":        [collate_folds],
    "grid_search_collators": [],
    "pipeline_class":        TFPipeline,
}

pnet_train_size_variation_configs = [
    {
        **copy.deepcopy(pnet_single_split_config),
        **copy.deepcopy(_train_size_base),
        "run_id":                        f"pnet_train_size_variation_{i}",
        "train_samples":                 ts,
        "model_params": {
            **copy.deepcopy(pnet_single_split_config["model_params"]),
            "sparse": True,
        },
        "run_method": "run_crossvalidation"
    }
    for i, ts in enumerate(train_size_samples)
]

