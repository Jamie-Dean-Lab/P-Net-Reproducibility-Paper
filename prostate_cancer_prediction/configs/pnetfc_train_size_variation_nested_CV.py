import copy
import pandas as pd

from architecture.pipeline import TFPipeline
from architecture.evaluation import plot_history, collate_folds
from .base_config import (data_dir, val_samples, test_samples, save_processor, auc_selection)
from .pnetfc_single_split import pnetfc_single_split_config

train_size_samples = [
    pd.read_csv(f"{data_dir}/prostate/splits/training_set_{s}.csv")["id"].to_list() + val_samples + test_samples
    for s in range(0, 20, 3)
]

_train_size_nested_CV_base = {
    "results_processors":    [save_processor, plot_history],
    "val_metric":            {"auc": auc_selection},
    "stratified":            True,
    "outer_kfolds":          5,
    "inner_kfolds":          5,
    "fold_collators":        [collate_folds],
    "pipeline_class":        TFPipeline,
}

pnetfc_train_size_variation_nested_CV_configs = []
for _i, _ts in enumerate(train_size_samples):
    _cfg = {
        **copy.deepcopy(pnetfc_single_split_config),
        **copy.deepcopy(_train_size_nested_CV_base),
        "run_id":             f"pnetfc_train_size_variation_nested_CV_{_i}",
        "samples_to_include": _ts,
        "val_samples":        [],
        "run_method":         "run_crossvalidation",
    }
    _cfg.pop("test_samples", None)
    pnetfc_train_size_variation_nested_CV_configs.append(_cfg)
