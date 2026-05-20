import lifelines
import pandas as pd
from functools import partial
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score, \
    root_mean_squared_error

from architecture.data_utils import ConcatMultiViewDataset
from architecture.pipeline import IdentityProcessor
from architecture.evaluation import save_results, save_supervised_result, collate_grid_search
from architecture.callbacks_custom import step_decay

import os

wd = "radiosensitivity_prediction"
data_dir = f"{wd}/data"
run_dir = f"{wd}/runs"

if not os.path.exists(data_dir):
    with open(f"{wd}/download_data.py") as file:
        exec(file.read())

selected_genes = list(set(pd.read_csv(f"{data_dir}/hugo_genes.txt", sep="\t")["symbol"]))

views = [
    ("gexpr", "ccle_gene_expression_preprocessed.csv", selected_genes, 0, lambda x: x, lambda x: x),
    ("methylation", "methylation_preprocessed.csv", selected_genes, 0, lambda x: x, lambda x: x),
]

step_decay_part = partial(step_decay, init_lr=0.001, drop=0.5, epochs_drop=25)

r2_selection = lambda x: r2_score(x["val_df"].ys, x["val_preds"])

save_processor = lambda x: save_results(x, save_supervised_result, {
    "r2": r2_score,
    "explained_variance": explained_variance_score,
    "mse": mean_squared_error,
    "rmse": root_mean_squared_error,
    "mae": mean_absolute_error,
    "concordance_index": lifelines.utils.concordance_index
}, "individual")

base_config = {
    "dataloader": ConcatMultiViewDataset,
    "feature_selector": IdentityProcessor(),
    "feature_preprocessor": IdentityProcessor(),
    "data_augmentor": lambda x: x,
    "data_dir": data_dir,
    "run_dir": run_dir,
    "views": views,
    "view_alignment_method": "drop samples",
    "labels": [("cleveland_auc_preprocessed.csv", 0)],
    "tv_split_seed": 42,
    "rng_seed": 42,
    "tt_split_seed": 42,
    "shuffle_seed": 42,
    "stratified": False,
    "inner_kfolds": 5,
    "outer_kfolds": 10,
    "use_validation_on_test": False,
    "val_metric": {"r2": r2_selection},
    "results_processors": [save_processor],
    "grid_search": [],
    "fold_collators": [],
    "grid_search_collators": [collate_grid_search],
    "drop_labels": True,
}
