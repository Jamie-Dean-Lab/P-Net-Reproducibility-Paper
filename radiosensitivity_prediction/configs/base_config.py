import lifelines
import pandas as pd
from functools import partial
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score, \
    root_mean_squared_error

from architecture.data_utils import ConcatMultiViewDataset
from architecture.pipeline import IdentityProcessor
from architecture.evaluation import save_results, save_supervised_result, collate_grid_search, collate_aggregate_results
from architecture.callbacks_custom import step_decay

import os

from radiosensitivity_prediction.preprocess import Preprocessor

wd = "radiosensitivity_prediction"
data_dir = f"{wd}/data"
run_dir = f"{wd}/runs"
_PREPROCESSING_SENTINEL = "nci60_rnaseq_preprocessed.csv"

if not os.path.exists(run_dir):
    os.mkdir(run_dir)

if not os.path.exists(os.path.join(data_dir, _PREPROCESSING_SENTINEL)):
    print("Preprocessed data not found — running preprocessing pipeline...")
    Preprocessor(data_dir).run_all()

selected_genes = list(set(pd.read_csv(f"{data_dir}/hugo_genes.txt", sep="\t", low_memory=False)["symbol"]))

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
    "pearson_r": lambda y, yhat: pearsonr(y, yhat)[0],
    "spearman_r": lambda y, yhat: spearmanr(y, yhat)[0],
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
    "outer_kfolds": 5,
    "hold_out_validation_for_final_fit": False,
    "val_metric": {"r2": r2_selection},
    "hyperparam_selection_metric": "r2",
    "results_processors": [save_processor],
    "grid_search": [],
    "fold_collators": [],
    "grid_search_collators": [collate_grid_search, collate_aggregate_results],
    "drop_labels": True,
    "external_datasets": [
        {
            "tag": "nci60",
            "views": [
                ("gexpr", "nci60_rnaseq_preprocessed.csv", selected_genes, 0, lambda x: x, lambda x: x),
                ("methylation", "nci60_methylation_preprocessed.csv", selected_genes, 0, lambda x: x, lambda x: x),
            ],
            "labels": [("nci60_auc_preprocessed.csv", 0)],
        }
    ],
    "external_validation_metrics": {
        "r2": r2_score,
        "explained_variance": explained_variance_score,
        "pearson_r": lambda y, yhat: pearsonr(y, yhat)[0],
        "spearman_r": lambda y, yhat: spearmanr(y, yhat)[0],
        "mae": mean_absolute_error,
        "mse": mean_squared_error,
        "rmse": root_mean_squared_error,
        "concordance_index": lifelines.utils.concordance_index,
    },
}
