import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, f1_score

from architecture.data_utils import ConcatMultiViewDataset
from architecture.pipeline import IdentityProcessor
from architecture.evaluation import save_results, save_supervised_result, collate_grid_search, collate_aggregate_results

import os

wd = "glioma_prediction"
data_dir = f"{wd}/data"
run_dir = f"{wd}/runs"

_PREPROCESSING_SENTINEL = "response.csv"

if not os.path.exists(run_dir):
    os.mkdir(run_dir)

if not os.path.exists(os.path.join(data_dir, _PREPROCESSING_SENTINEL)):
    print("Preprocessed data not found — running download and preprocessing pipeline...")
    with open(f"{wd}/download_data.py") as _f:
        exec(_f.read())
    with open(f"{wd}/preprocess.py") as _f:
        exec(_f.read())

selected_genes = list(set(pd.read_csv(f"{data_dir}/hugo_genes.txt", sep="\t", low_memory=False)["symbol"]))

views = [
    ("mut",   "mutations.csv", selected_genes, 0, lambda x: x,               lambda x: x),
    ("gexpr", "gexpr.csv",     selected_genes, 0, lambda x: np.log2(x + 1),  lambda x: x),
    ("cna",   "cnas.csv",      selected_genes, 0, lambda x: x,               lambda x: x),
]

auc_selection = lambda x: roc_auc_score(x["val_df"].ys, x["val_preds"])

save_processor = lambda x: save_results(x, save_supervised_result, {
    "auc":      roc_auc_score,
    "auprc":    average_precision_score,
    "f1":       lambda ys, preds: f1_score(ys, (preds > 0.5).astype(int)),
    "accuracy": lambda ys, preds: accuracy_score(ys, (preds > 0.5).astype(int)),
}, "individual")

base_config = {
    "dataloader":             ConcatMultiViewDataset,
    "feature_selector":       IdentityProcessor(),
    "feature_preprocessor":   IdentityProcessor(),
    "data_augmentor":         lambda x: x,
    "data_dir":               data_dir,
    "run_dir":                run_dir,
    "views":                  views,
    "view_alignment_method":  "zero fill",
    "labels":                 [("response.csv", 0)],
    "tv_split_seed":          42,
    "rng_seed":               42,
    "tt_split_seed":          42,
    "shuffle_seed":           42,
    "stratified":             False,
    "inner_kfolds":           1,
    "outer_kfolds":           5,
    "validation_prop":        0.1,
    "use_validation_on_test": False,
    "val_metric":             {"auc": auc_selection},
    "results_processors":     [save_processor],
    "grid_search":            [],
    "fold_collators":         [],
    "grid_search_collators":  [collate_grid_search, collate_aggregate_results],
    "drop_labels":            True,
}
