import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, f1_score

from architecture.data_utils import ConcatMultiViewDataset
from architecture.pipeline import IdentityProcessor
from architecture.evaluation import save_results, save_supervised_result, collate_grid_search

import os

wd = "tissue_type_classification"
download_dir = f"{wd}/data"
data_dir = f"{download_dir}"
run_dir = f"{wd}/runs"

if not os.path.exists(download_dir):
    with open(f"{wd}/download_data.py") as file:
        exec(file.read())

selected_genes = list(set(pd.read_csv(f"{download_dir}/hugo_genes.txt", sep="\t")["symbol"]))

views = [
    ("gexpr", "GTEx_gene_expression_preprocessed.csv", selected_genes, 0, lambda x: x, lambda x: x),
]

f1_selection = lambda x: f1_score(
    x["val_df"].ys,
    (x["val_preds"] >= np.sort(x["val_preds"], axis=1)[:, [-1]]).astype(int),
    average="weighted"
)

save_processor = lambda x: save_results(x, save_supervised_result, {
    "auc": lambda y, y_hat: roc_auc_score(y, y_hat, multi_class="ovr", average="micro"),
    "auprc": lambda y, y_hat: average_precision_score(y, y_hat, average="micro"),
    "f1": lambda ys, preds: f1_score(ys, (preds >= np.sort(preds, axis=1)[:, [-1]]).astype(int), average="weighted"),
    "accuracy": lambda ys, preds: accuracy_score(ys, (preds >= np.sort(preds, axis=1)[:, [-1]]).astype(int)),
}, "group")

save_processor_svc = lambda x: save_results(x, save_supervised_result, {
    "auc": lambda y, y_hat: roc_auc_score(y, y_hat, multi_class="ovr", average="micro"),
    "auprc": lambda y, y_hat: average_precision_score(y, y_hat, average="micro"),
    "f1": lambda ys, preds: f1_score(ys, ((preds >= np.sort(preds, axis=1)[:, [-1]]) & (preds > 0)).astype(int),
                                     average="weighted"),
    "accuracy": lambda ys, preds: accuracy_score(ys, ((preds >= np.sort(preds, axis=1)[:, [-1]]) & (preds > 0)).astype(
        int)),
}, "group")

base_config = {
    "dataloader": ConcatMultiViewDataset,
    "feature_selector": IdentityProcessor(),
    "feature_preprocessor": IdentityProcessor(),
    "data_augmentor": lambda x: x,
    "data_dir": data_dir,
    "run_dir": run_dir,
    "views": views,
    "view_alignment_method": "drop samples",
    "labels": [("GTEx_tissue_classes_encoded.csv", 0)],
    "tv_split_seed": 42,
    "rng_seed": 42,
    "tt_split_seed": 42,
    "shuffle_seed": 42,
    "inner_kfolds": 2,
    "outer_kfolds": 2,
    "stratified": False,
    "use_validation_on_test": False,
    "val_metric": {"f1": f1_selection},
    "results_processors": [save_processor],
    "grid_search": [],
    "fold_collators": [],
    "grid_search_collators": [collate_grid_search],
    "drop_labels": True,
    "external_datasets": [
        {
            "tag": "hpa",
            "views": [
                ("gexpr", "hpa_gene_expression_preprocessed.csv", selected_genes, 0, lambda x: x, lambda x: x),
            ],
            "labels": [("hpa_tissue_classes_encoded.csv", 0)],
        }
    ],
    "external_validation_task": "group",
    "external_validation_metrics": {
        "auc": lambda y, y_hat: roc_auc_score(y, y_hat, multi_class="ovr", average="micro"),
        "auprc": lambda y, y_hat: average_precision_score(y, y_hat, average="micro"),
        "f1": lambda ys, preds: f1_score(ys, ((preds >= np.sort(preds, axis=1)[:, [-1]]) & (preds > 0)).astype(int),
                                         average="weighted"),
        "accuracy": lambda ys, preds: accuracy_score(ys,
                                                     ((preds >= np.sort(preds, axis=1)[:, [-1]]) & (preds > 0)).astype(
                                                         int)),
    },
}
