import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, f1_score

from architecture.data_utils import ConcatMultiViewDataset
from architecture.pipeline import IdentityProcessor
from architecture.evaluation import save_results, save_supervised_result, collate_grid_search, collate_aggregate_results

import os

from tissue_type_classification.preprocess import Preprocessor

wd = "tissue_type_classification"
data_dir = f"{wd}/data"
run_dir = f"{wd}/runs"

_PREPROCESSING_SENTINEL = "GTEx_gene_expression_preprocessed.csv"

if not os.path.exists(run_dir):
    os.mkdir(run_dir)

if not os.path.exists(os.path.join(data_dir, _PREPROCESSING_SENTINEL)):
    print("Preprocessed data not found — running preprocessing pipeline...")
    Preprocessor(data_dir).run_all()

selected_genes = list(set(pd.read_csv(f"{data_dir}/hugo_genes.txt", sep="\t", low_memory=False)["symbol"]))

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
    "inner_kfolds": 5,
    "outer_kfolds": 5,
    "stratified": False,
    "hold_out_validation_for_final_fit": False,
    "val_metric": {"f1": f1_selection},
    "ext_validation_hyperparam_selection_metric": "f1",
    "results_processors": [save_processor],
    "grid_search": [],
    "fold_collators": [],
    "grid_search_collators": [collate_grid_search, collate_aggregate_results],
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
        # AUC/AUPRC require at least one positive example per class, so columns absent from the
        # external dataset (all-zero) are dropped. Predictions for those classes count as
        # mispredictions in the argmax-based metrics below.
        "auc": lambda y, y_hat: roc_auc_score(
            y[:, y.sum(axis=0) > 0], y_hat[:, y.sum(axis=0) > 0], multi_class="ovr", average="micro"
        ),
        "auprc": lambda y, y_hat: average_precision_score(
            y[:, y.sum(axis=0) > 0], y_hat[:, y.sum(axis=0) > 0], average="micro"
        ),
        "f1": lambda ys, preds: f1_score(ys, ((preds >= np.sort(preds, axis=1)[:, [-1]]) & (preds > 0)).astype(int),
                                         average="weighted"),
        "accuracy": lambda ys, preds: accuracy_score(ys,
                                                     ((preds >= np.sort(preds, axis=1)[:, [-1]]) & (preds > 0)).astype(
                                                         int)),
    },
}
