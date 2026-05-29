import numpy as np
import pandas as pd
import os
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, f1_score, precision_score, \
    recall_score

from architecture.data_utils import ConcatMultiViewDataset
from architecture.pipeline import IdentityProcessor
from architecture.evaluation import save_results, save_supervised_result, collate_grid_search, collate_aggregate_results


from prostate_cancer_prediction.preprocess import mut_binary, cnv_del, cnv_amp

wd = "glioma_prediction"
data_dir = f"{wd}/data"
run_dir = f"{wd}/runs"

os.makedirs(run_dir, exist_ok=True)

if not os.path.exists(os.path.join(data_dir, "tcga_labels_preprocessed.csv")):
    from pathlib import Path
    from glioma_prediction.preprocess import Preprocessor
    print("Preprocessed data not found — running preprocessing pipeline...")
    Preprocessor(Path(data_dir)).run_all()

selected_genes = list(set(pd.read_csv(f"{data_dir}/hugo_genes.txt", sep="\t", low_memory=False)["symbol"]))

views = [
    ("mut_important", "tcga_mutations_preprocessed.csv", selected_genes, 0, mut_binary, lambda x: x),
    ("cnv_amp", "tcga_cna_preprocessed.csv", selected_genes, 0, cnv_amp, lambda x: x),
    ("cnv_del", "tcga_cna_preprocessed.csv", selected_genes, 0, cnv_del, lambda x: x),
]

f1_selection = lambda x: f1_score(x["val_df"].ys, (x["val_preds"] >= 0.5).astype(int))
auprc_selection = lambda x: average_precision_score(x["val_df"].ys, x["val_preds"])
auc_selection = lambda x: roc_auc_score(x["val_df"].ys, x["val_preds"])

save_processor = lambda x: save_results(x, save_supervised_result, {
    "auc": roc_auc_score,
    "auprc": average_precision_score,
    "f1": lambda ys, preds: f1_score(ys, (preds > 0.5).astype(int)),
    "accuracy": lambda ys, preds: accuracy_score(ys, (preds > 0.5).astype(int)),
    "precision": lambda ys, preds: precision_score(ys, (preds > 0.5).astype(int)),
    "recall": lambda ys, preds: recall_score(ys, (preds > 0.5).astype(int)),
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
    "labels": [("tcga_labels_preprocessed.csv", 0)],
    "use_validation_on_test": False,
    "val_metric": {"f1": f1_selection, "auprc": auprc_selection, "auc": auc_selection},
    "results_processors": [save_processor],
    "rng_seed": 20080808,
    "tt_split_seed": 20080808,
    "tv_split_seed": 20080808,
    "shuffle_seed": 42,
    "inner_kfolds": 5,
    "outer_kfolds": 5,
    "stratified": True,
    "grid_search": [],
    "fold_collators": [],
    "grid_search_collators": [collate_grid_search, collate_aggregate_results],
    "drop_labels": True,
    "external_datasets": [
        {
            "tag": "cgga",
            "views": [
                ("mut_important", "cgga_mutations_preprocessed.csv", selected_genes, 0, mut_binary, lambda x: x),
                ("cnv_amp", "cgga_cna_preprocessed.csv", selected_genes, 0, cnv_amp, lambda x: x),
                ("cnv_del", "cgga_cna_preprocessed.csv", selected_genes, 0, cnv_del, lambda x: x),
            ],
            "labels": [("cgga_labels_preprocessed.csv", 0)],
        }
    ],
    "external_validation_metrics": {
        "auc": roc_auc_score,
        "auprc": average_precision_score,
        "f1": lambda ys, preds: f1_score(ys, (preds > 0.5).astype(int)),
        "accuracy": lambda ys, preds: accuracy_score(ys, (preds > 0.5).astype(int)),
        "precision": lambda ys, preds: precision_score(ys, (preds > 0.5).astype(int)),
        "recall": lambda ys, preds: recall_score(ys, (preds > 0.5).astype(int)),
    },
}
