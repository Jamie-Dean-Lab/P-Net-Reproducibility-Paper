import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, precision_score, recall_score

from architecture.data_utils import ConcatMultiViewDataset
from architecture.pipeline import IdentityProcessor
from architecture.evaluation import save_results, save_supervised_result, collate_grid_search, collate_aggregate_results
from prostate_cancer_prediction.feature_encoders import mut_binary, cnv_amp, cnv_del
from prostate_cancer_prediction.preprocess import Preprocessor

import os

wd = "prostate_cancer_prediction"
download_dir = f"{wd}/data"
data_dir = f"{download_dir}/_database"
prostate_dir = f"{data_dir}/prostate"
run_dir = f"{wd}/runs"
_PREPROCESSING_SENTINEL = f"{prostate_dir}/processed/response_paper_external_validation_1.csv"

_preprocessor = Preprocessor(prostate_dir)
_preprocessor.download_data()

if not os.path.exists(run_dir):
    os.mkdir(run_dir)

if not os.path.exists(_PREPROCESSING_SENTINEL):
    print("External-validation data not found — running preprocessing pipeline...")
    _preprocessor.run_all()

selected_genes = set(pd.read_csv(f"{data_dir}/genes/tcga_prostate_expressed_genes_and_cancer_genes.csv")["genes"])
hugo_genes = set(pd.read_csv(f"{data_dir}/genes/HUGO_genes/protein-coding_gene_with_coordinate_minimal.txt",
                          sep="\t", header=None).iloc[:, 3].unique())
selected_genes = list(selected_genes.intersection(hugo_genes))


train_samples = pd.read_csv(f"{data_dir}/prostate/splits/training_set.csv")["id"].to_list()
val_samples   = pd.read_csv(f"{data_dir}/prostate/splits/validation_set.csv")["id"].to_list()
test_samples  = pd.read_csv(f"{data_dir}/prostate/splits/test_set.csv")["id"].to_list()

views = [
    ("mut_important", "P1000_final_analysis_set_cross_important_only.csv", selected_genes, 0, mut_binary, lambda x: x),
    ("cnv_amp",       "P1000_data_CNA_paper.csv",                          selected_genes, 0, cnv_amp,    lambda x: x),
    ("cnv_del",       "P1000_data_CNA_paper.csv",                          selected_genes, 0, cnv_del,    lambda x: x),
]

f1_selection   = lambda x: f1_score(x["val_df"].ys, (x["val_preds"] >= 0.5).astype(int))
auprc_selection = lambda x: average_precision_score(x["val_df"].ys, x["val_preds"])
auc_selection  = lambda x: roc_auc_score(x["val_df"].ys, x["val_preds"])

save_processor = lambda x: save_results(x, save_supervised_result, {
    "auc":       roc_auc_score,
    "auprc":     average_precision_score,
    "f1":        lambda ys, preds: f1_score(ys, (preds > 0.5).astype(int)),
    "accuracy":  lambda ys, preds: accuracy_score(ys, (preds > 0.5).astype(int)),
    "precision": lambda ys, preds: precision_score(ys, (preds > 0.5).astype(int)),
    "recall":    lambda ys, preds: recall_score(ys, (preds > 0.5).astype(int)),
}, "individual")

base_config = {
    "dataloader":            ConcatMultiViewDataset,
    "feature_selector":      IdentityProcessor(),
    "feature_preprocessor":  IdentityProcessor(),
    "data_augmentor":        lambda x: x,
    "data_dir":              f"{data_dir}/prostate/processed",
    "run_dir":               run_dir,
    "views":                 views,
    "view_alignment_method": "drop samples",
    "labels":                [("response_paper.csv", 0)],
    "train_samples":         train_samples,
    "val_samples":           val_samples,
    "test_samples":          test_samples,
    "hold_out_validation_for_final_fit": False,
    "val_metric":            {},
    "results_processors":    [],
    "rng_seed":              20080808,
    "tt_split_seed":         20080808,
    "tv_split_seed":         20080808,
    "shuffle_seed":          42,
    "grid_search":           [],
    "fold_collators":        [],
    "grid_search_collators": [collate_grid_search, collate_aggregate_results],
    "drop_labels":           True,
}