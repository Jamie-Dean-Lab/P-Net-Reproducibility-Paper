import copy

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, precision_score, \
    recall_score

from .base_config import base_config, f1_selection, auprc_selection, auc_selection, selected_genes
from ..feature_encoders import mut_binary, cnv_amp, cnv_del

# Common settings shared by every nested cross-validation config (both the
# sklearn baselines and the P-NET variants). Model-specific keys (model,
# pipeline_class, results_processors, grid_search, fitting_params, run_id) are
# set by the individual configs that inherit this.
nested_CV_base_config = {
    **copy.deepcopy(base_config),
    "task": "binary classification",
    "val_metric": {"f1": f1_selection, "auprc": auprc_selection, "auc": auc_selection},
    "stratified": True,
    "inner_kfolds": 5,
    "outer_kfolds": 5,
    "run_method": "run_crossvalidation",
    "ext_validation_hyperparam_selection_metric": "auc",
    "external_datasets": [
        {
            "tag": "combined",
            "views": [
                ("mut_important", "../external_validation/combined/mut_combined.csv", selected_genes, 0,
                 mut_binary, lambda x: x),
                ("cnv_amp", "../external_validation/combined/cnv_combined.csv", selected_genes, 0, cnv_amp,
                 lambda x: x),
                ("cnv_del", "../external_validation/combined/cnv_combined.csv", selected_genes, 0, cnv_del,
                 lambda x: x),
            ],
            "labels": [("../external_validation/combined/labels_combined.csv", 0)],
        },
    ],
    "external_validation_metrics": {
        "auc": roc_auc_score,
        "auprc": average_precision_score,
        "f1": lambda ys, preds: f1_score(ys, (preds > 0.5).astype(int)),
        "accuracy": lambda ys, preds: accuracy_score(ys, (preds > 0.5).astype(int)),
        "precision": lambda ys, preds: precision_score(ys, (preds > 0.5).astype(int)),
        "recall": lambda ys, preds: recall_score(ys, (preds > 0.5).astype(int))
    }
}

# Nested CV generates its own train/val/test splits, so the fixed sample lists
# inherited from base_config are not used.
for _samples_key in ("train_samples", "val_samples", "test_samples"):
    nested_CV_base_config.pop(_samples_key, None)
