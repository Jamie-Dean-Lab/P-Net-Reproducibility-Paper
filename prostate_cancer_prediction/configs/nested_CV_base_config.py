import copy

from .base_config import base_config, f1_selection, auprc_selection, auc_selection

# Common settings shared by every nested cross-validation config (both the
# sklearn baselines and the P-NET variants). Model-specific keys (model,
# pipeline_class, results_processors, grid_search, fitting_params, run_id) are
# set by the individual configs that inherit this.
nested_CV_base_config = {
    **copy.deepcopy(base_config),
    "task":          "binary classification",
    "val_metric":    {"f1": f1_selection, "auprc": auprc_selection, "auc": auc_selection},
    "stratified":    True,
    "inner_kfolds":  5,
    "outer_kfolds":  5,
    "run_method":    "run_crossvalidation",
}

# Nested CV generates its own train/val/test splits, so the fixed sample lists
# inherited from base_config are not used.
for _samples_key in ("train_samples", "val_samples", "test_samples"):
    nested_CV_base_config.pop(_samples_key, None)
