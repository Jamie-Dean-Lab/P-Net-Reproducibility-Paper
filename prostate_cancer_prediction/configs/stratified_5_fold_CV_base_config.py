from architecture.evaluation import collate_folds

from .base_config import train_samples, val_samples

# Shared overrides that turn a model config into a stratified 5-fold
# cross-validation run. This is layered (as a second spread) on top of a model
# parent -- base_config for the sklearn baselines, or
# pnet_single_split_elmarakeby_config for the P-NET variants -- so the
# model-specific keys (run_id, model, pipeline_class, results_processors,
# model_params, ...) are set by the individual configs.
stratified_5_fold_CV_base_config = {
    "val_metric":            {},
    "grid_search":           [],
    "grid_search_collators": [],
    "inner_kfolds":          5,
    "fold_collators":        [collate_folds],
    "stratified":            True,
    "tt_split_seed":         123,
    "tv_split_seed":         123,
    "run_method":            "run_crossvalidation",
    "train_samples":         train_samples + val_samples,
    "val_samples":           [],
}
