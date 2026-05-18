import copy

from architecture.evaluation import plot_history, collate_folds
from base_config import save_processor, train_samples, val_samples
from pnet_single_split_elmarakeby import pnet_single_split_elmarakeby_config

n_hidden_layers = 5

pnet_stratified_5_fold_CV_config = {
    **copy.deepcopy(pnet_single_split_elmarakeby_config),
    "run_id":                    "pnet_stratified_5_fold_CV",
    "results_processors":        [save_processor, plot_history],
    "val_metric":                {},
    "grid_search":               [],
    "grid_search_collators":     [],
    "inner_kfolds":              5,
    "fold_collators":            [collate_folds],
    "stratified":                True,
    "tt_split_seed":             123,
    "tv_split_seed":             123,
    "run_method":                "run_crossvalidation",
    "train_samples":             train_samples + val_samples,
    "val_samples":               []
}
