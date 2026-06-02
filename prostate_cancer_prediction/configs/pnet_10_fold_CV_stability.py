import copy

from .base_config import auc_selection
from .pnet_single_split_elmarakeby import pnet_single_split_elmarakeby_config

pnet_10_fold_CV_stability_config = {
    **copy.deepcopy(pnet_single_split_elmarakeby_config),
    "run_id":                 "pnet_10_fold_CV_stability",
    "run_method":             "run_crossvalidation",
    "stratified":             True,
    "outer_kfolds":           10,
    "inner_kfolds":           1,
    "validation_prop":        0.2,
    "val_metric":             {"auc": auc_selection},
    "tt_split_seed":          123,
    "tv_split_seed":          123,
}
del pnet_10_fold_CV_stability_config["train_samples"]
del pnet_10_fold_CV_stability_config["val_samples"]
del pnet_10_fold_CV_stability_config["test_samples"]
