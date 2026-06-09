import copy

from .base_config import save_processor
from .dense_single_layer_single_split_elmarakeby import dense_single_layer_single_split_elmarakeby_config
from .stratified_5_fold_CV_base_config import stratified_5_fold_CV_base_config

dense_single_layer_stratified_5_fold_CV_config = {
    **copy.deepcopy(dense_single_layer_single_split_elmarakeby_config),
    **copy.deepcopy(stratified_5_fold_CV_base_config),
    "run_id":             "dense_single_layer_stratified_5_fold_CV",
    "results_processors": [save_processor],
}
