import copy

from keras.regularizers import L2

from architecture.evaluation import plot_history
from .base_config import save_processor
from .pnet_single_split_elmarakeby import pnet_single_split_elmarakeby_config, n_hidden_layers
from .stratified_5_fold_CV_base_config import stratified_5_fold_CV_base_config

pnetfc_stratified_5_fold_CV_config = {
    **copy.deepcopy(pnet_single_split_elmarakeby_config),
    **copy.deepcopy(stratified_5_fold_CV_base_config),
    "run_id":                    "pnetfc_stratified_5_fold_CV",
    "results_processors":        [save_processor, plot_history],
    "model_params": {
        **copy.deepcopy(pnet_single_split_elmarakeby_config["model_params"]),
        "sparse": False,
    },
}
