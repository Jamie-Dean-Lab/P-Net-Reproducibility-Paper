import copy

from architecture.evaluation import plot_history
from .base_config import save_processor
from .pnet_single_split_elmarakeby import pnet_single_split_elmarakeby_config
from .stratified_5_fold_CV_base_config import stratified_5_fold_CV_base_config

# Sparse P-NET with the default elmarakeby hyperparameters (inherited via
# model_params); the stratified base clears grid_search and switches to 5-fold CV.
pnet_stratified_5_fold_CV_config = {
    **copy.deepcopy(pnet_single_split_elmarakeby_config),
    **copy.deepcopy(stratified_5_fold_CV_base_config),
    "run_id":                    "pnet_stratified_5_fold_CV",
    "results_processors":        [save_processor, plot_history],
}
