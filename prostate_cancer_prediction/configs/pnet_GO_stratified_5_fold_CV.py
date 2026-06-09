import copy

from keras.regularizers import L2

from architecture.evaluation import plot_history
from .base_config import save_processor
from .pnet_GO_single_split import pnet_GO_single_split_config, _model_params_base, n_hidden_layers
from .stratified_5_fold_CV_base_config import stratified_5_fold_CV_base_config

pnet_GO_stratified_5_fold_CV_config = {
    **copy.deepcopy(pnet_GO_single_split_config),
    **copy.deepcopy(stratified_5_fold_CV_base_config),
    "run_id":             "pnet_GO_stratified_5_fold_CV",
    "results_processors": [save_processor, plot_history],
    "model_params": {
        **copy.deepcopy(_model_params_base),
        "h_reg": [(L2, {"l2": 1e-3})] * (n_hidden_layers + 1),
        "o_reg": [(L2, {"l2": 1e-2})] * (n_hidden_layers + 1),
    },
}
