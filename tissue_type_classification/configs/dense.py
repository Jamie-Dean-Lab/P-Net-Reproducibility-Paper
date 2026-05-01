import copy
from keras.regularizers import L2

from architecture.pipeline import TFPipeline
from architecture.pnet_model import compile_pnet
from .base_config import base_config, save_processor
from .pnet import _model_params_base, _fitting_params, n_hidden_layers

dense_config = {
    **copy.deepcopy(base_config),
    "run_id":             "dense",
    "model":              compile_pnet,
    "fitting_params":     _fitting_params,
    "results_processors": [save_processor],
    "pipeline_class":     TFPipeline,
    "run_method":         "run_crossvalidation",
    "grid_search": {
        "model_params": {
            f"reg_{l}": {**_model_params_base, "sparse": False,
                         "h_reg": [(L2, {"l2": 10 ** l})] * (n_hidden_layers + 1),
                         "o_reg": [(L2, {"l2": 10 ** l})] * (n_hidden_layers + 1)}
            for l in [-3, -4, -5, -6]
        }
    },
}