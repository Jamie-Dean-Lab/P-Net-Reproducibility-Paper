import copy
from keras.regularizers import L2

from architecture.pipeline import TFPipeline
from architecture.pnet_model import compile_pnet
from .base_config import base_config
from .pnet import _model_params_base, _fitting_params, n_hidden_layers

dense_config = {
    **copy.deepcopy(base_config),
    "run_id":         "dense",
    "model":          compile_pnet,
    "fitting_params": _fitting_params,
    "pipeline_class": TFPipeline,
    "run_method":     "run_crossvalidation",
    "grid_search": {
        "model_params": {
            f"h_reg_{h}_o_reg_{o}": {**_model_params_base, "sparse": False,
                                     "h_reg": [(L2, {"l2": h})] * (n_hidden_layers + 1),
                                     "o_reg": [(L2, {"l2": o})] * (n_hidden_layers + 1)}
            for h in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
            for o in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
        },
    },
}