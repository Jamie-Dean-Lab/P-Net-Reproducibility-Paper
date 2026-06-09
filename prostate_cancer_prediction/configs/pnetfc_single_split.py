import copy

from keras.regularizers import L2

from .pnet_single_split import pnet_single_split_config, _model_params_base, n_hidden_layers

pnetfc_single_split_config = {
    **copy.deepcopy(pnet_single_split_config),
    "run_id": "pnetfc_single_split",
    "grid_search": {
        "model_params": {
            f"h_reg_{h}_o_reg_{o}": {**copy.deepcopy(_model_params_base),
                                     "sparse": False,
                                     "h_reg": [(L2, {"l2": h})] * (n_hidden_layers + 1),
                                     "o_reg": [(L2, {"l2": o})] * (n_hidden_layers + 1)}
            for h in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
            for o in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
        },
    }
}
