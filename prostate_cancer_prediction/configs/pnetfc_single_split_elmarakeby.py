import copy

from architecture.evaluation import plot_history
from .base_config import save_processor
from .pnet_single_split_elmarakeby import pnet_single_split_elmarakeby_config, n_hidden_layers


pnetfc_single_split_elmarakeby_config = {
    **copy.deepcopy(pnet_single_split_elmarakeby_config),
    "run_id":             "pnetfc_single_split_elmarakeby",
    "results_processors": [save_processor, plot_history],
    "model_params": {
        **copy.deepcopy(pnet_single_split_elmarakeby_config["model_params"]),
        "sparse": False
    },
}
