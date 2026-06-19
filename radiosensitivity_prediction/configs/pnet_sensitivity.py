import copy
import json
import os
from functools import partial

from keras.regularizers import L2
from keras.callbacks import LearningRateScheduler

from architecture.evaluation import plot_history, collate_folds
from architecture.pipeline import TFPipeline
from architecture.pnet_model import compile_pnet
from architecture.callbacks_custom import step_decay
from .base_config import base_config, save_processor, data_dir
from radiosensitivity_prediction.extract_sensitivity_split import extract_sensitivity_split

n_hidden_layers = 5

# Fixed at values selected by the main h_reg/o_reg grid search (fold 0: h_reg_1e-06_o_reg_1e-06).
SELECTED_H_REG = 1e-6
SELECTED_O_REG = 1e-6

# Baseline hyperparameter values (matching pnet.py / Elmarakeby et al.).
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_DROP = 0.5
DEFAULT_EPOCHS_DROP = 25
DEFAULT_H_DROPOUT_FIRST = 0.5
DEFAULT_H_DROPOUT_REST = 0.1
DEFAULT_BATCH = 50

_model_params_fixed = {
    "pathway_dataset": "reactome",
    "pp_relations": "architecture/Reactome/ReactomePathwaysRelation.txt",
    "gp_relations": "architecture/Reactome/ReactomePathways.gmt",
    "n_hidden_layers": n_hidden_layers,
    "h_activation": ["tanh"] * (n_hidden_layers + 1),
    "o_activation": ["linear"] * (n_hidden_layers + 1),
    "h_kernel_initializer": ["lecun_uniform"] * (n_hidden_layers + 1),
    "h_kernel_constraints": [None] * (n_hidden_layers + 1),
    "h_bias_initializer": ["lecun_uniform"] * (n_hidden_layers + 1),
    "h_bias_constraints": [None] * (n_hidden_layers + 1),
    "batch_normal": False,
    "sparse": True,
    "dropout_testing": False,
    "loss": ["MeanSquaredError"] * (n_hidden_layers + 1),
    "loss_weights": [2, 7, 20, 54, 148, 400],
    "map_seed": 42,
    "h_reg": [(L2, {"l2": SELECTED_H_REG})] * (n_hidden_layers + 1),
    "o_reg": [(L2, {"l2": SELECTED_O_REG})] * (n_hidden_layers + 1),
}

_results_processors = [
    save_processor,
    plot_history
]


def _make_model_params(learning_rate=DEFAULT_LEARNING_RATE,
                       h_dropout_first=DEFAULT_H_DROPOUT_FIRST,
                       h_dropout_rest=DEFAULT_H_DROPOUT_REST):
    return {
        **copy.deepcopy(_model_params_fixed),
        "h_dropout": [h_dropout_first] + [h_dropout_rest] * n_hidden_layers,
        "optimizer": {"class_name": "Adam", "config": {"learning_rate": learning_rate}},
    }


DEFAULT_EPOCHS = 300

def _make_fitting_params(learning_rate=DEFAULT_LEARNING_RATE,
                         drop=DEFAULT_DROP,
                         epochs_drop=DEFAULT_EPOCHS_DROP,
                         batch=DEFAULT_BATCH,
                         epochs=DEFAULT_EPOCHS):
    return {
        "epochs": epochs,
        "batch": batch,
        "LRScheduler": LearningRateScheduler(
            partial(step_decay, init_lr=learning_rate, drop=drop, epochs_drop=epochs_drop),
            verbose=0,
        ),
        "early_stopping": None,
        "prediction_output": "average",
        "shuffle_samples": True,
        "class_weight": None,
    }


# One-at-a-time (OAT) sweep: each hyperparameter varied independently, others at default.
# The default value is included in each sweep to confirm baseline behaviour on this split.
LEARNING_RATE_VALUES = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
EPOCHS_VALUES = [100, 200, 300, 400, 500]
EPOCHS_DROP_VALUES = [5, 10, 25, 50, 100]
H_DROPOUT_FIRST_VALUES = [0.3, 0.4, 0.5, 0.6, 0.7]
BATCH_VALUES = [10, 25, 50, 100, 200]

_grid_search_params = (
    [
        {
            "model_params": _make_model_params(learning_rate=lr),
            "fitting_params": _make_fitting_params(learning_rate=lr),
            "model_params_choice": f"lr_{lr}",
        }
        for lr in LEARNING_RATE_VALUES
    ] + [
        {
            "model_params": _make_model_params(),
            "fitting_params": _make_fitting_params(epochs=e),
            "model_params_choice": f"epochs_{e}",
        }
        for e in EPOCHS_VALUES
    ] + [
        {
            "model_params": _make_model_params(),
            "fitting_params": _make_fitting_params(epochs_drop=ed),
            "model_params_choice": f"epochs_drop_{ed}",
        }
        for ed in EPOCHS_DROP_VALUES
    ] + [
        {
            "model_params": _make_model_params(h_dropout_first=do),
            "fitting_params": _make_fitting_params(),
            "model_params_choice": f"h_dropout_first_{do}",
        }
        for do in H_DROPOUT_FIRST_VALUES
    ] + [
        {
            "model_params": _make_model_params(),
            "fitting_params": _make_fitting_params(batch=b),
            "model_params_choice": f"batch_{b}",
        }
        for b in BATCH_VALUES
    ]
)

_split_path = os.path.join(data_dir, "sensitivity_split_fold0.json")
if not os.path.exists(_split_path):
    extract_sensitivity_split()
with open(_split_path) as f:
    _split = json.load(f)

_run_dir = os.path.join("radiosensitivity_prediction", "runs", "pnet_sensitivity")
os.makedirs(_run_dir, exist_ok=True)
_sweep_labels = {f"cv_{i}": p["model_params_choice"] for i, p in enumerate(_grid_search_params)}
with open(os.path.join(_run_dir, "sweep_labels.json"), "w") as f:
    json.dump(_sweep_labels, f, indent=2)

pnet_sensitivity_config = {
    **copy.deepcopy(base_config),
    "run_id": "pnet_sensitivity",
    "model": compile_pnet,
    "fitting_params": _make_fitting_params(),
    "pipeline_class": TFPipeline,
    "results_processors": _results_processors,
    "run_method": "run_crossvalidation",
    "fold_collators": [collate_folds],
    "train_samples": _split["train"],
    "val_samples": [],
    "test_samples": _split["test"],
    "grid_search": _grid_search_params,
}
