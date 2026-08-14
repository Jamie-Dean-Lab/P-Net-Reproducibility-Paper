"""
P-NET hyperparameter sensitivity sweep for the radiosensitivity prediction task
(radiosensitivity_prediction/runs/pnet_hyperparameter_sensitivity).

The regression label is AUC_log1p, so metrics are stored as AUC_log1p_<metric>.

The sweep itself and the plotting are shared with the other tasks — see
architecture/plotting/hyperparameter_sensitivity.py.
"""

import os

from architecture.plotting import hyperparameter_sensitivity as hps

RUN_DIR = os.path.join("radiosensitivity_prediction", "runs", "pnet_hyperparameter_sensitivity")
FIGURES_DIR = os.path.join("radiosensitivity_prediction", "figures", "pnet_hyperparameter_sensitivity")

LABEL_PREFIX = "AUC_log1p"
DEFAULT_METRIC = "r2"
METRIC_LABEL = {"auc": "AUROC", "r2": "$R^2$"}


def analyse(run_dir=RUN_DIR, figures_dir=FIGURES_DIR, metric=DEFAULT_METRIC):
    metric_label = METRIC_LABEL.get(metric, metric.replace("_", " ").capitalize())
    return hps.analyse(
        run_dir, figures_dir, metric, metric_label,
        label_prefix=LABEL_PREFIX,
        filename_token=metric,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    analyse()
