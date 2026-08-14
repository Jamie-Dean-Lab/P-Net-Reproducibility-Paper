"""
P-NET hyperparameter sensitivity sweep for the glioma prediction task
(glioma_prediction/runs/pnet_hyperparameter_sensitivity).

Binary-classification metrics are stored as response_<metric> in the summary CSVs.

The sweep itself and the plotting are shared with the other tasks — see
architecture/plotting/hyperparameter_sensitivity.py.
"""

import os

from architecture.plotting import hyperparameter_sensitivity as hps

RUN_DIR = os.path.join("glioma_prediction", "runs", "pnet_hyperparameter_sensitivity")
FIGURES_DIR = os.path.join("glioma_prediction", "figures", "pnet_hyperparameter_sensitivity")

LABEL_PREFIX = "response"
DEFAULT_METRIC = "auc"
METRIC_LABEL = {"auc": "AUROC", "r2": "R2"}


def analyse(run_dir=RUN_DIR, figures_dir=FIGURES_DIR, metric=DEFAULT_METRIC):
    metric_label = METRIC_LABEL.get(metric, metric.replace("_", " ").capitalize())
    return hps.analyse(
        run_dir, figures_dir, metric, metric_label,
        label_prefix=LABEL_PREFIX,
        filename_token=metric_label,
        bbox_inches=None,
    )


if __name__ == "__main__":
    analyse()
