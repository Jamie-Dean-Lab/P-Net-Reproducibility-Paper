"""
P-NET hyperparameter sensitivity sweep for the tissue type classification task
(tissue_type_classification/runs/pnet_hyperparameter_sensitivity).

This task is multiclass and the pipeline scores it with task="group", so the
summary CSVs store each metric under its bare name. Sensitivity is characterised
with the validation weighted F1 — the model-selection metric for this task.

The sweep itself and the plotting are shared with the other tasks — see
architecture/plotting/hyperparameter_sensitivity.py.
"""

import os

from architecture.plotting import hyperparameter_sensitivity as hps

RUN_DIR = os.path.join("tissue_type_classification", "runs", "pnet_hyperparameter_sensitivity")
FIGURES_DIR = os.path.join("tissue_type_classification", "figures", "pnet_hyperparameter_sensitivity")

LABEL_PREFIX = ""
DEFAULT_METRIC = "f1"
METRIC_LABEL = {"f1": "Weighted F1", "auc": "AUROC", "r2": "R2"}


def analyse(run_dir=RUN_DIR, figures_dir=FIGURES_DIR, metric=DEFAULT_METRIC):
    metric_label = METRIC_LABEL.get(metric, metric.replace("_", " ").capitalize())
    return hps.analyse(
        run_dir, figures_dir, metric, metric_label,
        label_prefix=LABEL_PREFIX,
        filename_token=metric_label.replace(" ", "_"),
        bbox_inches=None,
    )


if __name__ == "__main__":
    analyse()
