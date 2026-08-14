"""
One-at-a-time (OAT) hyperparameter sensitivity curves.

Each task runs the same sweep: a single fixed train/test split (fold 0 of the main
crossvalidation, see architecture/sensitivity_split.py), every hyperparameter varied
over a grid in turn while the rest stay at their baseline, and each setting scored by
an inner 5-fold crossvalidation over the *training* portion. Scores plotted here are
therefore validation scores — the held-out test fold plays no part in them.

Only the run directory, the metric and its label differ between tasks.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Baseline (default) value of each swept hyperparameter, mirroring each task's
# configs/pnet_hyperparameter_sensitivity.py. At its baseline value every sweep
# collapses to the same all-defaults configuration, so these points should agree
# across sweeps.
BASELINE = {
    "lr": 1e-3,
    "epochs": 300,
    "epochs_drop": 50,
    "h_dropout_first": 0.0,
    "h_dropout_rest": 0.0,
    "batch": 50,
}

DISPLAY = {
    "lr": "Learning rate",
    "epochs": "Epochs",
    "epochs_drop": "Epochs per LR drop",
    "h_dropout_first": "First-layer dropout",
    "h_dropout_rest": "Pathway-layer dropout",
    "batch": "Batch size",
}

# Hyperparameters whose grid spans orders of magnitude read better on a log x-axis.
LOG_X = {"lr", "batch"}

FONTSIZE = 18
FONTPROPERTIES = {"family": "Arial", "weight": "normal", "size": 20}


def load_val_results(run_dir, metric, label_prefix=""):
    """
    Tidy per-fold table of validation scores for the OAT sweep.

    args:
        label_prefix (str) : prefix the summary CSVs give the metric column, e.g.
                             "response" for the binary tasks. Empty when the pipeline
                             scored with task="group", which stores the bare name.

    returns:
        DataFrame with columns [param, value, fold, score]
    """
    with open(os.path.join(run_dir, "sweep_labels.json")) as f:
        sweep_labels = json.load(f)

    col = f"{label_prefix}_{metric}" if label_prefix else metric
    rows = []
    for cv_name, label in sweep_labels.items():
        # Labels are "<param>_<value>"; the param name may itself contain
        # underscores (e.g. "epochs_drop_25"), so split on the last underscore.
        param, raw_value = label.rsplit("_", 1)
        df = pd.read_csv(os.path.join(run_dir, "test_0", cv_name, "fold_summaries.csv"), index_col=0)
        val = df.loc[df["split"] == "val", [col, "fold"]]
        for _, r in val.iterrows():
            rows.append({"param": param, "value": float(raw_value),
                         "fold": int(r["fold"]), "score": r[col]})

    return pd.DataFrame(rows)


def aggregate(val_df):
    """Mean / std / sem of the validation score per (param, value), across folds."""
    agg = (
        val_df.groupby(["param", "value"])["score"]
        .agg(["mean", "std", "sem", "count"])
        .reset_index()
        .sort_values(["param", "value"])
        .reset_index(drop=True)
    )
    agg["is_baseline"] = agg.apply(
        lambda r: np.isclose(r["value"], BASELINE.get(r["param"], np.nan)), axis=1
    )
    return agg


def plot_sensitivity(agg, figures_dir, metric_label, filename_token, bbox_inches=None):
    """One OAT curve per hyperparameter: mean validation score +/- sd vs value."""
    os.makedirs(figures_dir, exist_ok=True)

    for param, sub in agg.groupby("param"):
        sub = sub.sort_values("value")
        fig, ax = plt.subplots(figsize=(7, 5))

        ax.errorbar(sub["value"], sub["mean"], yerr=sub["std"], marker="o",
                    capsize=3, color="tab:blue")

        # Highlight the baseline value so the OAT reference point is obvious.
        base = sub[sub["is_baseline"]]
        if not base.empty:
            ax.scatter(base["value"], base["mean"], s=120, facecolors="none",
                       edgecolors="tab:red", zorder=5)

        if param in LOG_X:
            ax.set_xscale("log")

        ax.set_xlabel(DISPLAY.get(param, param), FONTPROPERTIES)
        ax.set_ylabel(metric_label, FONTPROPERTIES)
        ax.tick_params(axis="both", which="major", labelsize=FONTSIZE)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()
        path = os.path.join(figures_dir, f"sensitivity_{param}_{filename_token}.pdf")
        if bbox_inches:
            fig.savefig(path, bbox_inches=bbox_inches)
        else:
            fig.savefig(path)
        plt.close(fig)


def analyse(run_dir, figures_dir, metric, metric_label, label_prefix="",
            filename_token=None, bbox_inches=None):
    """
    Runs the whole sensitivity analysis: load, aggregate, write the summary CSV and
    the per-hyperparameter figures.

    args:
        filename_token (str) : token used in the output filenames; defaults to the
                               metric label, which is what the binary tasks have
                               always used
    """
    if filename_token is None:
        filename_token = metric_label

    agg = aggregate(load_val_results(run_dir, metric, label_prefix))

    summary_path = os.path.join(run_dir, f"sensitivity_val_summary_{metric}.csv")
    agg.to_csv(summary_path, index=False)
    plot_sensitivity(agg, figures_dir, metric_label, filename_token, bbox_inches)

    print(f"Validation sensitivity summary ({metric}) written to {summary_path}")
    print(f"Figures written to {figures_dir}")
    return agg
