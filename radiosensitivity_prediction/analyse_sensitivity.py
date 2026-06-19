"""
Analyse the P-NET hyperparameter sensitivity sweep
(radiosensitivity_prediction/runs/pnet_sensitivity).

The sweep is a one-at-a-time (OAT) analysis run on fold 0 of the main
cross-validation (see extract_sensitivity_split.py / configs/pnet_sensitivity.py):
each hyperparameter is varied over a grid while the others are held at their
baseline values, and every configuration is scored with an inner 5-fold CV over
the *training* portion of the split.

LEAKAGE NOTE
------------
Each cv_*/fold_summaries.csv contains `train`, `val` and `test` rows. The `test`
rows are the held-out fold-0 test set, recomputed for every configuration purely
as a by-product of the pipeline threading test_df through every inner fold
(architecture/pipeline.py). Using those test numbers to characterise sensitivity
- or to pick a hyperparameter value - leaks the held-out set and biases the final
test estimate.

This script therefore builds every sensitivity curve from the inner-CV
*validation* folds only (split == "val"); the held-out test set is never read.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUN_DIR = os.path.join("radiosensitivity_prediction", "runs", "pnet_sensitivity")
FIGURES_DIR = os.path.join("radiosensitivity_prediction", "figures", "pnet_sensitivity")

# Response label prefix used in the summary CSVs (label is AUC_log1p).
LABEL_PREFIX = "AUC_log1p"
DEFAULT_METRIC = "r2"

# Baseline (default) value of each swept hyperparameter, mirroring
# configs/pnet_sensitivity.py. At its baseline value every sweep collapses to the
# same all-defaults configuration, so these points should agree across sweeps.
BASELINE = {
    "lr": 1e-3,
    "epochs": 300,
    "epochs_drop": 25,
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


def _metric_col(metric):
    return f"{LABEL_PREFIX}_{metric}"


def load_val_results(run_dir=RUN_DIR, metric=DEFAULT_METRIC):
    """
    Build a tidy per-fold table of validation scores for the OAT sweep.

    Returns a DataFrame with columns [param, value, fold, score], where `score`
    is the inner-CV validation metric for that hyperparameter value and fold.
    """
    with open(os.path.join(run_dir, "sweep_labels.json")) as f:
        sweep_labels = json.load(f)

    col = _metric_col(metric)
    rows = []
    for cv_name, label in sweep_labels.items():
        # Labels are "<param>_<value>"; the param name may itself contain
        # underscores (e.g. "epochs_drop_25"), so split on the last underscore.
        param, raw_value = label.rsplit("_", 1)
        value = float(raw_value)

        fold_summaries = os.path.join(run_dir, "test_0", cv_name, "fold_summaries.csv")
        if not os.path.exists(fold_summaries):
            # Sweep still running / partially complete - skip configs without results.
            print(f"Skipping {cv_name} ({label}): no fold_summaries.csv yet")
            continue
        df = pd.read_csv(fold_summaries, index_col=0)
        val = df.loc[df["split"] == "val", [col, "fold"]]
        for _, r in val.iterrows():
            rows.append({"param": param, "value": value, "fold": int(r["fold"]), "score": r[col]})

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


def plot_sensitivity(agg, figures_dir=FIGURES_DIR, metric=DEFAULT_METRIC):
    """One OAT curve per hyperparameter: mean validation score +/- sem vs value."""
    os.makedirs(figures_dir, exist_ok=True)
    metric_label = metric.upper() if metric == "r2" else metric.replace("_", " ").capitalize()

    for param, sub in agg.groupby("param"):
        sub = sub.sort_values("value")
        fig, ax = plt.subplots(figsize=(7, 5))

        ax.errorbar(sub["value"], sub["mean"], yerr=sub["sem"], marker="o", capsize=3, color="tab:blue")

        # Highlight the baseline value so the OAT reference point is obvious.
        base = sub[sub["is_baseline"]]
        if not base.empty:
            ax.scatter(base["value"], base["mean"], s=120, facecolors="none",
                       edgecolors="tab:red", zorder=5, label="Baseline")
            ax.legend(frameon=False)

        if param in LOG_X:
            ax.set_xscale("log")

        ax.set_xlabel(DISPLAY.get(param, param))
        ax.set_ylabel(f"Validation {metric_label} (mean +/- sem, 5 inner folds)")
        ax.set_title(f"P-NET sensitivity: {DISPLAY.get(param, param)}", loc="left",
                     fontdict={"fontsize": 12, "fontweight": "bold"})
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()
        fig.savefig(os.path.join(figures_dir, f"sensitivity_{param}_{metric}.png"), dpi=300)
        plt.close(fig)


def analyse(run_dir=RUN_DIR, figures_dir=FIGURES_DIR, metric=DEFAULT_METRIC):
    val_df = load_val_results(run_dir, metric)
    agg = aggregate(val_df)

    summary_path = os.path.join(run_dir, f"sensitivity_val_summary_{metric}.csv")
    agg.to_csv(summary_path, index=False)
    plot_sensitivity(agg, figures_dir, metric)

    print(f"Validation sensitivity summary ({metric}) written to {summary_path}")
    print(f"Figures written to {figures_dir}")
    return agg


if __name__ == "__main__":
    analyse()
