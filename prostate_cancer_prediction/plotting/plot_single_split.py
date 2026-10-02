import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)

from prostate_cancer_prediction.plotting.pnet_auprc import PlotAUPRC
from prostate_cancer_prediction.plotting.pnet_roc import PlotROC

# All single-split models plotted on one figure per block. The "elmarakeby" block
# uses the original paper's hyperparameters with the test + validation splits
# combined (concat_val=True); the default block uses our hyperparameters with the
# splits kept separate.
ELMARAKEBY_MODELS = [
    "pnet_single_split_elmarakeby",
    "pnetfc_single_split_elmarakeby",
    "dense_single_layer_single_split_elmarakeby",
    "decision_tree_single_split_elmarakeby",
    "adaboost_single_split_elmarakeby",
    "random_forest_single_split_elmarakeby",
    "linear_svm_single_split_elmarakeby",
    "rbf_svm_single_split_elmarakeby",
    "sgd_logistic_regression_single_split_elmarakeby",
]

SINGLE_SPLIT_MODELS = [
    "pnet_single_split",
    "pnet_GO_single_split",
    "pnetfc_single_split",
    "dense_single_layer_single_split",
    "decision_tree_single_split",
    "adaboost_single_split",
    "random_forest_single_split",
    "linear_svm_single_split",
    "rbf_svm_single_split",
    "sgd_logistic_regression_single_split"
]

_DEFAULT_MODELS = {"elmarakeby": ELMARAKEBY_MODELS, "": SINGLE_SPLIT_MODELS}

# Standardised legend labels, keyed by the model id with the
# "_single_split"/"_single_split_elmarakeby" suffix stripped. Kept in sync with
# plot_nested_cv.plot_nested_CV's models_display so a model reads identically
# across the single-split and nested-CV figures. Ids without an entry fall back
# to the stripped id.
_DISPLAY_NAMES = {
    "pnet":                    "P-NET",
    "pnet_GO":                 "P-NET-GO",
    "pnetfc":                  "P-NET-FC",
    "dense_single_layer":      "Dense Single Layer",
    "decision_tree":           "Decision Tree",
    "adaboost":                "Ada. Boosting",
    "linear_svm":              "Linear SVM",
    "random_forest":           "Random Forest",
    "rbf_svm":                 "RBF SVM",
    "sgd_logistic_regression": "Logistic Regression",
}


def _display_name(model):
    short = model.replace("_single_split_elmarakeby", "").replace("_single_split", "")
    return _DISPLAY_NAMES.get(short, short)


def _resolve_results_dir(run_dir, model, selection_metric):
    """Return the directory holding a model's result CSVs.

    When hyperparameters are selected via grid search the pipeline writes
    results to a ``best_{selection_metric}`` subdirectory (e.g. ``best_auc``);
    without grid search they sit directly in the model directory. Prefer the
    selection subdirectory and fall back to the model directory.
    """
    base = f"{run_dir}/{model}"
    best_dir = f"{base}/best_{selection_metric}"
    return best_dir if os.path.isdir(best_dir) else base


def _auprc_source_data(results):
    """Return the plotted precision/recall curve points as one long dataframe.

    Mirrors PlotAUPRC.plot so the CSV is the source data for the figure: one row
    per curve point, plus the average precision quoted in the legend.
    """
    frames = []
    for label, df in results.items():
        y_true = np.array(df["response"])
        pred_scores = np.array(df["response_pred"])
        precision, recall, thresholds = precision_recall_curve(y_true, pred_scores)
        # precision_recall_curve returns one fewer threshold than curve points:
        # the final (recall=0, precision=1) point has no corresponding threshold.
        frames.append(pd.DataFrame({
            "model": label,
            "auprc": average_precision_score(y_true, pred_scores),
            "threshold": np.append(thresholds, np.nan),
            "recall": recall,
            "precision": precision,
        }))
    return pd.concat(frames, ignore_index=True)


def _auroc_source_data(results):
    """Return the plotted FPR/TPR curve points as one long dataframe.

    Mirrors PlotROC.plot; the AUC column is the value quoted in the legend.
    """
    frames = []
    for label, df in results.items():
        y_true = np.array(df["response"])
        pred_scores = np.array(df["response_pred"])
        fpr, tpr, thresholds = roc_curve(y_true, pred_scores)
        frames.append(pd.DataFrame({
            "model": label,
            "auroc": roc_auc_score(y_true, pred_scores),
            "threshold": thresholds,
            "fpr": fpr,
            "tpr": tpr,
        }))
    return pd.concat(frames, ignore_index=True)


def plot_single_split_curves(run_dir, figures_dir, models=None, tag="", concat_val=False,
                             selection_metric="auc"):
    if models is None:
        models = _DEFAULT_MODELS[tag]
    results = {}
    tabular = []
    for model in models:
        base = _resolve_results_dir(run_dir, model, selection_metric)
        label = _display_name(model)
        test_df = pd.read_csv(f"{base}/test_results.csv", index_col=0)
        if concat_val:
            val_df = pd.read_csv(f"{base}/val_results.csv", index_col=0)
            results[label] = pd.concat([test_df, val_df])
        else:
            results[label] = test_df
        summary = pd.read_csv(f"{base}/summary_results.csv")
        summary["model"] = model
        summary.columns = ["split"] + summary.columns[1:].to_list()
        tabular.append(summary)

    suffix = f"_{tag}" if tag else ""
    for plotter, source_data, stem in [
            (PlotAUPRC, _auprc_source_data, f"single_split_auprc{suffix}"),
            (PlotROC, _auroc_source_data, f"single_split_auroc{suffix}")]:
        fig, ax = plt.subplots()
        plotter(results).plot(ax, "")
        fig.savefig(os.path.join(figures_dir, f"{stem}.pdf"))
        plt.close(fig)
        source_data(results).to_csv(
            os.path.join(figures_dir, f"{stem}_source_data.csv"), index=False)
