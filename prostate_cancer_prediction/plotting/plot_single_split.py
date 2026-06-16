import os

import pandas as pd
from matplotlib import pyplot as plt

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

# Pretty legend labels for model ids that need it; others fall back to the id
# with the "_single_split"/"_single_split_elmarakeby" suffix stripped.
_DISPLAY_NAMES = {
    "pnetfc_single_split": "pnet_FC",
    "pnetfc_single_split_elmarakeby": "pnet_FC",
}


def _display_name(model):
    if model in _DISPLAY_NAMES:
        return _DISPLAY_NAMES[model]
    return model.replace("_single_split_elmarakeby", "").replace("_single_split", "")


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
    for plotter, fname in [(PlotAUPRC, f"single_split_auprc{suffix}.png"),
                           (PlotROC, f"single_split_auroc{suffix}.png")]:
        fig, ax = plt.subplots()
        plotter(results).plot(ax, "")
        fig.savefig(os.path.join(figures_dir, fname), dpi=300)
        plt.close(fig)
