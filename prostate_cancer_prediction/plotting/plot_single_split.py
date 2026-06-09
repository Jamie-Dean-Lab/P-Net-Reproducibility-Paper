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
    "pnet_GO_single_split_config",
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


def plot_single_split_curves(run_dir, figures_dir, models=None, tag="", concat_val=False):
    if models is None:
        models = _DEFAULT_MODELS[tag]
    results = {}
    tabular = []
    for model in models:
        base = f"{run_dir}/{model}"
        test_df = pd.read_csv(f"{base}/test_results.csv", index_col=0)
        if concat_val:
            val_df = pd.read_csv(f"{base}/val_results.csv", index_col=0)
            results[model] = pd.concat([test_df, val_df])
        else:
            results[model] = test_df
        summary = pd.read_csv(f"{base}/summary_results.csv")
        summary["model"] = model
        summary.columns = ["split"] + summary.columns[1:].to_list()
        tabular.append(summary)

    suffix = f"_{tag}" if tag else ""
    for curve, plotter, fname in [("A", PlotAUPRC, f"single_split_auprc{suffix}.png"),
                                  ("A", PlotROC, f"single_split_roc{suffix}.png")]:
        fig, ax = plt.subplots()
        plotter(results).plot(ax, curve)
        fig.savefig(os.path.join(figures_dir, fname), dpi=300)
        plt.close(fig)
