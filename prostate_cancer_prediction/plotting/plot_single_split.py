import os

import pandas as pd
from matplotlib import pyplot as plt

from prostate_cancer_prediction.plotting.pnet_auprc import PlotAUPRC
from prostate_cancer_prediction.plotting.pnet_roc import PlotROC


def plot_single_split_curves(run_dir, wd, figures_dir, models, tag="", concat_val=False):
    results = {}
    tabular = []
    for model in models:
        base = f"{run_dir}/{model}/best_auc"
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
                                   ("A", PlotROC,   f"single_split_roc{suffix}.png")]:
        fig, ax = plt.subplots()
        plotter(results).plot(ax, curve)
        fig.savefig(os.path.join(figures_dir, fname), dpi=300)
        plt.close(fig)