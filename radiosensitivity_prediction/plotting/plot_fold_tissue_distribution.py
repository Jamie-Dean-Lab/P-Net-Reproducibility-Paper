"""
Tissue composition of each outer crossvalidation test fold.

Fold membership comes from architecture.data_loading.outer_fold_ids, i.e. the same
dataset construction and splitter run_crossvalidation uses, so this figure cannot
drift from the folds the experiments actually run on.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from architecture.data_loading import outer_fold_ids
from radiosensitivity_prediction.configs.base_config import base_config, data_dir, figures_dir


def plot_fold_tissue_distribution():
    folds = outer_fold_ids(base_config)
    n_folds = len(folds)

    model_list = pd.read_csv(f"{data_dir}/model_list_20250630.csv",
                             usecols=["model_id", "tissue"], index_col="model_id")

    fold_tissues = []
    for i, (_, test_ids) in enumerate(folds):
        tissues = model_list.reindex(test_ids)["tissue"].fillna("Unknown")
        fold_tissues.append(tissues.value_counts().rename(str(i + 1)))

    counts = pd.concat(fold_tissues, axis=1).fillna(0).astype(int)
    counts = counts.loc[counts.sum(axis=1).sort_values(ascending=False).index]

    cmap = plt.colormaps.get_cmap("tab20").resampled(len(counts))
    colours = {tissue: cmap(i) for i, tissue in enumerate(counts.index)}

    fig, ax = plt.subplots(figsize=(10, 6))
    bottoms = np.zeros(n_folds)
    for tissue in counts.index:
        vals = counts.loc[tissue].values
        ax.bar(counts.columns, vals, bottom=bottoms, color=colours[tissue], label=tissue)
        bottoms += vals

    ax.set_xlabel("Outer fold (test set)", fontsize=14)
    ax.set_ylabel("Number of cell lines", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [mpatches.Patch(color=colours[t], label=t) for t in counts.index]
    ax.legend(handles=handles, bbox_to_anchor=(1.01, 1), loc="upper left",
              fontsize=10, frameon=False)

    fig.tight_layout()
    os.makedirs(figures_dir, exist_ok=True)
    fig.savefig(f"{figures_dir}/fold_tissue_distribution.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {figures_dir}/fold_tissue_distribution.pdf")
    print(counts.to_string())


if __name__ == "__main__":
    plot_fold_tissue_distribution()
