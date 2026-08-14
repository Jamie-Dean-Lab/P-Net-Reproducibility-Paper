"""
Tissue composition of each outer crossvalidation test fold.

Fold membership comes from architecture.data_loading.outer_fold_ids, i.e. the same
dataset construction and splitter run_crossvalidation uses, so this figure cannot
drift from the folds the experiments actually run on. This matters here because the
task splits are unstratified, and the point of the figure is to show how unevenly
tissues land across folds.
"""

import os

import pandas as pd
import matplotlib.pyplot as plt

from architecture.data_loading import outer_fold_ids
from tissue_type_classification.configs.base_config import base_config, data_dir, figures_dir


def _tissue_map():
    """Tissue type per sample, decoded from the one-hot encoded label file."""
    labels = pd.read_csv(f"{data_dir}/GTEx_tissue_classes_encoded.csv", index_col=0).dropna()
    return labels.idxmax(axis=1)


def plot_fold_tissue_distribution():
    folds = outer_fold_ids(base_config)
    n_folds = len(folds)
    tissue_map = _tissue_map()

    fold_tissues = []
    for i, (_, test_ids) in enumerate(folds):
        tissues = tissue_map.reindex(test_ids).fillna("Unknown")
        fold_tissues.append(tissues.value_counts().rename(str(i + 1)))

    counts = pd.concat(fold_tissues, axis=1).fillna(0).astype(int)
    counts = counts.loc[counts.sum(axis=1).sort_values(ascending=False).index]

    row_height = 2.0
    fig, ax = plt.subplots(figsize=(26, row_height * len(counts)))
    im = ax.imshow(counts.values, aspect="auto", cmap="YlOrRd")

    ax.set_xticks(range(n_folds))
    ax.set_xticklabels(counts.columns, fontsize=52)
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels(counts.index, fontsize=48)
    ax.set_xlabel("Outer fold (test set)", fontsize=52)
    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)

    for i in range(len(counts)):
        for j in range(n_folds):
            val = counts.iloc[i, j]
            ax.text(j, i, str(val), ha="center", va="center", fontsize=48,
                    color="white" if val > counts.values.max() * 0.6 else "black")

    cbar = fig.colorbar(im, ax=ax, shrink=0.4)
    cbar.set_label("Number of samples", fontsize=48)
    cbar.ax.tick_params(labelsize=44)

    fig.tight_layout()
    os.makedirs(figures_dir, exist_ok=True)
    fig.savefig(f"{figures_dir}/fold_tissue_distribution.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {figures_dir}/fold_tissue_distribution.pdf")
    print(counts.to_string())


if __name__ == "__main__":
    plot_fold_tissue_distribution()
