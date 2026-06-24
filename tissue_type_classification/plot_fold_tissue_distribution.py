import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATA_DIR = "tissue_type_classification/data"
FIGURES_DIR = "tissue_type_classification/figures"
TT_SPLIT_SEED = 42
OUTER_KFOLDS = 5
DPI = 300


def _get_dataset_ids_and_tissues():
    """Return sorted sample IDs used by the pipeline and their tissue type.

    Intersection of expression and label samples, with samples missing labels dropped.
    Tissue type is decoded from the one-hot encoded label file.
    """
    expr_ids = set(pd.read_csv(f"{DATA_DIR}/GTEx_gene_expression_preprocessed.csv",
                               index_col=0, usecols=[0]).index)
    labels = pd.read_csv(f"{DATA_DIR}/GTEx_tissue_classes_encoded.csv", index_col=0)
    labels = labels.dropna()
    label_ids = set(labels.index)

    common_ids = sorted(expr_ids & label_ids)
    labels = labels.loc[common_ids]
    tissue_map = labels.idxmax(axis=1)
    return common_ids, tissue_map


def _get_outer_test_folds(ids, n_splits=5, seed=42):
    """Replicate pipeline's _get_k_splits (non-stratified). Returns list of test-fold ID lists."""
    idxs = np.arange(len(ids))
    step = len(ids) // n_splits
    splits = range(0, len(ids), step)
    np.random.default_rng(seed).shuffle(idxs)
    folds = []
    for i in range(n_splits):
        test_idxs = idxs[splits[i]:splits[i + 1]] if i < n_splits - 1 else idxs[splits[i]:]
        folds.append([ids[j] for j in test_idxs])
    return folds


def plot_fold_tissue_distribution():
    ids, tissue_map = _get_dataset_ids_and_tissues()
    folds = _get_outer_test_folds(ids, n_splits=OUTER_KFOLDS, seed=TT_SPLIT_SEED)

    fold_tissues = []
    for i, fold_ids in enumerate(folds):
        tissues = tissue_map.reindex(fold_ids).fillna("Unknown")
        fold_tissues.append(tissues.value_counts().rename(str(i + 1)))

    counts = pd.concat(fold_tissues, axis=1).fillna(0).astype(int)
    counts = counts.loc[counts.sum(axis=1).sort_values(ascending=False).index]

    row_height = 2.0
    fig, ax = plt.subplots(figsize=(26, row_height * len(counts)))
    im = ax.imshow(counts.values, aspect="auto", cmap="YlOrRd")

    ax.set_xticks(range(OUTER_KFOLDS))
    ax.set_xticklabels(counts.columns, fontsize=52)
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels(counts.index, fontsize=48)
    ax.set_xlabel("Outer fold (test set)", fontsize=52)
    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)

    for i in range(len(counts)):
        for j in range(OUTER_KFOLDS):
            val = counts.iloc[i, j]
            ax.text(j, i, str(val), ha="center", va="center", fontsize=48,
                    color="white" if val > counts.values.max() * 0.6 else "black")

    cbar = fig.colorbar(im, ax=ax, shrink=0.4)
    cbar.set_label("Number of samples", fontsize=48)
    cbar.ax.tick_params(labelsize=44)

    fig.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(f"{FIGURES_DIR}/fold_tissue_distribution.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {FIGURES_DIR}/fold_tissue_distribution.png")
    print(counts.to_string())


if __name__ == "__main__":
    plot_fold_tissue_distribution()
