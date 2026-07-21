import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


DATA_DIR = "radiosensitivity_prediction/data"
FIGURES_DIR = "radiosensitivity_prediction/figures"
TT_SPLIT_SEED = 42
OUTER_KFOLDS = 5


def _get_dataset_ids():
    """Return sorted sample IDs used by the pipeline: intersection of both views and labels."""
    expr_ids = set(pd.read_csv(f"{DATA_DIR}/ccle_gene_expression_preprocessed.csv",
                               index_col=0, usecols=[0]).index)
    meth_ids = set(pd.read_csv(f"{DATA_DIR}/methylation_preprocessed.csv",
                               index_col=0, usecols=[0]).index)
    label_ids = set(pd.read_csv(f"{DATA_DIR}/cleveland_auc_preprocessed.csv",
                                index_col=0).dropna().index)
    return sorted(expr_ids & meth_ids & label_ids)


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
    ids = _get_dataset_ids()
    folds = _get_outer_test_folds(ids, n_splits=OUTER_KFOLDS, seed=TT_SPLIT_SEED)

    model_list = pd.read_csv(f"{DATA_DIR}/model_list_20250630.csv",
                             usecols=["model_id", "tissue"], index_col="model_id")

    fold_tissues = []
    for i, fold_ids in enumerate(folds):
        tissues = model_list.reindex(fold_ids)["tissue"].fillna("Unknown")
        fold_tissues.append(tissues.value_counts().rename(str(i + 1)))

    counts = pd.concat(fold_tissues, axis=1).fillna(0).astype(int)
    counts = counts.loc[counts.sum(axis=1).sort_values(ascending=False).index]

    cmap = plt.colormaps.get_cmap("tab20").resampled(len(counts))
    colours = {tissue: cmap(i) for i, tissue in enumerate(counts.index)}

    fig, ax = plt.subplots(figsize=(10, 6))
    bottoms = np.zeros(OUTER_KFOLDS)
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
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(f"{FIGURES_DIR}/fold_tissue_distribution.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {FIGURES_DIR}/fold_tissue_distribution.pdf")
    print(counts.to_string())


if __name__ == "__main__":
    plot_fold_tissue_distribution()
