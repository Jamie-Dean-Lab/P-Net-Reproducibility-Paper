"""
Feature-importance stability analysis across cross-validation folds.

This module answers: which genes / pathways are consistently most important to
P-NET's prediction

It therefore ranks features directly from the saved importance columns -- the
exact quantities plot_sankey uses for node *selection* -- and ignores everything
downstream that is a plotting choice.

  * genes (layer h0)        -> ranked by `coef_combined` (degree-adjusted),
                               restricted to genes connected in the Reactome
                               graph (coef_graph > 0), matching plot_sankey.
  * pathways (layers h1..h5)-> ranked by raw `coef`, matching plot_sankey.

Stability is summarised per layer with:
  * top-K membership frequency per feature (how many folds rank it in the top K),
  * mean / std / best / worst rank across folds,
  * mean and variance (SD) of the importance score across folds,
  * pairwise Spearman correlation of the full rankings between folds.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


# Shared figure styling, consistent with the other figures in this project
# (e.g. plot_stratified_5_fold_CV, plot_network_order_variation).
TICK_SIZE = 14
LABEL_SIZE = 16
DPI = 300


def _load_fold_importance(fold_dirs, layer_key, value_col, connected_only, unit="fold"):
    """Load one importance column from every fold directory into a wide table.

    `fold_dirs` is a list of directories, each expected to contain a
    `feature_importance_{layer_key}.csv`. Returns a DataFrame indexed by feature
    with one column per repeat ('{unit}_{i}', e.g. 'fold_0' or 'run_0') holding
    the absolute `value_col`. Missing files are skipped with a warning.
    """
    series_by_fold = {}
    for i, fold_dir in enumerate(fold_dirs):
        path = f"{fold_dir}/feature_importance_{layer_key}.csv"
        if not os.path.exists(path):
            print(f"  [warn] missing {path} -- skipping {unit} {i}")
            continue
        df = pd.read_csv(path, index_col=0)
        if value_col not in df.columns:
            raise KeyError(f"{path} has no column {value_col!r}; "
                           f"columns={df.columns.tolist()}")
        s = df[value_col].abs()
        if connected_only and "coef_graph" in df.columns:
            # genes connected to >=1 pathway (degree>0); equivalent to the
            # link_weights[1].index filter plot_sankey applies to gene selection
            s = s[df["coef_graph"] > 0]
        series_by_fold[f"{unit}_{i}"] = s
    if not series_by_fold:
        raise FileNotFoundError(
            f"No feature_importance_{layer_key}.csv found in any of: {fold_dirs}")
    return pd.DataFrame(series_by_fold)


def _stability_table(wide, top_k):
    """Per-feature stability metrics from a wide (feature x fold) table."""
    # rank 1 = most important (largest importance) within each fold
    ranks = wide.rank(ascending=False, method="min")
    topk_member = ranks.le(top_k)
    table = pd.DataFrame({
        "topk_frequency":  topk_member.sum(axis=1).astype(int),
        "n_folds_present": wide.notna().sum(axis=1).astype(int),
        "median_rank":     ranks.median(axis=1),
        "iqr_rank":        ranks.quantile(0.75, axis=1) - ranks.quantile(0.25, axis=1),
        "mean_rank":       ranks.mean(axis=1),
        "std_rank":        ranks.std(axis=1),
        "best_rank":       ranks.min(axis=1),
        "worst_rank":      ranks.max(axis=1),
        "mean_importance": wide.mean(axis=1),
        "std_importance":  wide.std(axis=1),
        "var_importance":  wide.var(axis=1),
    })
    # consensus order: most frequently in the top-K, ties broken by best median
    # rank (rank is skewed across folds, so the median is more robust than the mean)
    table = table.sort_values(["topk_frequency", "median_rank"],
                              ascending=[False, True])
    return table, ranks, topk_member


def _pairwise_spearman(wide):
    """Spearman correlation of the full ranking between every pair of folds."""
    folds = list(wide.columns)
    n = len(folds)
    mat = np.eye(n)
    for a in range(n):
        for b in range(a + 1, n):
            sub = wide[[folds[a], folds[b]]].dropna()
            rho = (spearmanr(sub.iloc[:, 0], sub.iloc[:, 1]).correlation
                   if len(sub) > 2 else np.nan)
            mat[a, b] = mat[b, a] = rho
    return pd.DataFrame(mat, index=folds, columns=folds)


def _offdiag_mean(square_df):
    """Mean of the off-diagonal entries of a symmetric matrix DataFrame."""
    a = square_df.to_numpy(dtype=float)
    mask = ~np.eye(a.shape[0], dtype=bool)
    return np.nanmean(a[mask])


def _plot_topk_membership(table, display, top_k, out_dir, label_col=None, top_n=20, unit="fold"):
    """Save top-K membership frequency bars for the consensus features."""
    top = table.head(top_n).iloc[::-1]  # reverse so rank 1 sits at the top of barh
    labels = top[label_col] if label_col else top.index

    fig, ax = plt.subplots(figsize=(8, max(4.0, 0.45 * len(top))))

    # how many repeats place each consensus feature in the top-K,
    # coloured by mean rank (lighter/yellow = better, i.e. lower, average rank)
    cmap = plt.cm.viridis_r
    norm = plt.Normalize(vmin=top["mean_rank"].min(), vmax=top["mean_rank"].max())
    ax.barh(range(len(top)), top["topk_frequency"],
            color=cmap(norm(top["mean_rank"].to_numpy())))
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=TICK_SIZE)
    ax.set_xlabel(f"Number of {unit}s with feature in top {top_k}", fontsize=LABEL_SIZE)
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Mean rank", fontsize=LABEL_SIZE)
    cbar.ax.tick_params(labelsize=TICK_SIZE)

    fig.tight_layout()
    fig.savefig(f"{out_dir}/{display}_top{top_k}_membership.png", dpi=DPI)
    plt.close(fig)


def _plot_top_importance(wide, table, display, top_n, out_dir, label_col=None, unit="fold"):
    """Save a figure of mean +/- 1 SD importance for the top-N features by mean score.
    """
    top = table.sort_values("mean_importance", ascending=False).head(top_n)
    feats = list(top.index)
    label_of = dict(zip(feats, list(top[label_col]) if label_col else feats))

    order = feats[::-1]                 # reverse so rank 1 is at the top of the axis
    sub = wide.loc[order]               # feature x fold importance values
    means = sub.mean(axis=1).to_numpy()
    stds = sub.std(axis=1).to_numpy()
    y = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.5 * len(order))))

    # individual per-fold scores (shows the actual variance, not just the SD bar)
    for yi, f in zip(y, order):
        fold_vals = sub.loc[f].dropna().to_numpy()
        ax.scatter(fold_vals, np.full(len(fold_vals), yi),
                   color="0.65", s=22, zorder=2)

    # mean +/- 1 SD across repeats
    ax.errorbar(means, y, xerr=stds, fmt="o", color="C3", capsize=5,
                markersize=7, lw=1.8, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels([label_of[f] for f in order], fontsize=TICK_SIZE)
    ax.set_xlabel("Feature importance score", fontsize=LABEL_SIZE)
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.margins(y=0.02)

    fig.tight_layout()
    fig.savefig(f"{out_dir}/{display}_top{top_n}_importance.png", dpi=DPI)
    plt.close(fig)


def _plot_top_rank(ranks, table, display, top_n, out_dir, label_col=None, unit="fold"):
    """Save a figure of median rank with IQR (Q1-Q3) for the top-N features by median rank.

    Rank is bounded, discrete and typically right-skewed across repeats, so the
    median and inter-quartile range describe its spread more faithfully than
    mean +/- SD (which can place whiskers below rank 1).
    """
    top = table.sort_values(["median_rank", "mean_rank"], ascending=True).head(top_n)
    feats = list(top.index)
    label_of = dict(zip(feats, list(top[label_col]) if label_col else feats))

    order = feats[::-1]                 # reverse so rank 1 is at the top of the axis
    sub = ranks.loc[order]              # feature x fold ranks
    medians = sub.median(axis=1).to_numpy()
    q1 = sub.quantile(0.25, axis=1).to_numpy()
    q3 = sub.quantile(0.75, axis=1).to_numpy()
    # asymmetric whiskers: distance from the median out to each quartile
    xerr = np.vstack([medians - q1, q3 - medians])
    y = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.5 * len(order))))

    # individual per-fold ranks (shows the actual spread, not just the IQR bar)
    for yi, f in zip(y, order):
        fold_vals = sub.loc[f].dropna().to_numpy()
        ax.scatter(fold_vals, np.full(len(fold_vals), yi),
                   color="0.65", s=22, zorder=2)

    # median with IQR (Q1-Q3) across repeats
    ax.errorbar(medians, y, xerr=xerr, fmt="o", color="C0", capsize=5,
                markersize=7, lw=1.8, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels([label_of[f] for f in order], fontsize=TICK_SIZE)
    ax.set_xlabel("Rank", fontsize=LABEL_SIZE)
    ax.set_xlim(left=0)
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.margins(y=0.02)

    fig.tight_layout()
    fig.savefig(f"{out_dir}/{display}_top{top_n}_rank.png", dpi=DPI)
    plt.close(fig)


def analyse_importance_stability(run_dir, figures_dir, n_hidden_layers,
                                 run_id,
                                 top_k=10, best_metric="auc", n_folds=10,
                                 pathway_names="architecture/Reactome/ReactomePathways.txt",
                                 fold_dirs=None, unit="fold"):
    """Run the full stability analysis and write CSVs + figures.

    `fold_dirs` is the list of directories holding each repeat's
    `feature_importance_*.csv`. When omitted it defaults to the cross-validation
    layout ({run_dir}/{run_id}/test_{i}/best_{best_metric} for i in range(n_folds));
    pass it explicitly to analyse a set of separate single-split runs instead (e.g.
    the network-order variation runs). `run_id` only names the output subdirectory.

    `unit` is the noun used for each repeat in figure labels and the Spearman
    heatmap tick labels: "fold" for cross-validation folds (default), or "run" for
    the network-order variation runs (fixed split, only the network seed varies).

    Outputs to {figures_dir}/importance_stability/{run_id}/:
      * {layer}_stability.csv          -- per-feature metrics (consensus-ordered)
      * {layer}_top{K}_membership.png  -- top-K membership frequency bars
      * {layer}_top{K}_importance.png  -- mean +/- SD importance of the top-K features
      * {layer}_top{K}_rank.png        -- median & IQR rank of the top-K features
      * stability_summary.csv          -- one row per layer (mean Spearman)
    """
    if fold_dirs is None:
        fold_dirs = [f"{run_dir}/{run_id}/test_{i}/best_{best_metric}"
                     for i in range(n_folds)]

    out_dir = f"{figures_dir}/importance_stability/{run_id}"
    os.makedirs(out_dir, exist_ok=True)

    # pathway id -> human-readable name (col0=id, col1=name, col2=namespace);
    # same tab-separated format for Reactome (ReactomePathways.txt) and GO
    # (go_id_name_map.tsv)
    id_to_name = {}
    if pathway_names and os.path.exists(pathway_names):
        names = pd.read_csv(pathway_names, sep="\t", header=None, index_col=0)
        id_to_name = names[1].to_dict()

    # (layer key, ranking column, display name, restrict to connected genes)
    layers = [("h0", "coef_combined", "genes", True)]
    for i in range(1, n_hidden_layers + 1):
        layers.append((f"h{i}", "coef", f"pathway_layer_{i}", False))

    summary_rows = []
    for layer_key, value_col, display, connected_only in layers:
        print(f"\n=== {display} ({layer_key}, ranked by '{value_col}') ===")
        try:
            wide = _load_fold_importance(fold_dirs, layer_key, value_col,
                                         connected_only, unit=unit)
        except (FileNotFoundError, KeyError) as e:
            print(f"  {e}")
            continue

        table, ranks, _ = _stability_table(wide, top_k)
        spear = _pairwise_spearman(wide)

        label_col = None
        if layer_key != "h0":
            table.insert(0, "name", [id_to_name.get(idx, idx) for idx in table.index])
            label_col = "name"

        table.to_csv(f"{out_dir}/{display}_stability.csv")

        n_used = wide.shape[1]
        mean_spear = _offdiag_mean(spear)
        n_in_all = int((table["topk_frequency"] == n_used).sum())

        print(f"  features: {len(table)}, {unit}s used: {n_used}")
        print(f"  mean pairwise Spearman: {mean_spear:.3f}")
        print(f"  features in top-{top_k} of ALL {unit}s: {n_in_all}")
        cols = (["name"] if label_col else []) + ["topk_frequency", "median_rank", "iqr_rank", "mean_importance", "std_importance"]
        print(f"  top consensus features:\n{table.head(top_k)[cols].to_string()}")

        _plot_topk_membership(table, display, top_k, out_dir, label_col=label_col, unit=unit)
        _plot_top_importance(wide, table, display, 15, out_dir, label_col=label_col, unit=unit)
        _plot_top_rank(ranks, table, display, 15, out_dir, label_col=label_col, unit=unit)

        summary_rows.append({
            "layer": display,
            "value_col": value_col,
            "n_features": len(table),
            "n_folds": n_used,
            "mean_spearman": mean_spear,
            f"n_in_all_folds_top{top_k}": n_in_all,
        })

    pd.DataFrame(summary_rows).to_csv(f"{out_dir}/stability_summary.csv", index=False)
    print(f"\nSaved importance stability analysis to {out_dir}")


if __name__ == "__main__":
    wd = "prostate_cancer_prediction"
    analyse_importance_stability(
        run_dir=f"{wd}/runs",
        figures_dir=f"{wd}/figures",
        n_hidden_layers=5,
        run_id="pnet_10_fold_CV_stability",
        top_k=10,
    )
