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
import seaborn as sns
from scipy.stats import spearmanr


def _load_fold_importance(run_dir, run_id, layer_key, value_col, n_folds,
                          best_metric, connected_only):
    """Load one importance column from every outer fold into a wide table.

    Returns a DataFrame indexed by feature with one column per fold
    ('fold_{i}') holding the absolute `value_col`. Missing fold files are
    skipped with a warning.
    """
    series_by_fold = {}
    for i in range(n_folds):
        path = (f"{run_dir}/{run_id}/test_{i}/best_{best_metric}/"
                f"feature_importance_{layer_key}.csv")
        if not os.path.exists(path):
            print(f"  [warn] missing {path} -- skipping fold {i}")
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
        series_by_fold[f"fold_{i}"] = s
    if not series_by_fold:
        raise FileNotFoundError(
            f"No feature_importance_{layer_key}.csv found under "
            f"{run_dir}/{run_id}/test_*/best_{best_metric}/")
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


def _plot_topk_membership(table, display, top_k, out_dir, label_col=None, top_n=20):
    """Save top-K membership frequency bars for the consensus features."""
    top = table.head(top_n).iloc[::-1]  # reverse so rank 1 sits at the top of barh
    labels = top[label_col] if label_col else top.index

    fig, ax = plt.subplots(figsize=(7, max(4.0, 0.38 * len(top))))

    # how many folds place each consensus feature in the top-K,
    # coloured by mean rank (darker = better average rank)
    rng = top["mean_rank"].max() - top["mean_rank"].min()
    norm = (top["mean_rank"] - top["mean_rank"].min()) / (rng + 1e-9)
    ax.barh(range(len(top)), top["topk_frequency"],
            color=plt.cm.viridis_r(norm))
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel(f"# folds with feature in top-{top_k}")
    ax.set_title(f"{display}: top-{top_k} membership across folds")

    fig.tight_layout()
    fig.savefig(f"{out_dir}/{display}_top{top_k}_membership.png", dpi=200)
    plt.close(fig)


def _plot_pairwise_spearman(spear, display, out_dir):
    """Save the pairwise Spearman heatmap of the full per-fold rankings."""
    fig, ax = plt.subplots(figsize=(8, 7))

    sns.heatmap(spear, vmin=0, vmax=1, annot=True, fmt=".2f", cmap="rocket",
                ax=ax, cbar=True, annot_kws={"size": 6})
    ax.set_title(f"{display}: pairwise Spearman of rankings")

    fig.tight_layout()
    fig.savefig(f"{out_dir}/{display}_spearman.png", dpi=200)
    plt.close(fig)


def _plot_top_importance(wide, table, display, top_k, out_dir, label_col=None):
    """Save a figure of mean +/- 1 SD importance for the top-K consensus features.
    """
    top = table.head(top_k)
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
                   color="0.65", s=14, zorder=2)

    # mean +/- 1 SD across folds
    ax.errorbar(means, y, xerr=stds, fmt="o", color="C3", capsize=3,
                markersize=5, lw=1.2, zorder=3, label="mean +/- 1 SD across folds")

    ax.set_yticks(y)
    ax.set_yticklabels([label_of[f] for f in order], fontsize=7)
    ax.set_xlabel("DeepLIFT importance score")
    ax.set_title(f"{display}: top-{top_k} importance (mean & variance across folds)")
    ax.legend(fontsize=7, loc="lower right")
    ax.margins(y=0.02)

    fig.tight_layout()
    fig.savefig(f"{out_dir}/{display}_top{top_k}_importance.png", dpi=200)
    plt.close(fig)


def _plot_top_rank(ranks, table, display, top_k, out_dir, label_col=None):
    """Save a figure of median rank with IQR (Q1-Q3) for the top-K features.

    Rank is bounded, discrete and typically right-skewed across folds, so the
    median and inter-quartile range describe its spread more faithfully than
    mean +/- SD (which can place whiskers below rank 1).
    """
    top = table.head(top_k)
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
                   color="0.65", s=14, zorder=2)

    # median with IQR (Q1-Q3) across folds
    ax.errorbar(medians, y, xerr=xerr, fmt="o", color="C0", capsize=3,
                markersize=5, lw=1.2, zorder=3, label="median +/- IQR across folds")

    ax.set_yticks(y)
    ax.set_yticklabels([label_of[f] for f in order], fontsize=7)
    ax.set_xlabel("rank within fold (1 = most important)")
    ax.set_title(f"{display}: top-{top_k} rank (median & IQR across folds)")
    ax.set_xlim(left=0)
    ax.legend(fontsize=7, loc="lower right")
    ax.margins(y=0.02)

    fig.tight_layout()
    fig.savefig(f"{out_dir}/{display}_top{top_k}_rank.png", dpi=200)
    plt.close(fig)


def analyse_importance_stability(run_dir, figures_dir, n_hidden_layers,
                                 run_id="pnet_10_fold_CV_stability",
                                 top_k=10, best_metric="auc", n_folds=10,
                                 pathway_names="architecture/Reactome/ReactomePathways.txt"):
    """Run the full stability analysis and write CSVs + figures.

    Outputs to {figures_dir}/importance_stability/{run_id}/:
      * {layer}_stability.csv          -- per-feature metrics (consensus-ordered)
      * {layer}_top{K}_membership.png  -- top-K membership frequency bars
      * {layer}_spearman.png           -- pairwise Spearman heatmap of rankings
      * {layer}_top{K}_importance.png  -- mean +/- SD importance of the top-K features
      * {layer}_top{K}_rank.png        -- median & IQR rank of the top-K features
      * stability_summary.csv          -- one row per layer (mean Spearman)
    """
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
            wide = _load_fold_importance(run_dir, run_id, layer_key, value_col,
                                         n_folds, best_metric, connected_only)
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

        print(f"  features: {len(table)}, folds used: {n_used}")
        print(f"  mean pairwise Spearman: {mean_spear:.3f}")
        print(f"  features in top-{top_k} of ALL folds: {n_in_all}")
        cols = (["name"] if label_col else []) + ["topk_frequency", "median_rank", "iqr_rank", "mean_importance", "std_importance"]
        print(f"  top consensus features:\n{table.head(top_k)[cols].to_string()}")

        _plot_topk_membership(table, display, top_k, out_dir, label_col=label_col)
        _plot_pairwise_spearman(spear, display, out_dir)
        _plot_top_importance(wide, table, display, top_k, out_dir, label_col=label_col)
        _plot_top_rank(ranks, table, display, top_k, out_dir, label_col=label_col)

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
