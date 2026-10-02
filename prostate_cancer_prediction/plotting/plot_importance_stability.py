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


def _membership_source_data(top, top_k, label_col):
    """Return the plotted bars of a top-K membership figure as a dataframe.

    ``top`` is the consensus-ordered table the bars were drawn from. One row per
    bar, top of the figure first; ``mean_rank`` is the value that sets each bar's
    colour, and ``colour_scale_min``/``max`` are the limits of the colour bar.
    """
    data = pd.DataFrame({
        "position": range(1, len(top) + 1),
        "feature": top.index,
    })
    if label_col:
        data["name"] = top[label_col].to_numpy()
    data["top_k"] = top_k
    data["topk_frequency"] = top["topk_frequency"].to_numpy()
    data["n_present"] = top["n_folds_present"].to_numpy()
    data["mean_rank"] = top["mean_rank"].to_numpy()
    data["median_rank"] = top["median_rank"].to_numpy()
    data["colour_scale_min"] = top["mean_rank"].min()
    data["colour_scale_max"] = top["mean_rank"].max()
    return data


def _violin_source_data(values, feats, label_of, label_col, value_name,
                        selected_by, top, repeat_dirs, unit):
    """Return the plotted points of a violin figure as a long dataframe.

    One row per feature per repeat (features top of the figure first), holding the
    value the violin is estimated from, plus the median drawn as the black line
    and the statistic the top-N features were selected on. NaNs are dropped, as
    they are before plotting. ``repeat_dirs`` maps each repeat column to the
    directory its feature_importance_*.csv was read from.
    """
    frames = []
    for position, f in enumerate(feats, start=1):
        s = values.loc[f].dropna()
        frame = pd.DataFrame({"position": position, "feature": f,
                              unit: s.index, value_name: s.to_numpy()})
        if label_col:
            frame.insert(2, "name", label_of[f])
        frame["source_dir"] = [repeat_dirs.get(r, "") for r in s.index]
        frame["median"] = np.median(s.to_numpy())
        frame[selected_by] = top.loc[f, selected_by]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


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
    fig.savefig(f"{out_dir}/{display}_top{top_k}_membership.pdf")
    plt.close(fig)

    # bars were drawn bottom-up from the reversed table; undo that for the CSV
    _membership_source_data(top.iloc[::-1], top_k, label_col).to_csv(
        f"{out_dir}/{display}_top{top_k}_membership_source_data.csv", index=False)


def _plot_top_importance(wide, table, display, top_n, out_dir, label_col=None, unit="fold",
                         repeat_dirs=None):
    """Save a violin plot of importance score distributions for the top-N features by mean importance score."""
    top = table.sort_values("mean_importance", ascending=False).head(top_n)
    feats = list(top.index)
    label_of = dict(zip(feats, list(top[label_col]) if label_col else feats))

    order = feats[::-1]                 # reverse so rank 1 is at the top of the axis
    sub = wide.loc[order]
    y = np.arange(len(order))
    data = [sub.loc[f].dropna().to_numpy() for f in order]

    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.5 * len(order))))

    parts = ax.violinplot(data, positions=y, vert=False,
                          showmedians=True, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor("C3")
        body.set_alpha(0.7)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(2)

    ax.set_yticks(y)
    ax.set_yticklabels([label_of[f] for f in order], fontsize=TICK_SIZE)
    ax.set_xlabel("Feature importance score", fontsize=LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(y=0.05)

    fig.tight_layout()
    fig.savefig(f"{out_dir}/{display}_top{top_n}_importance.pdf")
    plt.close(fig)

    _violin_source_data(wide, feats, label_of, label_col, "importance", "mean_importance",
                        top, repeat_dirs or {}, unit).to_csv(
        f"{out_dir}/{display}_top{top_n}_importance_source_data.csv", index=False)


def _plot_top_rank(ranks, table, display, top_n, out_dir, label_col=None, unit="fold",
                   repeat_dirs=None):
    """Save a violin plot of rank distributions for the top-N features by mean rank."""
    top = table.sort_values("mean_rank", ascending=True).head(top_n)
    feats = list(top.index)
    label_of = dict(zip(feats, list(top[label_col]) if label_col else feats))

    order = feats[::-1]                 # reverse so rank 1 is at the top of the axis
    sub = ranks.loc[order]
    y = np.arange(len(order))
    data = [sub.loc[f].dropna().to_numpy() for f in order]

    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.5 * len(order))))

    parts = ax.violinplot(data, positions=y, vert=False,
                          showmedians=True, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor("C0")
        body.set_alpha(0.7)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(2)

    ax.set_yticks(y)
    ax.set_yticklabels([label_of[f] for f in order], fontsize=TICK_SIZE)
    ax.set_xlabel("Rank", fontsize=LABEL_SIZE)
    ax.set_xlim(left=0)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(y=0.05)

    fig.tight_layout()
    fig.savefig(f"{out_dir}/{display}_top{top_n}_rank.pdf")
    plt.close(fig)

    _violin_source_data(ranks, feats, label_of, label_col, "rank", "mean_rank",
                        top, repeat_dirs or {}, unit).to_csv(
        f"{out_dir}/{display}_top{top_n}_rank_source_data.csv", index=False)


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
      * {layer}_top{K}_membership.pdf  -- top-K membership frequency bars
      * {layer}_top{K}_importance.pdf  -- mean +/- SD importance of the top-K features
      * {layer}_top{K}_rank.pdf        -- median & IQR rank of the top-K features
      * {figure stem}_source_data.csv  -- the plotted values behind each figure
      * stability_summary.csv          -- one row per layer (mean Spearman)
    """
    if fold_dirs is None:
        fold_dirs = [f"{run_dir}/{run_id}/test_{i}/best_{best_metric}"
                     for i in range(n_folds)]

    out_dir = f"{figures_dir}/importance_stability/{run_id}"
    os.makedirs(out_dir, exist_ok=True)

    # repeat column name (as _load_fold_importance labels it) -> its directory
    repeat_dirs = {f"{unit}_{i}": d for i, d in enumerate(fold_dirs)}

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
        _plot_top_importance(wide, table, display, 15, out_dir, label_col=label_col, unit=unit,
                             repeat_dirs=repeat_dirs)
        _plot_top_rank(ranks, table, display, 15, out_dir, label_col=label_col, unit=unit,
                       repeat_dirs=repeat_dirs)

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
