import os
import glob

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


metric_display = {
    'auc':       'Area Under Curve (AUC)',
    'auprc':     'AUPRC',
    'f1':        'F1',
    'accuracy':  'Accuracy',
    'precision': 'Precision',
    'recall':    'Recall',
}


def _collect_metrics(run_dir, run_prefix, split):
    """Read every run's summary_results.csv and return one row of metrics per run."""
    paths = sorted(glob.glob(os.path.join(run_dir, f"{run_prefix}_*", "summary_results.csv")))
    records = []
    for path in paths:
        df = pd.read_csv(path, index_col=0)
        if split not in df.index:
            print(f"Warning: split '{split}' not in {path}, skipping")
            continue
        # Columns are saved as e.g. 'response_auc' -> strip the label prefix.
        row = {col.split("_", 1)[-1]: float(val) for col, val in df.loc[split].items()}
        records.append(row)
    return pd.DataFrame(records)


def _read_summary(path, split):
    """Read a single summary_results.csv and return its metrics for one split."""
    if not os.path.exists(path):
        print(f"Warning: {path} not found, no chosen-seed line will be drawn")
        return {}
    df = pd.read_csv(path, index_col=0)
    if split not in df.index:
        print(f"Warning: split '{split}' not in {path}, no chosen-seed line will be drawn")
        return {}
    return {col.split("_", 1)[-1]: float(val) for col, val in df.loc[split].items()}


def plot_network_order_variation(run_dir, figures_dir,
                                 run_prefix="pnet_network_order_variation",
                                 split="test",
                                 chosen_seed_run="pnet_network_order_fixed_seed_0"):
    """
    Plot the distribution of each test metric across the network-order variation
    runs (each run differs only in the network construction seed). One .png per
    metric is written to figures_dir/network_order_variation/.

    args:
        run_dir (str)         : directory holding the per-run output folders
        figures_dir (str)     : directory to write figures into
        run_prefix (str)      : run_id prefix shared by the variation runs
        split (str)           : which split to summarise ('train', 'val' or 'test')
        chosen_seed_run (str) : run_id whose metrics give the chosen-seed (42)
                                reference line drawn on each plot
    """
    data = _collect_metrics(run_dir, run_prefix, split)
    if data.empty:
        print(f"No results found for '{run_prefix}_*' under {run_dir}, nothing to plot.")
        return

    chosen = _read_summary(os.path.join(run_dir, chosen_seed_run, "summary_results.csv"), split)

    out_dir = os.path.join(figures_dir, "network_order_variation")
    os.makedirs(out_dir, exist_ok=True)

    fontsize = 8
    fontproperties = {'family': 'Arial', 'weight': 'normal', 'size': 9}
    sns.set_style("whitegrid")

    # Fixed metric order (matches the other figures), restricted to what is present
    metric_cols = [m for m in metric_display if m in data.columns]

    def draw_metric(ax, metric, fs=fontsize, fp=fontproperties):
        values = data[metric].dropna()
        mean, std = values.mean(), values.std()
        label = metric_display.get(metric, metric)

        sns.histplot(values, kde=True, ax=ax, color=sns.color_palette()[0],
                     edgecolor="white")

        # Reference line for our chosen seed (42)
        if metric in chosen:
            ax.axvline(chosen[metric], ls='--', linewidth=1, color="black",
                       label=f"chosen seed (42) = {chosen[metric]:.3f}")
            ax.legend(fontsize=fs, frameon=False)

        ax.set_title(f"{label}  (n={len(values)}, mean={mean:.3f}, sd={std:.3f})", fp)
        ax.set_xlabel(label, fp)
        ax.set_ylabel("Count", fp)
        ax.tick_params(axis='both', which='major', labelsize=fs)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Individual figure per metric
    for metric in metric_cols:
        if data[metric].dropna().empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        draw_metric(ax, metric)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{run_prefix}_{split}_{metric}.png"), dpi=300)
        plt.close()

    # Combined figure with every metric as a labelled panel (a, b, c, ...),
    # drawn with slightly larger fonts than the individual figures
    combined_fs = 12
    combined_fp = {'family': 'Arial', 'weight': 'normal', 'size': 13}
    n_cols = 2
    n_rows = -(-len(metric_cols) // n_cols)  # ceil division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = axes.flatten()
    for i, metric in enumerate(metric_cols):
        ax = axes[i]
        draw_metric(ax, metric, fs=combined_fs, fp=combined_fp)
        ax.text(-0.08, 1.02, chr(ord('a') + i), transform=ax.transAxes,
                fontsize=20, fontweight='bold', va='bottom', ha='right')
    for j in range(len(metric_cols), len(axes)):
        axes[j].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{run_prefix}_{split}_all.png"), dpi=300)
    plt.close(fig)

    # Save the collated metrics alongside the figures for reference.
    data.to_csv(os.path.join(out_dir, f"{run_prefix}_{split}_metrics.csv"), index=False)
    print(f"Wrote {len(metric_cols)} metric distributions (+ combined) for "
          f"{len(data)} runs to {out_dir}")
