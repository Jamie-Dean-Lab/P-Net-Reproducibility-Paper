import os
import glob

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


metric_display = {
    'auc':       'AUROC',
    'auprc':     'AUPRC',
    'f1':        'F1',
    'accuracy':  'Accuracy',
    'precision': 'Precision',
    'recall':    'Recall',
}


def _collect_metrics(run_dir, run_prefix, split):
    """Read every run's summary_results.csv and return one row of metrics per run,
    plus the run_id of each row (same order)."""
    paths = sorted(glob.glob(os.path.join(run_dir, f"{run_prefix}_*", "summary_results.csv")))
    records = []
    run_ids = []
    for path in paths:
        df = pd.read_csv(path, index_col=0)
        if split not in df.index:
            print(f"Warning: split '{split}' not in {path}, skipping")
            continue
        # Columns are saved as e.g. 'response_auc' -> strip the label prefix.
        row = {col.split("_", 1)[-1]: float(val) for col, val in df.loc[split].items()}
        records.append(row)
        run_ids.append(os.path.basename(os.path.dirname(path)))
    return pd.DataFrame(records), run_ids


def _histogram_source_data(ax, values, run_ids, label, chosen_seed_run, reference):
    """Return the plotted histogram of one metric as (bins, runs) dataframes.

    ``bins`` is read back from the bars seaborn drew, so the edges and counts are
    exactly those in the figure: one row per bar, empty bins included. ``runs`` has
    one row per run with the bin it was counted in (bins are left-closed, the last
    one closed on both sides, as in numpy.histogram). Both carry the chosen-seed
    value that sets the dashed line (NaN if that run's summary was missing).
    """
    bars = sorted(ax.patches, key=lambda p: p.get_x())
    left = np.array([p.get_x() for p in bars])
    right = left + np.array([p.get_width() for p in bars])
    bins = pd.DataFrame({
        "metric": label,
        "bin": range(len(bars)),
        "bin_left": left,
        "bin_right": right,
        "count": [int(round(p.get_height())) for p in bars],
        "reference_run": chosen_seed_run,
        "reference_value": reference,
    })
    bin_index = np.clip(np.searchsorted(left, values.to_numpy(), side="right") - 1,
                        0, len(bars) - 1)
    runs = pd.DataFrame({
        "metric": label,
        "run_id": [run_ids[i] for i in values.index],
        "value": values.to_numpy(),
        "bin": bin_index,
        "reference_run": chosen_seed_run,
        "reference_value": reference,
    })
    return bins, runs


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
    runs (each run differs only in the network construction seed). One .pdf per
    metric is written to figures_dir/network_order_variation/.

    args:
        run_dir (str)         : directory holding the per-run output folders
        figures_dir (str)     : directory to write figures into
        run_prefix (str)      : run_id prefix shared by the variation runs
        split (str)           : which split to summarise ('train', 'val' or 'test')
        chosen_seed_run (str) : run_id whose metrics give the chosen-seed (42)
                                reference line drawn on each plot
    """
    data, run_ids = _collect_metrics(run_dir, run_prefix, split)
    if data.empty:
        print(f"No results found for '{run_prefix}_*' under {run_dir}, nothing to plot.")
        return

    chosen = _read_summary(os.path.join(run_dir, chosen_seed_run, "summary_results.csv"), split)

    out_dir = os.path.join(figures_dir, "network_order_variation")
    os.makedirs(out_dir, exist_ok=True)

    # consistent fonts across the individual and combined figures (and with the
    # other figures in this project, e.g. plot_stratified_5_fold_CV)
    fontsize = 16
    fontproperties = {'family': 'Arial', 'weight': 'normal', 'size': 18}
    sns.set_style("white")

    # Fixed metric order (matches the other figures), restricted to what is present
    metric_cols = [m for m in metric_display if m in data.columns]

    def draw_metric(ax, metric, fs=fontsize, fp=fontproperties):
        values = data[metric].dropna()
        label = metric_display.get(metric, metric)

        sns.histplot(values, kde=False, ax=ax, color="#3D5A80",
                     edgecolor="white")

        # Reference line for our chosen seed
        if metric in chosen:
            ax.axvline(chosen[metric], ls='--', linewidth=1.2, color="0.3")

        ax.set_xlabel(label, fp)
        ax.set_ylabel("Count", fp)
        ax.tick_params(axis='both', which='major', labelsize=fs)
        ax.grid(False)  # no gridlines for a cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Individual figure per metric
    for metric in metric_cols:
        if data[metric].dropna().empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        draw_metric(ax, metric)
        plt.tight_layout()
        # 'auc' is the data-column key; the file is named with the AUROC label
        slug = "auroc" if metric == "auc" else metric
        stem = f"{run_prefix}_{split}_{slug}"
        plt.savefig(os.path.join(out_dir, f"{stem}.pdf"))
        # read the drawn bars back before the figure is closed
        bins, runs = _histogram_source_data(
            ax, data[metric].dropna(), run_ids, metric_display.get(metric, metric),
            chosen_seed_run, chosen.get(metric, np.nan))
        plt.close()
        bins.to_csv(os.path.join(out_dir, f"{stem}_bins_source_data.csv"), index=False)
        runs.to_csv(os.path.join(out_dir, f"{stem}_runs_source_data.csv"), index=False)

    # Save the collated metrics alongside the figures for reference.
    data.to_csv(os.path.join(out_dir, f"{run_prefix}_{split}_metrics.csv"), index=False)
    print(f"Wrote {len(metric_cols)} metric distributions (+ combined) for "
          f"{len(data)} runs to {out_dir}")
