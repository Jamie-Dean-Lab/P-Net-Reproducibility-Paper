"""
Shared model-comparison figures.

Every task draws the same two figures: a per-fold distribution of each test metric
across models (nested crossvalidation), and a single bar per model (external
validation, which is one unreplicated fit). Only the model list, the metric list
and a few per-task conventions differ, so those are declared in a ModelRegistry and
the drawing lives here.

The registry is also what significance_testing imports, so a model's directory name
and display name cannot disagree between a figure and the significance table.
"""

import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import ticker

# seaborn's stripplot jitter draws from the global numpy RNG. Without a fixed seed
# the same data produces visibly different figures on every run.
JITTER_SEED = 0

FONTSIZE = 18
FONTPROPERTIES = {"family": "Arial", "weight": "normal", "size": 20}


class ModelRegistry:
    """
    The models a task compares, in one place.

    args:
        names (list[str])   : run directory names, in the order results are loaded
        display (dict)      : run directory name -> name shown in figures and tables
        order (list[str])   : display names in the left-to-right order used on the
                              x-axis; also fixes the colour assignment
        reference (str)     : display name of the model the dashed line marks
        col_prefix (str)    : label prefix stripped from result columns, e.g.
                              "response_" for a label column named "response"
    """

    def __init__(self, names, display, order, reference="P-NET", col_prefix=""):
        self.names = names
        self.display = display
        self.order = order
        self.reference = reference
        self.col_prefix = col_prefix
        palette = sns.color_palette(None, len(order))
        self.palette = {m: palette[i] for i, m in enumerate(order)}

    def label(self, name):
        return self.display.get(name, name)

    def present(self, columns):
        """Display names from `order` that are actually present, preserving order."""
        return [m for m in self.order if m in columns]


def _strip(df, col_prefix):
    if col_prefix:
        df = df.copy()
        df.columns = [c.replace(col_prefix, "") for c in df.columns]
    return df


def load_per_fold_scores(run_dir, registry, metrics, selection_metric, strict=True):
    """
    Per-fold test scores from ``<model>/test_*/best_<selection_metric>/summary_results.csv``.

    args:
        strict (bool) : when True a model missing any requested metric raises, which is
                        the historical behaviour for the binary-classification tasks;
                        when False the missing columns are simply skipped

    returns:
        DataFrame with a (metric, model) MultiIndex on the columns, or None if nothing loaded
    """
    all_data = []
    for name in registry.names:
        model_dir = os.path.join(run_dir, name)
        if not os.path.isdir(model_dir):
            print(f"Warning: {model_dir} not found, skipping")
            continue

        fold_scores = []
        for test_dir in sorted(d for d in os.listdir(model_dir) if d.startswith("test_")):
            path = os.path.join(model_dir, test_dir, f"best_{selection_metric}", "summary_results.csv")
            if not os.path.exists(path):
                print(f"Warning: {path} not found, skipping")
                continue
            df = _strip(pd.read_csv(path, index_col=0), registry.col_prefix)
            cols = metrics if strict else [c for c in metrics if c in df.columns]
            fold_scores.append(df[df.index == "test"][cols])

        if fold_scores:
            all_data.append(_tag(pd.concat(fold_scores).reset_index(drop=True), name))

    return _combine(all_data)


def load_run_scores(run_dir, registry, metrics):
    """
    Per-fold test scores from a single ``<model>/results.csv`` whose ``index`` column
    marks the held-out ``test`` rows.

    returns:
        DataFrame with a (metric, model) MultiIndex on the columns, or None if nothing loaded
    """
    all_data = []
    for name in registry.names:
        path = os.path.join(run_dir, name, "results.csv")
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping")
            continue
        df = _strip(pd.read_csv(path, index_col=0), registry.col_prefix)
        rows = df[df["index"] == "test"][[c for c in metrics if c in df.columns]]
        all_data.append(_tag(rows.reset_index(drop=True), name))

    return _combine(all_data)


def _tag(model_df, name):
    model_df = model_df.copy()
    model_df.columns = pd.MultiIndex.from_tuples([(name, col) for col in model_df.columns])
    return model_df


def _combine(all_data):
    if not all_data:
        print("No results found — have the runs completed?")
        return None
    combined = pd.concat(all_data, axis=1)
    combined.columns = combined.columns.swaplevel(0, 1)
    return combined


def _finish_axes(ax, ylabel):
    """Axis styling shared by both figure types."""
    ax.set_ylabel(ylabel, FONTPROPERTIES)
    ax.set_xlabel("")
    ax.get_xaxis().set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(axis="both", which="major", labelsize=FONTSIZE)
    ax.minorticks_off()
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)


def plot_fold_distribution(combined, registry, metric_display, figures_dir,
                           filename, bounded_metrics=None):
    """
    One figure per metric: each fold's score as a jittered point per model, with a
    mean +/- sd marker and a dashed line at the reference model's mean.

    args:
        combined (DataFrame)   : output of load_per_fold_scores / load_run_scores
        metric_display (dict)  : metric key -> axis label
        filename (callable)    : metric key -> output filename (without directory)
        bounded_metrics (set)  : metrics capped at 1.02 on the y-axis; None means all
    """
    if combined is None:
        return
    if bounded_metrics is None:
        bounded_metrics = set(metric_display)

    sns.set_style("white")
    os.makedirs(figures_dir, exist_ok=True)

    for metric in metric_display:
        if metric not in combined.columns.get_level_values(0):
            continue

        dd = combined[metric].copy()
        dd.columns = [registry.label(c) for c in dd.columns]
        means, stds = dd.mean(), dd.std()
        order = registry.present(dd.columns)

        fig, ax = plt.subplots(figsize=(10, 5))
        # seeded per figure, so each one is independently reproducible
        np.random.seed(JITTER_SEED)
        sns.stripplot(
            ax=ax, x="variable", y="value", hue="variable", data=dd.melt(),
            order=order, palette=registry.palette, legend=False,
            size=7, alpha=0.8, jitter=0.15, edgecolor="black", linewidth=0.5,
        )
        ax.errorbar(
            list(range(len(order))), means[order].values, yerr=stds[order].values,
            fmt="_", markersize=20, markeredgewidth=2, color="black",
            capsize=4, elinewidth=1.2, zorder=10,
        )
        ax.axhline(means[registry.reference], ls="--", linewidth=1)
        ax.autoscale(enable=True, axis="y")
        ax.margins(y=0.08)
        if metric in bounded_metrics:
            ax.set_ylim(top=min(ax.get_ylim()[1], 1.02))
        # tick labels before _finish_axes, so its tick_params call still lands last
        # exactly as in the original per-task plotters
        ax.set_ylabel(metric_display[metric], FONTPROPERTIES)
        ax.set_xlabel("")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30,
                           horizontalalignment="right", fontsize=FONTSIZE)
        _finish_axes(ax, metric_display[metric])

        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, filename(metric)))
        plt.close()


def load_single_values(run_dir, registry, metrics, dataset_tag, subdir="external_validation"):
    """
    One score per model from ``<model>/<subdir>/<tag>/metrics.csv``.

    returns:
        DataFrame indexed by display name, columns = metrics (missing ones absent)
    """
    records = {}
    for name in registry.names:
        path = os.path.join(run_dir, name, subdir, dataset_tag, "metrics.csv")
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping")
            continue
        df = _strip(pd.read_csv(path), registry.col_prefix)
        records[registry.label(name)] = {c: df.iloc[0][c] for c in metrics if c in df.columns}

    if not records:
        print("No external validation results found.")
        return None
    return pd.DataFrame(records).T


def plot_single_values(scores, registry, metric_display, figures_dir, filename,
                       bounded_metrics=None):
    """
    One figure per metric: a single bar per model, with a dashed line at the
    reference model. Used for external validation, which is one fit per model and
    therefore has no spread to show.
    """
    if scores is None:
        return
    if bounded_metrics is None:
        bounded_metrics = set(metric_display)

    sns.set_style("white")
    os.makedirs(figures_dir, exist_ok=True)
    order = registry.present(scores.index)

    for metric in metric_display:
        if metric not in scores.columns:
            continue
        vals = scores.loc[order, metric]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(range(len(order)), vals.values,
               color=[registry.palette[m] for m in order],
               edgecolor="black", linewidth=0.5)
        ax.axhline(vals[registry.reference], ls="--", linewidth=1)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=30, horizontalalignment="right", fontsize=FONTSIZE)
        ax.set_xlim(-0.5, len(order) - 0.5)

        ymin = min(vals.min() * 1.15, 0)
        ymax = max(vals.max() * 1.15, 0)
        if metric in bounded_metrics:
            ymax = min(ymax, 1.02)
        if ymax == 0:
            ymax = abs(ymin) * 0.05
        ax.set_ylim(ymin, ymax)
        _finish_axes(ax, metric_display[metric])

        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, filename(metric)))
        plt.close()
