import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum
from scipy.stats import ttest_ind

from statsmodels.stats.multitest import multipletests

from architecture.significance_utils import corrected_paired_stats


class FigureComparativeAnalysisConfiguration(Enum):
    plot_size = (10, 7)

    pnet_auroc_color = 'blue'
    dense_auroc_color = 'orange'

    marker = "."
    marker_size = 14

    pnet_label = 'P-NET'
    dense_label = 'Dense'

    legend_fontsize = 18
    legend_frame = False

    top_spine_visibility = False
    bottom_spine_visibility = True
    left_spine_visibility = True
    right_spine_visibility = False

    spine_thickness = 1
    tick_size = 18
    label_size = 20


# Fixed limits for metrics that are well-behaved on [0, 1];
# None means let matplotlib autoscale (used for F1, precision, recall).
METRICS = ["auc", "auprc", "f1", "accuracy", "precision", "recall"]

METRICS_Y_LIMITS = {
    "auc": (0.4, 1.0),
    "auprc": (0.4, 1.0),
    "f1": None,
    "accuracy": None,
    "precision": None,
    "recall": None,
}

# Display labels / filename slugs for metrics whose identifier differs from how
# they should be shown. The "auc" column is reported as AUROC everywhere.
METRIC_DISPLAY = {
    "auc": "AUROC",
}


class ComparativeAnalysis:

    def __init__(self, results):
        self.results = results
        self.config = FigureComparativeAnalysisConfiguration

    def process_results(self):
        self.number_of_samples = np.array(self.results['number_of_samples'])
        self.pnet_auroc = np.array(self.results['pnet_auroc'])
        self.dense_auroc = np.array(self.results['dense_auroc'])
        self.pnet_lower_bound = np.array(self.results['pnet_lower_bound'])
        self.pnet_upper_bound = np.array(self.results['pnet_upper_bound'])
        self.dense_lower_bound = np.array(self.results['dense_lower_bound'])
        self.dense_upper_bound = np.array(self.results['dense_upper_bound'])
        self.statistically_significant = np.array(self.results['statistically_significant'])

    def compute_xticks(self):
        maximum_x_value = max(self.number_of_samples)
        minimum_x_value = min(self.number_of_samples)
        space = int((maximum_x_value - minimum_x_value) / len(self.number_of_samples)) + 1
        x_ticks = range(minimum_x_value, maximum_x_value, space)
        return x_ticks

    def compute_xtickslabels(self):
        # number_of_samples and statistically_significant are both in sorted
        # n_samples order (guaranteed by _aggregate_train_size and _compute_stats)
        significance = ['*' if sig else 'NS' for sig in self.statistically_significant]
        x_ticks_labels = [
            str(nb_sample) + '\n' + sig
            for nb_sample, sig in zip(self.number_of_samples, significance)
        ]
        return x_ticks_labels

    def compute_yticks_and_ytickslabels(self):
        return [0.6, 0.7, 0.8, 0.9], [0.6, 0.7, 0.8, 0.9]

    def _configure_spine_visibility(self, ax):
        ax.spines['right'].set_visible(self.config.right_spine_visibility.value)
        ax.spines['left'].set_visible(self.config.left_spine_visibility.value)
        ax.spines['top'].set_visible(self.config.top_spine_visibility.value)
        ax.spines['bottom'].set_visible(self.config.bottom_spine_visibility.value)
        return ax

    def _configure_spine_linewidth(self, ax):
        ax.spines['left'].set_linewidth(self.config.spine_thickness.value)
        ax.spines['bottom'].set_linewidth(self.config.spine_thickness.value)
        return ax

    def format_spines(self, ax, y_limit, ylabel):
        ax = self._configure_spine_visibility(ax)
        ax = self._configure_spine_linewidth(ax)

        if y_limit is not None:
            ax.set_ylim(*y_limit)
        else:
            ax.margins(y=0.05)

        ax.tick_params(axis='x', labelsize=self.config.tick_size.value)
        ax.tick_params(axis='y', labelsize=self.config.tick_size.value)

        ax.set_xlabel('Number of samples', fontsize=self.config.label_size.value)
        ax.set_ylabel(ylabel, fontsize=self.config.label_size.value)

        return ax

    def plot(self, ax, title, ylabel='AUROC', y_limit=(0.4, 1.0), dense_label=None):
        self.process_results()

        x_ticks = self.compute_xticks()
        y_ticks, y_tick_lbls = self.compute_yticks_and_ytickslabels()
        x_ticks_labels = self.compute_xtickslabels()

        ax.set_title(title, loc="left", fontdict={"fontsize": 14, "fontweight": "bold"})

        ax.plot(
            x_ticks, self.pnet_auroc,
            c=self.config.pnet_auroc_color.value,
            marker=self.config.marker.value,
            markersize=self.config.marker_size.value,
            label=self.config.pnet_label.value,
        )
        ax.plot(
            x_ticks, self.dense_auroc,
            c=self.config.dense_auroc_color.value,
            marker=self.config.marker.value,
            markersize=self.config.marker_size.value,
            label=self.config.dense_label.value if dense_label is None else dense_label,
        )

        ax.legend(
            loc='upper left',
            frameon=self.config.legend_frame.value,
            fontsize=self.config.legend_fontsize.value,
        )

        ax.fill_between(
            x_ticks, self.pnet_lower_bound, self.pnet_upper_bound,
            color=self.config.pnet_auroc_color.value, edgecolor=None, alpha=0.15,
        )
        ax.fill_between(
            x_ticks, self.dense_lower_bound, self.dense_upper_bound,
            color=self.config.dense_auroc_color.value, edgecolor=None, alpha=0.15,
        )

        ax = self.format_spines(ax, y_limit=y_limit, ylabel=ylabel)

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks_labels)

        if y_limit is not None:
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_tick_lbls)


def _load_train_size_results(run_dir, prefix, metric="auc"):
    metric_col = f"response_{metric}"
    results = []
    for exp_dir in [x for x in os.listdir(run_dir) if re.fullmatch(re.escape(prefix) + r'_\d+', x)]:
        data = pd.read_csv(
            f"{run_dir}/{exp_dir}/test_0/cv_0/fold_summaries.csv", index_col=0
        )
        n_samples = (
                pd.read_csv(f"{run_dir}/{exp_dir}/test_0/cv_0/fold_0/train_results.csv").shape[0]
                + pd.read_csv(f"{run_dir}/{exp_dir}/test_0/cv_0/fold_0/val_results.csv").shape[0]
        )
        data = data.loc[data["split"] == "val", [metric_col, "fold"]]
        data = data.rename(columns={metric_col: "response_metric"})
        data["n_samples"] = n_samples
        results.append(data)
    return pd.concat(results)


def _load_train_size_results_nested_CV(run_dir, prefix, metric="auc", selection_metric="auc"):
    """Load per-outer-fold *test* set metrics for a nested-CV train-size sweep.

    Mirrors the format returned by :func:`_load_train_size_results` (one row per
    fold with columns ``response_metric``, ``fold``, ``n_samples``) but the value
    is the held-out test metric of each outer fold rather than an inner-fold
    validation metric. ``n_samples`` is the total number of samples in the run
    (outer train + outer test), which is constant across the outer folds.
    """
    metric_col = f"response_{metric}"
    results = []
    for exp_dir in [x for x in os.listdir(run_dir) if re.fullmatch(re.escape(prefix) + r'_\d+', x)]:
        exp_path = f"{run_dir}/{exp_dir}"
        for test_dir in [d for d in os.listdir(exp_path) if re.fullmatch(r'test_\d+', d)]:
            best_dir = f"{exp_path}/{test_dir}/best_{selection_metric}"
            summary = pd.read_csv(f"{best_dir}/summary_results.csv", index_col=0)
            n_samples = (
                    pd.read_csv(f"{best_dir}/train_results.csv").shape[0]
                    + pd.read_csv(f"{best_dir}/test_results.csv").shape[0]
            )
            results.append(pd.DataFrame({
                "response_metric": [summary.loc["test", metric_col]],
                "fold": [int(test_dir.split("_")[1])],
                "n_samples": [n_samples],
            }))
    return pd.concat(results)


def _aggregate_train_size(df):
    return (
        df.groupby("n_samples")["response_metric"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("n_samples")
        .reset_index(drop=True)
    )


def _compute_stats(pnet_results, other_results, alpha=0.05):
    """Unpaired Student t-test per training-set size, one-sided (P-NET > other).

    Retained *only* for :func:`plot_train_size_comparisons`, whose scores are
    inner-fold validation metrics from a single outer split. This test is
    statistically wrong for cross-validation scores — the folds are paired and
    their training sets overlap, so it is both unpaired and uncorrected — but the
    published figure was produced with it and is kept reproducible.
    """
    # sorted() ensures order matches _aggregate_train_size output
    shared_sizes = sorted(pnet_results["n_samples"].unique())
    return [
        ttest_ind(
            pnet_results.loc[pnet_results["n_samples"] == n, "response_metric"].to_numpy(),
            other_results.loc[other_results["n_samples"] == n, "response_metric"].to_numpy(),
            alternative="greater",
        ).pvalue < alpha
        for n in shared_sizes
    ]


def _compute_pvalues_corrected(pnet_results, other_results):
    """Raw Nadeau & Bengio corrected resampled paired t-test p-values, one per
    training-set size.

    Both models are evaluated on the same outer k-fold split of the same sample
    pool (same ``samples_to_include`` and ``tt_split_seed``), so scores pair by
    outer fold index. Each size therefore contributes one score per outer fold
    and the default ``rho = 1/(k-1)`` in
    :func:`~architecture.significance_utils.corrected_paired_stats` is the
    correct ``n_test/n_train`` ratio — the same test the per-task
    ``significance_testing.py`` scripts run.

    Returns the *raw* p-values; multiplicity correction across sizes and
    comparisons is applied by :func:`_significance_corrected_fdr`.
    """
    # sorted() ensures order matches _aggregate_train_size output
    shared_sizes = sorted(pnet_results["n_samples"].unique())
    pvalues = []
    for n in shared_sizes:
        # Sorting by fold aligns the two models on the outer fold they share;
        # without it the pairing would follow directory listing order.
        pnet_fold = pnet_results.loc[pnet_results["n_samples"] == n].sort_values("fold")
        other_fold = other_results.loc[other_results["n_samples"] == n].sort_values("fold")
        if list(pnet_fold["fold"]) != list(other_fold["fold"]):
            raise ValueError(
                f"Fold mismatch at n_samples={n}: the corrected paired test needs "
                f"both models scored on the same outer folds, got "
                f"{list(pnet_fold['fold'])} and {list(other_fold['fold'])}."
            )
        stats = corrected_paired_stats(
            pnet_fold["response_metric"].to_numpy(),
            other_fold["response_metric"].to_numpy(),
        )
        pvalues.append(stats["p_raw"])
    return pvalues


def _significance_legacy(pnet_results, other_results, alpha=0.05):
    """Per-comparison significance flags with no multiplicity correction.

    Retained for :func:`plot_train_size_comparisons` exactly as published: each
    training-set size is thresholded at ``alpha`` on its own.
    """
    return _compute_stats(pnet_results, other_results, alpha=alpha)


def _significance_corrected_fdr(pnet_results, other_results, alpha=0.05):
    """Corrected paired p-values with Benjamini-Hochberg FDR across the
    training-set sizes of this comparison.

    Each comparison is its own family — the correction spans the sizes within a
    comparison, not the comparisons themselves.
    """
    return _fdr_across_sizes(_compute_pvalues_corrected(pnet_results, other_results), alpha)


def _fdr_across_sizes(pvalues, alpha):
    """Benjamini-Hochberg across the training-set sizes of a single comparison."""
    rejected, _, _, _ = multipletests(pvalues, alpha=alpha, method="fdr_bh")
    return list(rejected)


def _build_comparison_results(pnet_df, other_df, stats):
    assert list(pnet_df["n_samples"]) == list(other_df["n_samples"]), (
        "Sample size ordering mismatch between pnet and comparison model — "
        "both dataframes must be sorted by n_samples before calling this function."
    )
    return {
        "number_of_samples": pnet_df["n_samples"].to_numpy(),
        "pnet_auroc": pnet_df["mean"].to_numpy(),
        "pnet_lower_bound": (pnet_df["mean"] - pnet_df["std"]).to_numpy(),
        "pnet_upper_bound": (pnet_df["mean"] + pnet_df["std"]).to_numpy(),
        "dense_auroc": other_df["mean"].to_numpy(),
        "dense_lower_bound": (other_df["mean"] - other_df["std"]).to_numpy(),
        "dense_upper_bound": (other_df["mean"] + other_df["std"]).to_numpy(),
        "statistically_significant": np.array(stats),
    }


# Each comparison is (run-directory prefix, legend label, filename slug). The
# prefix has the sweep's suffix (e.g. "_nested_CV") appended before loading.
COMPARISON_DENSE = ("dense_single_layer_train_size_variation", "Dense Single Layer", "dense")
COMPARISON_PNETFC = ("pnetfc_train_size_variation", "P-NET-FC", "pnetfc")


def _render_train_size_comparisons(run_dir, figures_dir, loader, stats_fn, prefix_suffix,
                                   fname_prefix, metrics, comparisons):
    for metric in metrics:
        pnet_results = loader(run_dir, f"pnet_train_size_variation{prefix_suffix}", metric)

        y_limit = METRICS_Y_LIMITS[metric]
        # METRIC_DISPLAY overrides the shown name (auc -> AUROC); otherwise
        # acronym metrics stay upper-case and the rest read as normal words.
        ylabel = METRIC_DISPLAY.get(
            metric,
            metric.upper() if metric in {"auprc", "f1"} else metric.capitalize(),
        )
        # filename slug follows the display name so plots are named e.g. *_auroc.pdf
        slug = METRIC_DISPLAY.get(metric, metric).lower()

        for prefix, label, model_slug in comparisons:
            other_results = loader(run_dir, f"{prefix}{prefix_suffix}", metric)

            # compute stats before aggregation (need per-fold values for the
            # t-test); sorted() inside the stats functions keeps the per-size
            # order matching _aggregate_train_size.
            stats = stats_fn(pnet_results, other_results)

            comparison_results = _build_comparison_results(
                _aggregate_train_size(pnet_results),
                _aggregate_train_size(other_results),
                stats,
            )

            fig, ax = plt.subplots(figsize=(10, 7))
            ComparativeAnalysis(comparison_results).plot(
                ax, "", ylabel=ylabel, y_limit=y_limit, dense_label=label
            )
            fig.tight_layout()
            fig.savefig(
                os.path.join(figures_dir, f"{fname_prefix}_pnet_vs_{model_slug}_{slug}.pdf")
            )
            plt.close(fig)


def plot_train_size_comparisons(run_dir, figures_dir, metrics=METRICS):
    # Inner-fold validation metric (mean +- SD across inner CV folds).
    # Significance stars come from the original unpaired, uncorrected t-test —
    # kept as published for reproducibility. See _compute_stats.
    # Only the dense comparison is available: the single-split P-NET-FC
    # train-size sweep was not run (it exists for nested CV only).
    _render_train_size_comparisons(
        run_dir, figures_dir, _load_train_size_results, _significance_legacy,
        prefix_suffix="", fname_prefix="train_size", metrics=metrics,
        comparisons=[COMPARISON_DENSE],
    )


def plot_train_size_comparisons_nested_CV(run_dir, figures_dir, metrics=METRICS):
    # Held-out test metric (mean +- SD across the outer CV folds).
    # Significance stars come from the Nadeau & Bengio corrected resampled
    # paired t-test, with BH-FDR across the training-set sizes within each
    # comparison (the two comparisons are separate families — see
    # _significance_corrected_fdr).
    _render_train_size_comparisons(
        run_dir, figures_dir, _load_train_size_results_nested_CV, _significance_corrected_fdr,
        prefix_suffix="_nested_CV", fname_prefix="train_size_nested_CV", metrics=metrics,
        comparisons=[COMPARISON_DENSE, COMPARISON_PNETFC],
    )
