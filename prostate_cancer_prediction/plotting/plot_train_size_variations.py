import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum
from scipy.stats import ttest_ind


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
    for exp_dir in [x for x in os.listdir(run_dir) if x.find(prefix) > -1]:
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


def _aggregate_train_size(df):
    return (
        df.groupby("n_samples")["response_metric"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("n_samples")
        .reset_index(drop=True)
    )


def _compute_stats(pnet_results, other_results):
    # sorted() ensures order matches _aggregate_train_size output
    shared_sizes = sorted(pnet_results["n_samples"].unique())
    return [
        ttest_ind(
            pnet_results.loc[pnet_results["n_samples"] == n, "response_metric"].to_numpy(),
            other_results.loc[other_results["n_samples"] == n, "response_metric"].to_numpy(),
        ).pvalue < 0.05
        for n in shared_sizes
    ]


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


def plot_train_size_comparisons(run_dir, figures_dir, metrics=METRICS):
    for metric in metrics:
        pnet_results = _load_train_size_results(run_dir, "pnet_train_size_variation", metric)
        pnetfc_results = _load_train_size_results(run_dir, "pnetfc_train_size_variation", metric)
        dense_results = _load_train_size_results(run_dir, "dense_single_layer_train_size_variation", metric)

        # compute stats before aggregation (need per-fold values for t-test),
        # sorted() in _compute_stats ensures order matches _aggregate_train_size
        pnet_dense_stats = _compute_stats(pnet_results, dense_results)
        pnet_pnetfc_stats = _compute_stats(pnet_results, pnetfc_results)

        # aggregate after stats — both will be sorted by n_samples
        pnet_results = _aggregate_train_size(pnet_results)
        pnetfc_results = _aggregate_train_size(pnetfc_results)
        dense_results = _aggregate_train_size(dense_results)

        y_limit = METRICS_Y_LIMITS[metric]
        # METRIC_DISPLAY overrides the shown name (auc -> AUROC); otherwise
        # acronym metrics stay upper-case and the rest read as normal words.
        ylabel = METRIC_DISPLAY.get(
            metric,
            metric.upper() if metric in {"auprc", "f1"} else metric.capitalize(),
        )
        # filename slug follows the display name so plots are named e.g. *_auroc.png
        slug = METRIC_DISPLAY.get(metric, metric).lower()

        plots = [
            (
                "",
                _build_comparison_results(pnet_results, dense_results, pnet_dense_stats),
                "Dense Single Layer",
                f"train_size_pnet_vs_dense_{slug}.png",
            ),
            (
                "",
                _build_comparison_results(pnet_results, pnetfc_results, pnet_pnetfc_stats),
                "P-NET fully connected",
                f"train_size_pnet_vs_pnetfc_{slug}.png",
            ),
        ]

        for title, comparison_results, dense_label, fname in plots:
            fig, ax = plt.subplots(figsize=(10, 7))
            ComparativeAnalysis(comparison_results).plot(
                ax, title, ylabel=ylabel, y_limit=y_limit, dense_label=dense_label
            )
            fig.tight_layout()
            fig.savefig(os.path.join(figures_dir, fname), dpi=300)
            plt.close(fig)
