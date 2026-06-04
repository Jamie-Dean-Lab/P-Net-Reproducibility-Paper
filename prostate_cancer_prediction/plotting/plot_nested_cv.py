import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import ticker


def plot_nested_CV(run_dir, figures_dir, selection_metric="auc"):
    model_names = [
        "pnet_nested_CV",
        "pnet_GO_nested_CV",
        "decision_tree_nested_CV",
        "adaboost_nested_CV",
        "linear_svm_nested_CV",
        "random_forest_nested_CV",
        "rbf_svm_nested_CV",
        "sgd_logistic_regression_nested_CV"
    ]

    models_display = {
        "pnet": "P-NET",
        "pnet_GO": "P-NET-GO",
        "decision_tree": "Decision Tree",
        "adaboost": "Ada. Boosting",
        "linear_svm": "Linear SVM",
        "random_forest": "Random Forest",
        "rbf_svm": "RBF SVM",
        "sgd_logistic_regression": "Logistic Regression"
    }

    metric_display = {
        'auc': 'Area Under Curve (AUC)',
        'auprc': 'AUPRC',
        'f1': 'F1',
        'accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall'
    }

    paper_model_order = ['Decision Tree', 'Logistic Regression', 'Random Forest', 'Ada. Boosting', 'Linear SVM', 'RBF SVM', 'P-NET', 'P-NET-GO']
    current_palette = sns.color_palette(None, len(paper_model_order))
    my_pal = {m: current_palette[i] for i, m in enumerate(paper_model_order)}

    fontsize = 16
    fontproperties = {'family': 'Arial', 'weight': 'normal', 'size': 18}
    metric_cols = ["auc", "auprc", "f1", "accuracy", "precision", "recall"]

    all_data = []
    for model_name in model_names:
        short_name = model_name.replace("_nested_CV", "")
        model_dir = f"{run_dir}/{model_name}"

        # collect one row per outer fold from the best_{selection_metric} test results
        fold_scores = []
        test_dirs = sorted([d for d in os.listdir(model_dir) if d.startswith("test_")])
        for test_dir in test_dirs:
            path = f"{model_dir}/{test_dir}/best_{selection_metric}/summary_results.csv"
            if not os.path.exists(path):
                print(f"Warning: {path} not found, skipping")
                continue
            df = pd.read_csv(path, index_col=0)
            df.columns = [c.replace("response_", "") for c in df.columns]  # strip prefix
            test_row = df[df.index == "test"][metric_cols]
            fold_scores.append(test_row)

        if not fold_scores:
            continue

        model_df = pd.concat(fold_scores).reset_index(drop=True)
        model_df.columns = pd.MultiIndex.from_tuples(
            [(short_name, col) for col in metric_cols]
        )
        all_data.append(model_df)

    combined = pd.concat(all_data, axis=1)
    combined.columns = combined.columns.swaplevel(0, 1)

    sns.set_style("whitegrid")

    def draw_metric(ax, metric):
        dd = combined[metric].copy()
        dd.columns = [models_display[c] for c in dd.columns]

        # With only 5 outer folds per model, show every fold as a point and
        # summarise with mean +/- SD rather than a boxplot (whose quartiles
        # collapse on ties for such small n).
        means = dd.mean()
        stds = dd.std()
        order = list(means.sort_values().index)
        avg = means['P-NET']

        dd_long = dd.melt()

        # individual outer-fold scores
        sns.stripplot(
            ax=ax,
            x="variable",
            y="value",
            hue="variable",
            data=dd_long,
            order=order,
            palette=my_pal,
            legend=False,
            size=7,
            alpha=0.8,
            jitter=0.15,
            edgecolor="black",
            linewidth=0.5,
        )

        # mean +/- SD overlay
        x_pos = list(range(len(order)))
        ax.errorbar(
            x_pos,
            means[order].values,
            yerr=stds[order].values,
            fmt="_",
            markersize=20,
            markeredgewidth=2,
            color="black",
            capsize=4,
            elinewidth=1.2,
            zorder=10,
        )

        ax.axhline(avg, ls='--', linewidth=1)
        ax.set_ylim([0.4, 1.05])
        ax.set_ylabel(metric_display[metric], fontproperties)
        ax.set_xlabel('')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, horizontalalignment='right', fontsize=fontsize)
        ax.get_xaxis().set_minor_locator(ticker.AutoMinorLocator())
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        # visible tick marks under each model name to associate label with box
        ax.tick_params(axis='x', which='major', length=6, width=1.2, bottom=True)
        ax.minorticks_off()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    def draw_metric_boxplot(ax, metric):
        # Original boxplot rendering, kept for reference / reverting. To use,
        # call this instead of draw_metric in the figure loops below.
        dd = combined[metric].copy()
        dd.columns = [models_display[c] for c in dd.columns]

        avg = dd['P-NET'].median()
        order = list(dd.median().sort_values().index)

        dd = dd.melt()

        sns.boxplot(
            ax=ax,
            x="variable",
            y="value",
            hue="variable",
            data=dd,
            whis=1.5,
            order=order,
            palette=my_pal,
            legend=False,
            linewidth=1,
            showfliers=False
        )

        ax.axhline(avg, ls='--', linewidth=1)
        ax.set_ylim([0.4, 1.05])
        ax.set_ylabel(metric_display[metric], fontproperties)
        ax.set_xlabel('')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, horizontalalignment='right', fontsize=fontsize)
        ax.get_xaxis().set_minor_locator(ticker.AutoMinorLocator())
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        # visible tick marks under each model name to associate label with box
        ax.tick_params(axis='x', which='major', length=6, width=1.2, bottom=True)
        ax.minorticks_off()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    # Individual figure per metric
    for metric in metric_cols:
        fig, ax = plt.subplots(figsize=(8, 5))
        draw_metric(ax, metric)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f"nested_CV_{metric}.png"), dpi=300)
        plt.close()

    # Combined figure with every metric as a labelled panel (a, b, c, ...)
    n_cols = 2
    n_rows = -(-len(metric_cols) // n_cols)  # ceil division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 5 * n_rows))
    axes = axes.flatten()
    for i, metric in enumerate(metric_cols):
        ax = axes[i]
        draw_metric(ax, metric)
        ax.text(-0.08, 1.02, chr(ord('a') + i), transform=ax.transAxes,
                fontsize=22, fontweight='bold', va='bottom', ha='right')
    for j in range(len(metric_cols), len(axes)):
        axes[j].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "nested_CV_all.png"), dpi=300)
    plt.close(fig)
