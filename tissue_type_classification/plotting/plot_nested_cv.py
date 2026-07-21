import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import ticker


def plot_nested_CV(run_dir, figures_dir):
    model_names = [
        "pnet",
        "dense",
        "decision_tree",
        "adaboost",
        "sgd_logistic_regression",
        "svc",
        "rbf_svm",
        "lgbm",
        "xgb",
        "random_forest",
    ]

    models_display = {
        "pnet":                    "P-NET",
        "dense":                   "P-NET-FC",
        "decision_tree":           "Decision Tree",
        "adaboost":                "Ada. Boosting",
        "sgd_logistic_regression": "Logistic Reg.",
        "svc":                     "Linear SVM",
        "rbf_svm":                 "RBF SVM",
        "lgbm":                    "LightGBM",
        "xgb":                     "XGBoost",
        "random_forest":           "Random Forest",
    }

    metric_display = {
        "auc":      "AUROC",
        "auprc":    "AUPRC",
        "f1":       "Weighted F1",
        "accuracy": "Accuracy",
    }

    # All classification metrics live in [0, 1]; cap the y-axis just above 1.
    bounded_metrics = {"auc", "auprc", "f1", "accuracy"}

    paper_model_order = [
        "Decision Tree", "Ada. Boosting", "Logistic Reg.", "Linear SVM",
        "RBF SVM", "LightGBM", "XGBoost", "Random Forest",
        "P-NET-FC", "P-NET",
    ]
    current_palette = sns.color_palette(None, len(paper_model_order))
    my_pal = {m: current_palette[i] for i, m in enumerate(paper_model_order)}

    fontsize = 18
    fontproperties = {"family": "Arial", "weight": "normal", "size": 20}
    metric_cols = list(metric_display.keys())

    all_data = []
    for model_name in model_names:
        path = os.path.join(run_dir, model_name, "results.csv")
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping")
            continue

        df = pd.read_csv(path, index_col=0)
        test_rows = df[df["index"] == "test"][[c for c in metric_cols if c in df.columns]]
        model_df = test_rows.reset_index(drop=True)
        model_df.columns = pd.MultiIndex.from_tuples(
            [(model_name, col) for col in model_df.columns]
        )
        all_data.append(model_df)

    if not all_data:
        print("No results found — have the runs completed?")
        return

    combined = pd.concat(all_data, axis=1)
    combined.columns = combined.columns.swaplevel(0, 1)

    sns.set_style("white")

    def draw_metric(ax, metric):
        dd = combined[metric].copy()
        dd.columns = [models_display[c] for c in dd.columns]

        means = dd.mean()
        stds = dd.std()
        order = [m for m in paper_model_order if m in dd.columns]
        avg = means["P-NET"]

        dd_long = dd.melt()

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

        ax.axhline(avg, ls="--", linewidth=1)
        ax.autoscale(enable=True, axis="y")
        ax.margins(y=0.08)
        if metric in bounded_metrics:
            ax.set_ylim(top=min(ax.get_ylim()[1], 1.02))
        ax.set_ylabel(metric_display[metric], fontproperties)
        ax.set_xlabel("")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, horizontalalignment="right", fontsize=fontsize)
        ax.get_xaxis().set_minor_locator(ticker.AutoMinorLocator())
        ax.tick_params(axis="both", which="major", labelsize=fontsize)
        ax.minorticks_off()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

    os.makedirs(figures_dir, exist_ok=True)

    for metric in metric_cols:
        if metric not in combined.columns.get_level_values(0):
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        draw_metric(ax, metric)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f"nested_CV_{metric}.pdf"))
        plt.close()
