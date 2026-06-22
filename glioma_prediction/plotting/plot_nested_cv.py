import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import ticker


def plot_nested_CV(run_dir, figures_dir, selection_metric="auc"):
    model_names = [
        "pnet",
        "pnet_GO",
        "dense",
        "decision_tree",
        "adaboost",
        "svc",
        "random_forest",
        "rbf_svm",
        "xgb",
        "lgbm",
        "sgd_logistic_regression",
    ]

    models_display = {
        "pnet": "P-NET",
        "pnet_GO": "P-NET-GO",
        "dense": "P-NET-FC",
        "decision_tree": "Decision Tree",
        "adaboost": "Ada. Boosting",
        "svc": "Linear SVM",
        "random_forest": "Random Forest",
        "rbf_svm": "RBF SVM",
        "xgb": "XGBoost",
        "lgbm": "LightGBM",
        "sgd_logistic_regression": "Logistic Regression",
    }

    metric_display = {
        "auc": "AUROC",
        "auprc": "AUPRC",
        "f1": "F1",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
    }

    paper_model_order = [
        "Decision Tree", "Ada. Boosting", "Logistic Regression", "Linear SVM",
        "P-NET-FC", "LightGBM", "XGBoost", "Random Forest", "RBF SVM",
        "P-NET", "P-NET-GO",
    ]
    current_palette = sns.color_palette(None, len(paper_model_order))
    my_pal = {m: current_palette[i] for i, m in enumerate(paper_model_order)}

    fontsize = 16
    fontproperties = {"family": "Arial", "weight": "normal", "size": 18}
    metric_cols = ["auc", "auprc", "f1", "accuracy", "precision", "recall"]

    all_data = []
    for model_name in model_names:
        model_dir = os.path.join(run_dir, model_name)
        if not os.path.isdir(model_dir):
            print(f"Warning: {model_dir} not found, skipping")
            continue

        fold_scores = []
        test_dirs = sorted([d for d in os.listdir(model_dir) if d.startswith("test_")])
        for test_dir in test_dirs:
            path = os.path.join(model_dir, test_dir, f"best_{selection_metric}", "summary_results.csv")
            if not os.path.exists(path):
                print(f"Warning: {path} not found, skipping")
                continue
            df = pd.read_csv(path, index_col=0)
            df.columns = [c.replace("response_", "") for c in df.columns]
            test_row = df[df.index == "test"][metric_cols]
            fold_scores.append(test_row)

        if not fold_scores:
            continue

        model_df = pd.concat(fold_scores).reset_index(drop=True)
        model_df.columns = pd.MultiIndex.from_tuples(
            [(model_name, col) for col in metric_cols]
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
        ax.set_ylim(top=min(ax.get_ylim()[1], 1.02))
        ax.set_ylabel(metric_display[metric], fontproperties)
        ax.set_xlabel("")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, horizontalalignment="right", fontsize=fontsize)
        ax.get_xaxis().set_minor_locator(ticker.AutoMinorLocator())
        ax.tick_params(axis="both", which="major", labelsize=fontsize)
        ax.tick_params(axis="x", which="major", length=6, width=1.2, bottom=True)
        ax.minorticks_off()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

    os.makedirs(figures_dir, exist_ok=True)

    for metric in metric_cols:
        fig, ax = plt.subplots(figsize=(10, 5))
        draw_metric(ax, metric)
        plt.tight_layout()
        slug = "auroc" if metric == "auc" else metric
        plt.savefig(os.path.join(figures_dir, f"nested_CV_{slug}.png"), dpi=300)
        plt.close()

