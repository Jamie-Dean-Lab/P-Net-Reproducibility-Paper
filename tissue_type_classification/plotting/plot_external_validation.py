import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import ticker


def plot_external_validation(run_dir, figures_dir, dataset_tag="hpa"):
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

    records = {}
    for model_name in model_names:
        path = os.path.join(run_dir, model_name, "external_validation", dataset_tag, "metrics.csv")
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping")
            continue
        df = pd.read_csv(path)
        available = {c: df.iloc[0][c] for c in metric_cols if c in df.columns}
        records[models_display[model_name]] = available

    if not records:
        print("No external validation results found.")
        return

    scores = pd.DataFrame(records).T
    order = [m for m in paper_model_order if m in scores.index]

    sns.set_style("white")

    def draw_metric(ax, metric):
        if metric not in scores.columns:
            return
        vals = scores.loc[order, metric]
        colors = [my_pal[m] for m in order]
        avg = vals["P-NET"]

        ax.bar(range(len(order)), vals.values, color=colors, edgecolor="black", linewidth=0.5)
        ax.axhline(avg, ls="--", linewidth=1)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=30, horizontalalignment="right", fontsize=fontsize)
        ax.set_xlim(-0.5, len(order) - 0.5)
        ymin = min(vals.min() * 1.15, 0)
        if metric in bounded_metrics:
            ymax = min(max(vals.max() * 1.15, 0), 1.02)
        else:
            ymax = max(vals.max() * 1.15, 0)
        if ymax == 0:
            ymax = abs(ymin) * 0.05
        ax.set_ylim(ymin, ymax)
        ax.set_ylabel(metric_display[metric], fontproperties)
        ax.set_xlabel("")
        ax.get_xaxis().set_minor_locator(ticker.AutoMinorLocator())
        ax.tick_params(axis="both", which="major", labelsize=fontsize)
        ax.minorticks_off()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

    os.makedirs(figures_dir, exist_ok=True)

    for metric in metric_cols:
        if metric not in scores.columns:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        draw_metric(ax, metric)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f"external_validation_{metric}.png"), dpi=300)
        plt.close()
