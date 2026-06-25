import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import ticker


def plot_external_validation(run_dir, figures_dir, dataset_tag="combined"):
    model_names = [
        "pnet_nested_CV",
        "pnet_GO_nested_CV",
        "pnetfc_nested_CV",
        "dense_single_layer_nested_CV",
        "decision_tree_nested_CV",
        "adaboost_nested_CV",
        "linear_svm_nested_CV",
        "random_forest_nested_CV",
        "rbf_svm_nested_CV",
        "sgd_logistic_regression_nested_CV",
    ]

    models_display = {
        "pnet_nested_CV":                    "P-NET",
        "pnet_GO_nested_CV":                 "P-NET-GO",
        "pnetfc_nested_CV":                  "P-NET-FC",
        "dense_single_layer_nested_CV":      "Dense Single Layer",
        "decision_tree_nested_CV":           "Decision Tree",
        "adaboost_nested_CV":                "Ada. Boosting",
        "linear_svm_nested_CV":              "Linear SVM",
        "random_forest_nested_CV":           "Random Forest",
        "rbf_svm_nested_CV":                 "RBF SVM",
        "sgd_logistic_regression_nested_CV": "Logistic Regression",
    }

    metric_display = {
        "auc":       "AUROC",
        "auprc":     "AUPRC",
        "f1":        "F1",
        "accuracy":  "Accuracy",
        "precision": "Precision",
        "recall":    "Recall",
    }

    paper_model_order = [
        "Decision Tree", "Ada. Boosting", "Logistic Regression", "Linear SVM",
        "Dense Single Layer", "P-NET-FC", "Random Forest", "RBF SVM",
        "P-NET", "P-NET-GO",
    ]
    current_palette = sns.color_palette(None, len(paper_model_order))
    my_pal = {m: current_palette[i] for i, m in enumerate(paper_model_order)}

    fontsize = 18
    fontproperties = {"family": "Arial", "weight": "normal", "size": 20}
    metric_cols = ["auc", "auprc", "f1", "accuracy", "precision", "recall"]

    records = {}
    for model_name in model_names:
        path = os.path.join(run_dir, model_name, "external_validation", dataset_tag, "metrics.csv")
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping")
            continue
        df = pd.read_csv(path)
        df.columns = [c.replace("metastatic_", "") for c in df.columns]
        records[models_display[model_name]] = df.iloc[0][metric_cols].to_dict()

    if not records:
        print("No external validation results found.")
        return

    scores = pd.DataFrame(records).T
    order = [m for m in paper_model_order if m in scores.index]

    sns.set_style("white")

    def draw_metric(ax, metric):
        vals = scores.loc[order, metric]
        colors = [my_pal[m] for m in order]
        avg = vals["P-NET"]

        ax.bar(range(len(order)), vals.values, color=colors, edgecolor="black", linewidth=0.5)
        ax.axhline(avg, ls="--", linewidth=1)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=30, horizontalalignment="right", fontsize=fontsize)
        ax.set_xlim(-0.5, len(order) - 0.5)
        ax.set_ylim(0, min(vals.max() * 1.15, 1.02))
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
        fig, ax = plt.subplots(figsize=(10, 5))
        draw_metric(ax, metric)
        plt.tight_layout()
        slug = "auroc" if metric == "auc" else metric
        plt.savefig(os.path.join(figures_dir, f"external_validation_{slug}.png"), dpi=300)
        plt.close()
