import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import ticker


def plot_stratified_5_fold_CV(run_dir, figures_dir):
    model_names = [
        "pnet_stratified_5_fold_CV",
        "pnetfc_stratified_5_fold_CV",
        "pnet_GO_stratified_5_fold_CV",
        "dense_single_layer_stratified_5_fold_CV",
        "decision_tree_stratified_5_fold_CV",
        "adaboost_stratified_5_fold_CV",
        "linear_svm_stratified_5_fold_CV",
        "random_forest_stratified_5_fold_CV",
        "rbf_svm_stratified_5_fold_CV",
        "sgd_logistic_regression_stratified_5_fold_CV"
    ]

    models_display = {
        "pnet": "P-NET",
        "pnetfc": "P-NET-FC",
        "pnet_GO": "P-NET-GO",
        "dense_single_layer": "Dense Single Layer",
        "decision_tree": "Decision Tree",
        "adaboost": "Ada. Boosting",
        "linear_svm": "Linear SVM",
        "random_forest": "Random Forest",
        "rbf_svm": "RBF SVM",
        "sgd_logistic_regression": "Logistic Regression"
    }

    metric_display = {
        'auc': 'AUROC',
        'auprc': 'AUPRC',
        'f1': 'F1',
        'accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall'
    }

    # Match the original paper's colour order exactly
    paper_model_order = ['Decision Tree', 'Logistic Regression', 'Random Forest', 'Ada. Boosting', 'Linear SVM', 'RBF SVM', 'P-NET', 'P-NET-FC', 'P-NET-GO', 'Dense Single Layer']
    current_palette = sns.color_palette(None, len(paper_model_order))
    my_pal = {m: current_palette[i] for i, m in enumerate(paper_model_order)}

    # close to plot_train_size_comparisons (tick_size=18, label_size=20), nudged down slightly
    fontsize = 16
    fontproperties = {'family': 'Arial', 'weight': 'normal', 'size': 18}

    col_names = ["split", "auc", "auprc", "f1", "accuracy", "precision", "recall", "fold"]
    metric_cols = ["auc", "auprc", "f1", "accuracy", "precision", "recall"]

    all_data = []
    for model_name in model_names:
        path = f"{run_dir}/{model_name}/test_0/cv_0/fold_summaries.csv"
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping")
            continue
        data = pd.read_csv(path, index_col=0, header=0, names=col_names, skiprows=1)
        data[metric_cols] = data[metric_cols].astype(float)
        val_data = data[data["split"] == "val"][metric_cols].copy()
        short_name = model_name.replace("_stratified_5_fold_CV", "")
        val_data.columns = pd.MultiIndex.from_tuples(
            [(short_name, col) for col in metric_cols]
        )
        all_data.append(val_data)

    combined = pd.concat(all_data, axis=1)
    combined.columns = combined.columns.swaplevel(0, 1)

    flierprops = dict(marker='o', markersize=1, alpha=0.7)

    for metric in metric_cols:
        fig, ax = plt.subplots(figsize=(8, 5))
        dd = combined[metric].copy()
        dd.columns = [models_display[c] for c in dd.columns]

        avg = dd['P-NET'].median()
        order = list(dd.median().sort_values().index)

        sns.set_style("whitegrid")
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
            flierprops=flierprops
        )

        ax.axhline(avg, ls='--', linewidth=1)
        ax.autoscale(axis='y')
        ax.set_ylabel(metric_display[metric], fontproperties)
        ax.set_xlabel('')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, horizontalalignment='right', fontsize=fontsize)
        ax.get_xaxis().set_minor_locator(ticker.AutoMinorLocator())
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        ax.minorticks_off()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.yaxis.grid(False)  # remove horizontal grid lines

        # 'auc' is the data-column key; the file is named with the AUROC label
        slug = "auroc" if metric == "auc" else metric
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f"stratified_5_fold_CV_{slug}.png"), dpi=300)
        plt.close()
