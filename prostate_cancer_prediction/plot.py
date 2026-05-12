import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import ticker
from scipy.stats import ttest_ind
from architecture.pnet_model import PNetArchitectureGenerator, get_layer_maps
from prostate_cancer_prediction.figure_pnet_vs_dense import ComparativeAnalysis

from prostate_cancer_prediction.pnet_auprc import PlotAUPRC
from prostate_cancer_prediction.pnet_roc import PlotROC
from prostate_cancer_prediction.sankey import SankeyDiagram


def plot_stratified_5_fold_CV(run_dir, figures_dir):
    model_names = [
        "pnet_stratified_5_fold_CV",
        "decision_tree_stratified_5_fold_CV",
        "adaboost_stratified_5_fold_CV",
        "linear_svm_stratified_5_fold_CV",
        "random_forest_stratified_5_fold_CV",
        "rbf_svm_stratified_5_fold_CV",
        "sgd_logistic_regression_stratified_5_fold_CV"
    ]

    models_display = {
        "pnet": "P-NET",
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

    # Match the original paper's colour order exactly
    paper_model_order = ['Decision Tree', 'Logistic Regression', 'Random Forest', 'Ada. Boosting', 'Linear SVM', 'RBF SVM', 'P-NET']
    current_palette = sns.color_palette(None, len(paper_model_order))
    my_pal = {m: current_palette[i] for i, m in enumerate(paper_model_order)}

    fontsize = 8
    fontproperties = {'family': 'Arial', 'weight': 'normal', 'size': 9}

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
            data=dd,
            whis=1.5,
            order=order,
            palette=my_pal,
            linewidth=1,
            flierprops=flierprops
        )

        ax.axhline(avg, ls='--', linewidth=1)
        ax.set_ylim([0.4, 1.05])
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

        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f"stratified_5_fold_CV_{metric}.png"), dpi=300)
        plt.close()

def plot_nested_CV(run_dir, figures_dir, selection_metric="f1"):
    model_names = [
        "pnet_nested_CV",
        "decision_tree_nested_CV",
        "adaboost_nested_CV",
        "linear_svm_nested_CV",
        "random_forest_nested_CV",
        "rbf_svm_nested_CV",
        "sgd_logistic_regression_nested_CV"
    ]

    models_display = {
        "pnet": "P-NET",
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

    paper_model_order = ['Decision Tree', 'Logistic Regression', 'Random Forest', 'Ada. Boosting', 'Linear SVM', 'RBF SVM', 'P-NET']
    current_palette = sns.color_palette(None, len(paper_model_order))
    my_pal = {m: current_palette[i] for i, m in enumerate(paper_model_order)}

    fontsize = 8
    fontproperties = {'family': 'Arial', 'weight': 'normal', 'size': 9}
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
            data=dd,
            whis=1.5,
            order=order,
            palette=my_pal,
            linewidth=1,
            flierprops=flierprops
        )

        ax.axhline(avg, ls='--', linewidth=1)
        ax.set_ylim([0.4, 1.05])
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

        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f"nested_CV_{metric}.png"), dpi=300)
        plt.close()

def _load_train_size_results(run_dir, prefix):
    results = []
    for exp_dir in [x for x in os.listdir(run_dir) if x.find(prefix) > -1]:
        data = pd.read_csv(f"{run_dir}/{exp_dir}/test_0/cv_0/fold_summaries.csv", index_col=0)
        n_samples = (
            pd.read_csv(f"{run_dir}/{exp_dir}/test_0/cv_0/fold_0/train_results.csv").shape[0]
            + pd.read_csv(f"{run_dir}/{exp_dir}/test_0/cv_0/fold_0/val_results.csv").shape[0]
        )
        data = data.loc[data["split"] == "val", ["response_auc", "fold"]]
        data["n_samples"] = n_samples
        results.append(data)
    return pd.concat(results)


def _aggregate_train_size(df):
    return df.groupby("n_samples")["response_auc"].agg(["mean", "std"]).reset_index()


def _build_comparison_results(pnet_df, other_df, stats):
    return {
        "number_of_samples":    pnet_df["n_samples"],
        "pnet_auc":             pnet_df["mean"],
        "pnet_lower_bound":     pnet_df["mean"] - pnet_df["std"],
        "pnet_upper_bound":     pnet_df["mean"] + pnet_df["std"],
        "dense_auc":            other_df["mean"],
        "dense_lower_bound":    other_df["mean"] - other_df["std"],
        "dense_upper_bound":    other_df["mean"] + other_df["std"],
        "statistically_significant": np.array(stats),
    }

#TODO list of directories to plot rather than regex
def plot_single_split_curves(run_dir, wd, figures_dir, concat_val=False):
    results = {}
    tabular = []
    for model in [x for x in os.listdir(run_dir) if x.find("elmarakeby") > -1 or x.find("pnet_GO_single_split") > -1]:
        if model.find("pnet") == -1:
            base = f"{run_dir}/{model}/best_f1"
            test_df = pd.read_csv(f"{base}/test_results.csv", index_col=0)
            if concat_val:
                '''
                Believe this is incorrect since the validation set should not be used for both hyperparameter selection
                and model evaluation (data leakage). However, including it here to be able to reproduce results from original 
                study.
                '''
                val_df = pd.read_csv(f"{base}/val_results.csv", index_col=0)
                results[model] = pd.concat([test_df, val_df])
            else:
                results[model] = test_df
            summary = pd.read_csv(f"{base}/summary_results.csv")
        else:
            test_df = pd.read_csv(f"{run_dir}/{model}/test_results.csv", index_col=0)
            if concat_val:
                val_df = pd.read_csv(f"{run_dir}/{model}/val_results.csv", index_col=0)
                results["pnet"] = pd.concat([test_df, val_df])
            else:
                results["pnet"] = test_df
            summary = pd.read_csv(f"{run_dir}/{model}/summary_results.csv")
        summary["model"] = model
        summary.columns = ["split"] + summary.columns[1:].to_list()
        tabular.append(summary)

    for curve, plotter, fname in [("A", PlotAUPRC, "single_split_auprc.png"),
                                   ("A", PlotROC,   "single_split_roc.png")]:
        fig, ax = plt.subplots()
        plotter(results).plot(ax, curve)
        fig.savefig(os.path.join(figures_dir, fname), dpi=300)
        plt.close(fig)

    pd.concat(tabular).to_csv(f"{wd}/specific_split_results.csv")


def plot_train_size_comparisons(run_dir, ax_dense, ax_fc, figures_dir):
    pnet_results   = _load_train_size_results(run_dir, "pnet_train_size_variation")
    pnetfc_results = _load_train_size_results(run_dir, "pnetfc_train_size_variation")
    dense_results  = _load_train_size_results(run_dir, "dense_train_size_variation")

    pnet_dense_stats = [
        ttest_ind(
            pnet_results.loc[pnet_results["n_samples"] == n, "response_auc"].to_numpy(),
            dense_results.loc[dense_results["n_samples"] == n, "response_auc"].to_numpy(),
        ).pvalue < 0.05
        for n in pnet_results["n_samples"].unique()
    ]
    pnet_pnetfc_stats = [
        ttest_ind(
            pnet_results.loc[pnet_results["n_samples"] == n, "response_auc"].to_numpy(),
            pnetfc_results.loc[pnetfc_results["n_samples"] == n, "response_auc"].to_numpy(),
        ).pvalue < 0.05
        for n in pnet_results["n_samples"].unique()
    ]

    pnet_results   = _aggregate_train_size(pnet_results)
    pnetfc_results = _aggregate_train_size(pnetfc_results)
    dense_results  = _aggregate_train_size(dense_results)

    ComparativeAnalysis(_build_comparison_results(pnet_results, dense_results, pnet_dense_stats)).plot(
        ax_dense, "B", dense_label="Dense Single Layer"
    )
    ComparativeAnalysis(_build_comparison_results(pnet_results, pnetfc_results, pnet_pnetfc_stats)).plot(
        ax_fc, "C", dense_label="P-NET fully connected"
    )

    fig = ax_dense.get_figure()
    fig.savefig(os.path.join(figures_dir, "train_size_comparisons.png"), dpi=300)


def plot_sankey(wd, run_dir, selected_genes, n_hidden_layers, figures_dir):
    pnet_run_dir = f"{run_dir}/pnet_specific_train_split"

    deeplift = {
        fn.split("_")[-1].replace(".csv", ""): pd.read_csv(f"{pnet_run_dir}/{fn}", index_col=0)
        for fn in os.listdir(pnet_run_dir) if fn.find("feature_importance") > -1
    }

    reactome = PNetArchitectureGenerator()
    netx = reactome.get_reactome_networkx("architecture/Reactome/ReactomePathwaysRelation.txt")
    maps = reactome.get_layers(netx, n_hidden_layers, "architecture/Reactome/ReactomePathways.gmt", selected_genes)
    maps = get_layer_maps(deeplift["h0"].index, maps, False)

    pathwaynames = pd.read_csv("architecture/Reactome/ReactomePathways.txt", sep="\t", index_col=0, header=None)
    pathwaynames.columns = ["name", "species"]

    layers = {}
    weights = {}
    for i in range(len(maps)):
        nodes = pathwaynames.loc[maps[i].index, "name"].to_numpy() if i > 0 else maps[i].index.to_numpy()
        layers[f"layer_{i+1}"] = nodes
        weights[f"layer_{i+1}"] = maps[i].to_numpy() * deeplift[f"h{i}"].to_numpy()

    SankeyDiagram(layers, weights).plot([10, 10, 10, 10, 10, 6], figures_dir)


def plot(wd, run_dir, selected_genes, n_hidden_layers):
    figures_dir = os.path.join(wd, "figures")
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    #plot_nested_CV(run_dir, figures_dir)

    #fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(7, 14))
    plot_single_split_curves(run_dir, wd, figures_dir)
    # plot_train_size_comparisons(run_dir, ax[1], ax[2], figures_dir)
    # plt.tight_layout()
    # plt.savefig(f"{wd}/figure_1.jpg")
    # plt.close()
    # plot_sankey(wd, run_dir, selected_genes, n_hidden_layers, figures_dir)
    #
    # plot_stratified_5_fold_CV(run_dir, figures_dir)