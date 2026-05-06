import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from architecture.pnet_model import PNetArchitectureGenerator, get_layer_maps
from prostate_cancer_prediction.figure_pnet_vs_dense import ComparativeAnalysis

from prostate_cancer_prediction.pnet_auprc import PlotAUPRC
from prostate_cancer_prediction.sankey import SankeyDiagram


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


def plot_single_split_auprc(run_dir, wd, ax):
    results = {}
    tabular = []
    for model in [x for x in os.listdir(run_dir) if x.find("elmarakeby") > -1 or x.find("specific_train_split") > -1]:
        if model.find("pnet") == -1:
            results[model] = pd.read_csv(f"{run_dir}/{model}/best_f1/test_results.csv", index_col=0)
            summary = pd.read_csv(f"{run_dir}/{model}/best_f1/summary_results.csv")
        else:
            results["pnet"] = pd.read_csv(f"{run_dir}/{model}/test_results.csv", index_col=0)
            summary = pd.read_csv(f"{run_dir}/{model}/summary_results.csv")
        summary["model"] = model
        summary.columns = ["split"] + summary.columns[1:].to_list()
        tabular.append(summary)

    PlotAUPRC(results).plot(ax, "A")
    pd.concat(tabular).to_csv(f"{wd}/specific_split_results.csv")


def plot_train_size_comparisons(run_dir, ax_dense, ax_fc):
    pnet_results  = _load_train_size_results(run_dir, "pnet_train_size_variation")
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


def plot_sankey(wd, run_dir, selected_genes, n_hidden_layers):
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

    SankeyDiagram(layers, weights).plot([10, 10, 10, 10, 10, 6], wd)


def plot(wd, run_dir, selected_genes, n_hidden_layers):
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(7, 14))
    plot_single_split_auprc(run_dir, wd, ax[0])
    plot_train_size_comparisons(run_dir, ax[1], ax[2])
    plt.tight_layout()
    plt.savefig(f"{wd}/figure_1.jpg")
    plt.close()
    plot_sankey(wd, run_dir, selected_genes, n_hidden_layers)