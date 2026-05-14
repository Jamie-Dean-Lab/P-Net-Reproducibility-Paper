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
import plotly.graph_objects as go
from plotly.offline import plot
import matplotlib.pyplot as plt


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


# def plot_sankey(wd, run_dir, selected_genes, n_hidden_layers, figures_dir):
#     pnet_run_dir = f"{run_dir}/pnet_specific_train_split"
#
#     deeplift = {
#         fn.split("_")[-1].replace(".csv", ""): pd.read_csv(f"{pnet_run_dir}/{fn}", index_col=0)
#         for fn in os.listdir(pnet_run_dir) if fn.find("feature_importance") > -1
#     }
#
#     reactome = PNetArchitectureGenerator()
#     netx = reactome.get_networkx("architecture/Reactome/ReactomePathwaysRelation.txt", "reactome")
#     maps = reactome.get_layers(netx, n_hidden_layers, "architecture/Reactome/ReactomePathways.gmt", selected_genes)
#     maps = get_layer_maps(deeplift["h0"].index, maps, False)
#
#     pathwaynames = pd.read_csv("architecture/Reactome/ReactomePathways.txt", sep="\t", index_col=0, header=None)
#     pathwaynames.columns = ["name", "species"]
#
#     layers = {}
#     weights = {}
#
#     # build input layer from deeplift['inputs']
#     inputs_df = deeplift["inputs"].copy()
#     inputs_df.index.name = "feature"
#     inputs_df = inputs_df.reset_index()
#     inputs_df["input_type"] = inputs_df["feature"].str.extract(r"^(mut_important|cnv_amp|cnv_del)")
#     inputs_df["gene"] = inputs_df["feature"].str.replace(r"^(mut_important|cnv_amp|cnv_del)_", "", regex=True)
#
#     input_types = ["mut_important", "cnv_amp", "cnv_del"]
#     input_labels = np.array(["Mutation", "Amplification", "Deletion"])
#
#     gene_order = deeplift["h0"].index.tolist()
#
#     pivot = inputs_df.pivot(index="input_type", columns="gene", values="feature_importance")
#     pivot = pivot.reindex(index=input_types, columns=gene_order).fillna(0)
#
#     layers["layer_0"] = input_labels
#     weights["layer_0"] = pivot.to_numpy()
#
#     # remaining layers from maps and deeplift
#     for i in range(len(maps)):
#         nodes = pathwaynames.loc[maps[i].index, "name"].to_numpy() if i > 0 else maps[i].index.to_numpy()
#         layers[f"layer_{i+1}"] = nodes
#         col = "feature_importance"
#
#         # align deeplift to map row order before multiplying
#         deeplift_aligned = deeplift[f"h{i}"][col].reindex(maps[i].index)
#
#         n_nan = deeplift_aligned.isna().sum()
#         print(f"Layer {i}: deeplift aligned to maps[{i}], n_nan after reindex = {n_nan}")
#         if n_nan > 0:
#             print(f"  !! maps[{i}].index sample:       {maps[i].index.tolist()[:3]}")
#             print(f"  !! deeplift['h{i}'].index sample: {deeplift[f'h{i}'].index.tolist()[:3]}")
#
#         weights[f"layer_{i+1}"] = maps[i].to_numpy() * deeplift_aligned.to_numpy()[:, np.newaxis]
#
#     SankeyDiagram(layers, weights).plot([3, 10, 10, 10, 10, 10, 6], figures_dir)


def plot_sankey(wd, run_dir, selected_genes, n_hidden_layers, figures_dir):
    """
        Generates a Sankey diagram visualising the P-Net model's feature importance flow
        from input genomic features through gene and pathway layers to the outcome node.

        The function reproduces the visualisation from Elmarakeby et al. (2021). It loads
        pre-computed DeepLIFT importance scores and trained model link weights, selects the
        top N most important nodes per layer (genes by coef_combined, pathways by coef),
        and builds a layered edge graph connecting inputs (mutation, amplification, deletion)
        -> genes -> pathway hierarchy (6 layers) -> outcome.

        Non-selected nodes are collapsed into 'residual' nodes per layer rather than dropped,
        preserving total flow. Edge weights for the pathway layers (1+) are adjusted via
        adjust_values() which reweights each edge as the minimum of source-normalised and
        target-normalised importance, ensuring edges only appear thick when both endpoints
        are important. The first layer (input -> gene) edges are NOT passed through
        adjust_values(), matching the original codebase which constructs these edges
        separately and concatenates them after adjustment.

        Node y-positions are computed from flow values (max of incoming/outgoing), with
        residual nodes forced to the bottom of each layer. Output is saved as PDF, PNG,
        and interactive HTML.
        """
    pnet_run_dir = f"{run_dir}/pnet_single_split"

    deeplift = {
        fn.split("_")[-1].replace(".csv", ""): pd.read_csv(f"{pnet_run_dir}/{fn}", index_col=0)
        for fn in os.listdir(pnet_run_dir) if fn.startswith("feature_importance_")
    }
    link_weights = {
        int(fn.split("_")[-1].replace(".csv", "")): pd.read_csv(f"{pnet_run_dir}/{fn}", index_col=0)
        for fn in os.listdir(pnet_run_dir) if fn.startswith("link_weights_")
    }

    print(f"\n=== plot_sankey ===")
    print(f"deeplift keys: {sorted(deeplift.keys())}")
    print(f"link_weights keys: {sorted(link_weights.keys())}")
    for k, v in deeplift.items():
        print(f"  deeplift['{k}'] shape={v.shape}, columns={v.columns.tolist()}")

    pathwaynames = pd.read_csv("architecture/Reactome/ReactomePathways.txt", sep="\t", index_col=0, header=None)
    pathwaynames.columns = ["name", "species"]
    id_to_name = pathwaynames["name"].to_dict()

    nlargest = [10, 10, 10, 10, 6, 6, 6]

    # -------------------------------------------------------------------
    # 1. select top N important nodes per layer
    #    genes: coef_combined, pathways: coef — matching original
    # -------------------------------------------------------------------
    print(f"\n--- Section 1: Node selection ---")
    gene_importance = deeplift["h0"]["coef_combined"].copy()
    gene_importance = gene_importance[gene_importance.index.isin(link_weights[1].index)]
    print(f"  gene_importance after Reactome filter: {len(gene_importance)} genes")

    high_nodes = {}
    high_nodes[0] = ["mut_important", "cnv_amp", "cnv_del"]
    high_nodes[1] = gene_importance.nlargest(nlargest[0]).index.tolist()
    print(f"  high_nodes[1] (genes): {high_nodes[1]}")

    for i in range(1, n_hidden_layers + 1):
        k = f"h{i}"
        if k in deeplift:
            high_nodes[i + 1] = deeplift[k]["coef"].nlargest(nlargest[i]).index.tolist()
            print(f"  high_nodes[{i+1}] (layer {k}): {high_nodes[i+1]}")

    # -------------------------------------------------------------------
    # 2. node importance lookup
    #    matching original: 100 * coef / layer_sum over ALL nodes, then log(1+x)
    # -------------------------------------------------------------------
    print(f"\n--- Section 2: Node importance ---")
    node_importance = {}

    gi_all = deeplift["h0"]["coef"].clip(lower=0)
    layer_sum = gi_all.sum()
    gi_norm_all = np.log(1. + 100. * gi_all / layer_sum) if layer_sum > 0 else gi_all
    node_importance.update(gi_norm_all.to_dict())
    print(f"  h0 (genes): layer_sum={layer_sum:.4f}, "
          f"importance range=[{gi_norm_all.min():.4f}, {gi_norm_all.max():.4f}]")

    for i in range(1, n_hidden_layers + 1):
        k = f"h{i}"
        if k in deeplift:
            pi_all = deeplift[k]["coef"].clip(lower=0)
            layer_sum = pi_all.sum()
            pi_norm_all = np.log(1. + 100. * pi_all / layer_sum) if layer_sum > 0 else pi_all
            node_importance.update(pi_norm_all.to_dict())
            print(f"  {k}: layer_sum={layer_sum:.4f}, "
                  f"importance range=[{pi_norm_all.min():.4f}, {pi_norm_all.max():.4f}]")

    # others nodes get small fixed importance matching original coef=1
    for i in range(1, n_hidden_layers + 2):
        node_importance[f"others{i}"] = 0.1

    # input nodes: fixed importance matching original coef=1
    gene_layer_sum = deeplift["h0"]["coef"].clip(lower=0).sum()
    input_importance_val = float(np.log(1. + 100. * 1.0 / gene_layer_sum)) if gene_layer_sum > 0 else 1.0
    for input_id in ["mut_important", "cnv_amp", "cnv_del"]:
        node_importance[input_id] = input_importance_val
    print(f"  input node importance: {input_importance_val:.4f}")
    print(f"  Total nodes in importance lookup: {len(node_importance)}")
    print(f"  Sample high-node importances:")
    for g in high_nodes[1][:3]:
        print(f"    {g}: {node_importance.get(g, 'MISSING'):.4f}")

    # -------------------------------------------------------------------
    # 3. assemble edges
    # -------------------------------------------------------------------
    print(f"\n--- Section 3: Edge assembly ---")
    all_edges = []

    # layer 0: inputs -> genes
    # matching original gradient_importance_0.csv: 2D per-gene, per-input-type scores
    # reconstruct using deeplift input scores masked by link_weights[0] connectivity
    print(f"  Layer 0: building per-gene input importance matrix")
    lw0 = link_weights[0].copy()
    print(f"  lw0 shape: {lw0.shape}, index[:3]: {lw0.index.tolist()[:3]}, cols[:3]: {lw0.columns.tolist()[:3]}")

    input_importance = deeplift["inputs"]["feature_importance"].copy()
    print(f"  input_importance shape: {input_importance.shape}, index[:3]: {input_importance.index.tolist()[:3]}")

    lw0_nonzero = (lw0 != 0).astype(float)
    input_imp_aligned = input_importance.reindex(lw0.index).fillna(0).abs()
    print(f"  input_imp_aligned nulls: {input_imp_aligned.isna().sum()}, "
          f"zeros: {(input_imp_aligned == 0).sum()}")

    input_gene_matrix = lw0_nonzero.multiply(input_imp_aligned.values, axis=0)
    print(f"  input_gene_matrix shape: {input_gene_matrix.shape}, "
          f"non-zero: {(input_gene_matrix != 0).sum().sum()}")

    input_types = pd.Series(lw0.index, index=lw0.index).str.extract(r"^(mut_important|cnv_amp|cnv_del)")[0]
    print(f"  input_types nulls: {input_types.isna().sum()}")
    input_gene_matrix.index = input_types.values
    input_gene_matrix = input_gene_matrix.groupby(input_gene_matrix.index).sum()
    print(f"  input_gene_matrix after groupby shape: {input_gene_matrix.shape}")
    print(f"  input_gene_matrix row totals:\n{input_gene_matrix.sum(axis=1).sort_values(ascending=False)}")

    input_gene_matrix.index.name = "source"
    first_edges = input_gene_matrix.reset_index().melt(
        id_vars="source", var_name="target", value_name="value"
    )
    first_edges = first_edges[first_edges["value"] != 0].copy()
    first_edges["layer"] = 0
    print(f"  first_edges before collapsing: {len(first_edges)} edges")

    # collapse non-top genes to others1
    first_edges["target"] = first_edges["target"].map(
        lambda x: x if x in high_nodes[1] else "others1"
    )
    first_edges = first_edges.groupby(["source", "target"])["value"].sum().reset_index()
    first_edges["layer"] = 0
    print(f"  first_edges after collapsing to others1: {len(first_edges)} edges")
    print(f"  others1 edges: {(first_edges['target'] == 'others1').sum()}")

    # normalize by target gene then scale by gene importance * 150
    gene_imp_raw = np.log(1. + deeplift["h0"]["coef_combined"].clip(lower=0).abs())
    gene_imp_raw["others1"] = 10.0
    first_edges["value"] = first_edges["value"] / first_edges.groupby("target")["value"].transform("sum")
    first_edges["gene_imp"] = first_edges["target"].map(gene_imp_raw.to_dict()).fillna(0)
    first_edges["value"] = first_edges["value"] * first_edges["gene_imp"] * 150.
    first_edges = first_edges[first_edges["value"] > 0][["source", "target", "value", "layer"]]

    input_totals = first_edges.groupby("source")["value"].sum().sort_values(ascending=False)
    print(f"  Input type totals (layer 0, no adjust_values applied):\n{input_totals.to_string()}")
    print(f"  others1 edges after scaling: {(first_edges['target'] == 'others1').sum()}")
    # layer 0 edges are NOT passed through adjust_values — matching original
    # which concatenates first_layer_df AFTER adjust_values is run on pathway edges
    all_edges.append(first_edges)

    # layers 1+: with others residual nodes matching filter_connections(add_unk=True)
    pathway_edges = []
    for i in range(n_hidden_layers):
        lw = link_weights[i + 1].abs()
        src_high = set(high_nodes[i + 1])
        tgt_high = set(high_nodes[i + 2])
        others_src = f"others{i + 1}"
        others_tgt = f"others{i + 2}"

        df_layer = lw.copy()
        df_layer.index.name = "source"
        edges = df_layer.reset_index().melt(id_vars="source", var_name="target", value_name="value")
        edges = edges[edges["value"] != 0].copy()

        ind1 = edges["source"].isin(src_high)
        ind2 = edges["target"].isin(tgt_high)
        edges = edges[ind1 | ind2].copy()

        edges["source"] = edges["source"].map(lambda x: x if x in src_high else others_src)
        edges["target"] = edges["target"].map(lambda x: x if x in tgt_high else others_tgt)
        edges = edges.groupby(["source", "target"])["value"].sum().reset_index()
        edges["layer"] = i + 1

        print(f"  Layer {i+1}: {len(edges)} edges, "
              f"sources={sorted(edges['source'].unique().tolist())[:5]}..., "
              f"targets={sorted(edges['target'].unique().tolist())[:5]}...")
        pathway_edges.append(edges)

    pathway_edges_df = pd.concat(pathway_edges, ignore_index=True)
    print(f"  Total pathway edges before adjust_values: {len(pathway_edges_df)}")

    # -------------------------------------------------------------------
    # 4. adjust_values on pathway edges only (layers 1+)
    #    matching original which runs adjust_values before concatenating first layer
    # -------------------------------------------------------------------
    print(f"\n--- Section 4: adjust_values (pathway edges only) ---")
    df_pw = pathway_edges_df.copy()
    df_pw["value_abs"] = df_pw["value"].abs()
    df_pw["child_sum_target"] = df_pw.groupby("target")["value_abs"].transform("sum")
    df_pw["child_sum_source"] = df_pw.groupby("source")["value_abs"].transform("sum")
    df_pw["value_normalized_by_target"] = 100. * df_pw["value_abs"] / df_pw["child_sum_target"]
    df_pw["value_normalized_by_source"] = 100. * df_pw["value_abs"] / df_pw["child_sum_source"]
    df_pw["target_importance"] = df_pw["target"].map(node_importance).fillna(0)
    df_pw["source_importance"] = df_pw["source"].map(node_importance).fillna(0)

    missing_src = df_pw[df_pw["source_importance"] == 0]["source"].unique()
    missing_tgt = df_pw[df_pw["target_importance"] == 0]["target"].unique()
    if len(missing_src) > 0:
        print(f"  WARNING: zero source_importance for: {missing_src.tolist()}")
    if len(missing_tgt) > 0:
        print(f"  WARNING: zero target_importance for: {missing_tgt.tolist()}")

    df_pw["A"] = df_pw["value_normalized_by_source"] * df_pw["source_importance"]
    df_pw["B"] = df_pw["value_normalized_by_target"] * df_pw["target_importance"]
    df_pw["value_final"] = df_pw[["A", "B"]].min(axis=1)

    df_pw["source_fan_out"] = df_pw.groupby("source")["value_final"].transform("sum")
    df_pw["source_fan_out_error"] = np.abs(df_pw["source_fan_out"] - 100. * df_pw["source_importance"])
    df_pw["target_fan_in"] = df_pw.groupby("target")["value_final"].transform("sum")
    df_pw["target_fan_in_error"] = np.abs(df_pw["target_fan_in"] - 100. * df_pw["target_importance"])

    df_pw["value_final_corrected"] = df_pw["value_final"]
    ind_src = df_pw["source"].str.contains("others", na=False)
    df_pw.loc[ind_src, "value_final_corrected"] = (
        df_pw.loc[ind_src, "value_final"] + df_pw.loc[ind_src, "target_fan_in_error"]
    )
    ind_tgt = df_pw["target"].str.contains("others", na=False)
    df_pw.loc[ind_tgt, "value_final_corrected"] = (
        df_pw.loc[ind_tgt, "value_final_corrected"] + df_pw.loc[ind_tgt, "source_fan_out_error"]
    )
    df_pw["value"] = df_pw["value_final_corrected"]
    df_pw = df_pw[df_pw["value"] > 0][["source", "target", "value", "layer"]].copy()
    print(f"  Pathway edges after adjust_values: {len(df_pw)}")
    print(f"  Value range: [{df_pw['value'].min():.4f}, {df_pw['value'].max():.4f}]")

    # concatenate first layer (unadjusted) + adjusted pathway edges — matching original
    df = pd.concat([first_edges, df_pw], ignore_index=True)
    print(f"  Total edges after concat: {len(df)}")

    input_totals_final = df[df["source"].isin(["mut_important", "cnv_amp", "cnv_del"])]
    input_totals_final = input_totals_final.groupby("source")["value"].sum().sort_values(ascending=False)
    print(f"  Input type totals (final):\n{input_totals_final.to_string()}")

    # -------------------------------------------------------------------
    # 5. build node list — only include others nodes that have edges
    # -------------------------------------------------------------------
    print(f"\n--- Section 5: Node list ---")
    others_with_edges = set(
        df[df["source"].str.startswith("others", na=False)]["source"].tolist() +
        df[df["target"].str.startswith("others", na=False)]["target"].tolist()
    )
    print(f"  others nodes with edges: {sorted(others_with_edges)}")

    layer_nodes = {}
    layer_nodes[0] = ["mut_important", "cnv_amp", "cnv_del"]
    layer_nodes[1] = high_nodes[1] + (["others1"] if "others1" in others_with_edges else [])
    for i in range(1, n_hidden_layers + 1):
        others_id = f"others{i + 1}"
        layer_nodes[i + 1] = high_nodes[i + 1] + ([others_id] if others_id in others_with_edges else [])
    root_layer = n_hidden_layers + 2
    layer_nodes[root_layer] = ["root"]

    for layer_idx, nodes in sorted(layer_nodes.items()):
        print(f"  layer_nodes[{layer_idx}]: {len(nodes)} nodes — {nodes}")

    all_node_ids = []
    node_to_idx = {}
    for layer_idx in sorted(layer_nodes.keys()):
        for node_id in layer_nodes[layer_idx]:
            key = (layer_idx, node_id)
            if key not in node_to_idx:
                node_to_idx[key] = len(all_node_ids)
                all_node_ids.append(node_id)
    print(f"  Total nodes: {len(all_node_ids)}")

    display_name_map = {
        "mut_important": "mutation",
        "cnv_amp": "amplification",
        "cnv_del": "deletion",
        "root": "outcome",
    }
    all_node_labels = []
    for node_id in all_node_ids:
        if node_id in display_name_map:
            all_node_labels.append(display_name_map[node_id])
        elif node_id.startswith("others"):
            all_node_labels.append("residual")
        else:
            all_node_labels.append(id_to_name.get(node_id, node_id))

    # -------------------------------------------------------------------
    # 6. encode edges
    # -------------------------------------------------------------------
    print(f"\n--- Section 6: Edge encoding ---")
    diagram_source = []
    diagram_target = []
    diagram_values = []
    missing_keys = []

    for _, row in df.iterrows():
        layer_idx = int(row["layer"])
        src_key = (layer_idx, row["source"])
        tgt_key = (layer_idx + 1, row["target"])
        if src_key in node_to_idx and tgt_key in node_to_idx:
            diagram_source.append(node_to_idx[src_key])
            diagram_target.append(node_to_idx[tgt_key])
            diagram_values.append(row["value"])
        else:
            missing_keys.append((src_key, src_key in node_to_idx,
                                  tgt_key, tgt_key in node_to_idx))

    if missing_keys:
        print(f"  WARNING: {len(missing_keys)} edges dropped due to missing keys:")
        for mk in missing_keys[:5]:
            print(f"    src={mk[0]} found={mk[1]}, tgt={mk[2]} found={mk[3]}")

    print(f"  Encoded edges (excl. root): {len(diagram_source)}")

    # connect last pathway layer to root
    last_layer_idx = n_hidden_layers + 1
    root_idx = node_to_idx[(root_layer, "root")]
    last_importance = deeplift[f"h{n_hidden_layers}"]["coef"]
    n_root_edges = 0
    for node_id in layer_nodes[last_layer_idx]:
        key = (last_layer_idx, node_id)
        if key in node_to_idx:
            imp = 1.0 if node_id.startswith("others") else float(last_importance.get(node_id, 0))
            if imp > 0:
                diagram_source.append(node_to_idx[key])
                diagram_target.append(root_idx)
                diagram_values.append(imp)
                n_root_edges += 1
    print(f"  Root edges added: {n_root_edges}")
    print(f"  Total encoded edges: {len(diagram_source)}")

    # -------------------------------------------------------------------
    # 7. node positions matching original get_x_y
    #    use max(source_flow, target_flow) per node matching original
    # -------------------------------------------------------------------
    print(f"\n--- Section 7: Node positions ---")
    x_positions_map = {
        0: 0.01, 1: 0.08, 2: 0.14, 3: 0.32, 4: 0.48, 5: 0.64, 6: 0.8,
        root_layer: 0.99
    }

    n_nodes = len(all_node_labels)

    # matching original get_x_y: max of source_sum and target_sum per node
    source_flow = np.zeros(n_nodes)
    target_flow = np.zeros(n_nodes)
    for src, tgt, val in zip(diagram_source, diagram_target, diagram_values):
        source_flow[src] += abs(val)
        target_flow[tgt] += abs(val)
    node_flow = np.maximum(source_flow, target_flow)
    print(f"  node_flow range: [{node_flow.min():.2f}, {node_flow.max():.2f}]")

    x_pos = [0.0] * n_nodes
    y_pos = [0.0] * n_nodes

    for layer_idx in sorted(layer_nodes.keys()):
        nodes = layer_nodes[layer_idx]
        idxs = [node_to_idx[(layer_idx, n)] for n in nodes]
        flows = node_flow[idxs].copy()

        print(f"  Layer {layer_idx} flows (pre-sort):")
        for n, f in zip(nodes, flows):
            print(f"    {n}: {f:.2f}")

        # others forced to bottom then restored — matching original get_x_y
        is_others = np.array([n.startswith("others") or n == "root" for n in nodes])
        others_flows = flows[is_others].copy()
        flows[is_others] = 0.0
        sort_order = np.argsort(flows)[::-1]
        flows[is_others] = others_flows
        flows = flows[sort_order]

        print(f"  Layer {layer_idx} sort order: {[nodes[i] for i in sort_order]}")

        layer_total = flows.sum() or 1.0
        cumulative = np.cumsum(flows)
        ys = (cumulative - 0.5 * flows) / (1.5 * layer_total)

        x_val = x_positions_map.get(layer_idx, 0.99)
        for rank, orig_rank in enumerate(sort_order):
            node_idx = idxs[orig_rank]
            x_pos[node_idx] = x_val
            y_pos[node_idx] = float(ys[rank])
            print(f"    rank={rank} node={nodes[orig_rank]} y={ys[rank]:.3f}")

    # root fixed at 0.33 matching original
    y_pos[root_idx] = 0.33
    print(f"  root y fixed at 0.33")

    # -------------------------------------------------------------------
    # 8. colours matching original get_node_colors_ordered
    # -------------------------------------------------------------------
    print(f"\n--- Section 8: Colours ---")
    cmap = plt.cm.Reds
    node_colours = ['rgba(128,128,128,0.5)'] * n_nodes

    for layer_idx in sorted(layer_nodes.keys()):
        nodes = layer_nodes[layer_idx]
        important_nodes = [n for n in nodes if not n.startswith("others") and n != "root"]
        n = len(important_nodes)
        colour_idx = np.linspace(1.0, 0.0, n)
        for j, node_id in enumerate(important_nodes):
            key = (layer_idx, node_id)
            if key in node_to_idx:
                colors = list(cmap(colour_idx[j]))
                colors = [int(255 * c) for c in colors[:3]]
                node_colours[node_to_idx[key]] = f'rgba({colors[0]},{colors[1]},{colors[2]},0.7)'
        for node_id in nodes:
            if node_id.startswith("others"):
                key = (layer_idx, node_id)
                if key in node_to_idx:
                    node_colours[node_to_idx[key]] = 'rgba(232,232,232,0.5)'

    node_colours[root_idx] = 'rgba(255,100,100,0.7)'

    for node_id, colour in [
        ("mut_important", 'rgba(105,189,210,0.7)'),
        ("cnv_amp", 'rgba(224,123,57,0.7)'),
        ("cnv_del", 'rgba(1,55,148,0.7)')
    ]:
        key = (0, node_id)
        if key in node_to_idx:
            node_colours[node_to_idx[key]] = colour
            print(f"  Set {node_id} colour to {colour}")

    edge_colours = []
    for src in diagram_source:
        base = node_colours[src]
        edge_colours.append(base.replace('0.7', '0.2').replace('0.5', '0.2'))

    # -------------------------------------------------------------------
    # 9. render matching original layout
    # -------------------------------------------------------------------
    print(f"\n--- Section 9: Render ---")
    print(f"  Total nodes: {len(all_node_labels)}")
    print(f"  Total edges: {len(diagram_source)}")
    print(f"  x_pos range: [{min(x_pos):.3f}, {max(x_pos):.3f}]")
    print(f"  y_pos range: [{min(y_pos):.3f}, {max(y_pos):.3f}]")

    scale = 1.0
    width = 600. / scale
    height = 0.5 * width / scale

    data_trace = dict(
        type='sankey',
        arrangement='snap',
        domain=dict(x=[0, 1.], y=[0, 1.]),
        orientation="h",
        valueformat=".0f",
        node=dict(
            pad=2,
            thickness=10,
            line=dict(color="white", width=0.5),
            label=all_node_labels,
            x=x_pos,
            y=y_pos,
            color=node_colours,
        ),
        link=dict(
            source=diagram_source,
            target=diagram_target,
            value=diagram_values,
            color=edge_colours,
        )
    )

    layout = dict(
        height=height,
        width=width,
        margin=dict(l=0, r=0, b=0.1, t=8),
        font=dict(size=6, family='Arial')
    )

    fig = go.Figure(dict(data=[data_trace], layout=layout))
    fig.write_image(f"{figures_dir}/sankey.pdf", scale=1, width=width, height=height, format='pdf')
    fig.write_image(f"{figures_dir}/sankey.png", scale=5, width=width, height=height, format='png')
    print(f"  Saved sankey.pdf and sankey.png")

    scale = 0.5
    width_html = 600. / scale
    height_html = 0.5 * width_html
    layout_html = dict(
        height=height_html,
        width=width_html,
        margin=dict(l=0, r=0, b=0.1, t=8),
        font=dict(size=12, family='Arial')
    )
    fig_html = go.Figure(dict(data=[data_trace], layout=layout_html))
    fig_html.write_html(f"{figures_dir}/sankey.html")
    print(f"  Saved sankey.html")
    print(f"=== plot_sankey complete ===\n")


def plot(wd, run_dir, selected_genes, n_hidden_layers):
    figures_dir = os.path.join(wd, "figures")
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    #plot_nested_CV(run_dir, figures_dir)

    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(7, 14))
    #plot_single_split_curves(run_dir, wd, figures_dir)
    #plot_train_size_comparisons(run_dir, ax[1], ax[2], figures_dir)
    # plt.tight_layout()
    # plt.savefig(f"{wd}/figure_1.jpg")
    # plt.close()
    plot_sankey(wd, run_dir, selected_genes, n_hidden_layers, figures_dir)
    #
    # plot_stratified_5_fold_CV(run_dir, figures_dir)