import os, sys
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

from keras.models import Sequential

from pnet_model import get_layer_maps, PNetArchitectureGenerator

# Anchor sys.path to the project root relative to this file's location
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import architecture.coef_weights_utils as mcw


def collate_grid_search(results: dict):
    """
    Collates summary results from the best hyperparameter directories across all
    test folds into a single results.csv file for the run.
    """
    summaries = []
    run_dir = results["save_dir"]
    gs_params = results["params"]
    gs_dirs = results["gs_dirs"]
    for i, d in enumerate(gs_dirs):
        df = pd.read_csv(f"{d}/summary_results.csv", index_col=0)
        df["test_fold"] = os.path.basename(os.path.dirname(d))
        df["metric"] = os.path.basename(d).replace("best_", "")
        df["hyperparams"] = str(gs_params[i])
        summaries.append(df)
    summaries = pd.concat(summaries).reset_index()
    summaries.to_csv(f"{run_dir}/results.csv")


def collate_aggregate_results(results: dict):
    """
    Reads the results.csv produced by collate_grid_search and writes
    aggregated_results.csv (mean/std per split per hyperparameter set) to the same directory.
    """
    run_dir = results["save_dir"]
    df = pd.read_csv(f"{run_dir}/results.csv", index_col=0)
    non_metric_cols = {"index", "test_fold", "metric", "hyperparams"}
    metric_cols = [c for c in df.columns if c not in non_metric_cols]
    agg = df.groupby("index")[metric_cols].agg(["mean", "std"])
    agg.reset_index().to_csv(f"{run_dir}/aggregated_results.csv", index=False)


def collate_folds(results: dict):
    """
    Function to perform any post-processing across folds after the runs have finished.
    """
    summaries = []
    for i, result in enumerate(results["results"]):
        df = pd.read_csv("{}/summary_results.csv".format(result), index_col=0)
        df = df.reset_index(names="split")
        df["fold"] = i
        summaries.append(df)
    summaries = pd.concat(summaries)
    summaries.to_csv("{}/fold_summaries.csv".format(results["save_dir"]))


def save_results(results: dict, processor, metrics: dict, task: str, pred_idx: int = 0):
    """
    Function for saving the results based on a given processor function and metrics provided.

    args:
        results (dict) : Results dictionary passed in via pipeline
        processor (function) : A function accepting a results dictionary, the fold string,
                                run directory, and metrics dictionary - outputs a dictionary
                                of results with keys same as metrics
        metrics (dict) : Dictionary of metric functions to evaluate the results
    """
    run_dir = results["save_dir"]
    result_summary = []
    idxs = []
    idxs.append("train")
    result_summary.append(processor(results, "train", run_dir, metrics, task, pred_idx))
    if len(results["val_df"]) > 0:
        idxs.append("val")
        result_summary.append(processor(results, "val", run_dir, metrics, task, pred_idx))
    if len(results["test_df"]) > 0:
        idxs.append("test")
        result_summary.append(processor(results, "test", run_dir, metrics, task, pred_idx))
    result_summary = pd.DataFrame(result_summary, index=idxs)
    result_summary.to_csv(f"{run_dir}/summary_results.csv")
    return result_summary


def save_supervised_result(results: dict, split: str, run_dir: str, metrics: dict, task: str,
                            pred_idx: int = 0):
    """
    Saves each split of supervised results e.g train, validation, test.

    args:
        results (dict) : results dictionary containing all the information from the run
        split (str) : one of train, val, test to specify which split to save over
        run_dir (str) : path to the directory of the current fold run
        metrics (dict) : dictionary containing the metrics that you want computed
        task (str) : Specifies whether to treat the input labels as a group or individually
        pred_idx (int) : Assumes that model predictions are returned as a tuple, identifies
                        the item in the tuple that corresponds to supervised predictions

    returns:
        dict : Dictionary consisting of the summary metrics computed on the data
    """
    preds = results[f"{split}_preds"][pred_idx] if type(results[f"{split}_preds"]) is tuple else results[f"{split}_preds"]
    label_dims = len(results[f"{split}_df"].ys.shape)
    # Fix: use pure NumPy instead of PyTorch-style flatten(start_dim=...)
    if len(preds.shape) > label_dims:
        preds = preds.reshape(preds.shape[0], -1).mean(axis=-1, keepdims=True)
    preds = preds.reshape(results[f"{split}_df"].ys.shape)
    cols = results[f"{split}_df"].get_labels() + [f"{x}_pred" for x in results[f"{split}_df"].get_labels()]
    df = pd.DataFrame(
        np.concatenate((results[f"{split}_df"].ys, preds), axis=1),
        columns=cols,
        index=results[f"{split}_df"].ids
    )
    df.to_csv(f"{run_dir}/{split}_results.csv")
    out = {}
    for metric_name, metric_fn in metrics.items():
        if task == "individual":
            for i, label in enumerate(results[f"{split}_df"].get_labels()):
                is_na = np.isnan(results[f"{split}_df"].ys[:, i])
                out[f"{label}_{metric_name}"] = metric_fn(results[f"{split}_df"].ys[~is_na, i], preds[~is_na, i])
        elif task == "group":
            out[f"{metric_name}"] = metric_fn(results[f"{split}_df"].ys, preds)
    return out


def evaluate_on_external(results: dict, external_df: pd.DataFrame, tag: str):
    """
    Evaluates the current best model on an external dataset.
    """
    sample_ids = external_df.index
    model = results["model"]
    model_inputs = set(results["train_df"].alignment_ids)
    external_inputs = set(external_df.columns)
    missing_inputs = list(model_inputs - external_inputs)
    with open(results["save_dir"] + f"/{tag}_feature_info.csv", "w") as f:
        f.write("model_features,dataset_features,overlap_features,missing_features\n")
        f.write("{},{},{},{}".format(
            len(model_inputs), len(external_inputs),
            len(model_inputs.intersection(external_inputs)), len(missing_inputs)
        ))
    missing_df = pd.DataFrame(
        np.zeros((external_df.shape[0], len(missing_inputs))),
        index=external_df.index,
        columns=missing_inputs
    )
    df = pd.concat((external_df, missing_df), axis=1)
    df = df[results["train_df"].alignment_ids]
    preds = model.predict(df)
    preds = pd.DataFrame(preds, index=sample_ids, columns=["auc_pred"])
    preds.to_csv(results["save_dir"] + f"/{tag}_results.csv", index=True)


def plot_channels(
    history,
    channels,
    filename: str,
    folder_name: str,
    xlabel: str = "epochs",
    ylabel: str = " ",
):
    """
    Plot the training history.
    """
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    plt.figure()
    for k in channels:
        v = history[k]
        plt.plot(v)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(channels)
    filename = os.path.join(folder_name, filename)
    plt.savefig(filename + ".pdf")
    plt.close()


def plot_history(results):
    """
    Make some plots of the history of training the model.
    """
    folder_name = results["save_dir"] + "/train_hx"
    history = results["train_hx"].history
    keys = list(history.keys())

    losses = [x for x in keys if ("_loss" in x) and (x != "val_loss")]
    val_losses = [x for x in losses if "val_" in x]
    train_losses = [x for x in losses if ("val_" not in x) and (x != "loss")]

    monitors = [x for x in keys if "loss" not in x]
    val_monitors = [x for x in monitors if "val_" in x]
    train_monitors = [x for x in monitors if ("val_" not in x) and (x != "loss") and (x != "lr")]

    monitors.sort()
    val_monitors.sort()
    train_monitors.sort()
    train_losses.sort()
    val_losses.sort()

    plot_channels(history, val_monitors, "val_monitors", folder_name, ylabel="Score [arb]")
    plot_channels(history, train_monitors, "train_monitors", folder_name, ylabel="Score [arb]")

    for v, t in zip(val_monitors, train_monitors):
        plot_channels(history, [v, t], t, folder_name, ylabel="Score [arb]")

    plot_channels(history, val_losses, "validation_loss", folder_name, ylabel="Loss ")
    plot_channels(history, train_losses, "training_loss", folder_name, ylabel="Loss")

    if "val_loss" in keys:
        plot_channels(history, ["val_loss", "loss"], "loss", folder_name, ylabel="Loss")
    else:
        plot_channels(history, ["loss"], "loss", folder_name, ylabel="Score [arb]")

    for v, t in zip(val_losses, train_losses):
        plot_channels(history, [v, t], t, folder_name, ylabel="Score [arb]")
    pd.DataFrame(history).to_csv(f"{folder_name}/train_hx.csv")


def get_coef_importance(model, X_train, y_train, target, feature_importance, detailed=True, **kwargs):

    print(feature_importance)

    if feature_importance.startswith("skf"):
        coef_ = mcw.get_skf_weights(model, X_train, y_train, feature_importance)
    elif feature_importance == "loss_gradient":
        coef_ = mcw.get_gradient_weights(model, X_train, y_train, signed=False, detailed=detailed, normalize=True)
    elif feature_importance == "loss_gradient_signed":
        coef_ = mcw.get_gradient_weights(model, X_train, y_train, signed=True, detailed=detailed, normalize=True)
    elif feature_importance == "gradient_outcome":
        coef_ = mcw.get_weights_gradient_outcome(model, X_train, y_train, target, multiply_by_input=False, signed=False)
    elif feature_importance == "gradient_outcome_signed":
        coef_ = mcw.get_weights_gradient_outcome(model, X_train, y_train, target=target, detailed=detailed, multiply_by_input=False, signed=True)
    elif feature_importance == "gradient_outcome*input":
        coef_ = mcw.get_weights_gradient_outcome(model, X_train, y_train, target, multiply_by_input=True, signed=False)
    elif feature_importance == "gradient_outcome*input_signed":
        coef_ = mcw.get_weights_gradient_outcome(model, X_train, y_train, target, multiply_by_input=True, signed=True)
    elif feature_importance.startswith("deepexplain"):
        method = feature_importance.split("_")[1]
        coef_ = mcw.get_deep_explain_scores(model, X_train, y_train, target, method_name=method, detailed=detailed, **kwargs)
    elif feature_importance.startswith("shap"):
        method = feature_importance.split("_")[1]
        coef_ = mcw.get_shap_scores(model, X_train, y_train, target, method_name=method, detailed=detailed)
    elif feature_importance == "gradient_with_repeated_outputs":
        coef_ = mcw.get_gradient_weights_with_repeated_output(model, X_train, y_train, target)
    elif feature_importance == "permutation":
        coef_ = mcw.get_permutation_weights(model, X_train, y_train)
    elif feature_importance == "linear":
        coef_ = mcw.get_weights_linear_model(model, X_train, y_train)
    elif feature_importance == "one_to_one":
        weights = model.layers[1].get_weights()
        coef_ = np.abs(weights[0])
    else:
        coef_ = None
    return coef_


def get_layers(model, level=1):
    layers = []
    for l in model.layers:
        if isinstance(l, Sequential):
            layers.extend(get_layers(l, level + 1))
        else:
            layers.append(l)
    return layers

def adjust_deeplift_for_degree(coef_df, maps, layer_idx):
    curr = maps[layer_idx].copy()
    curr[curr != 0] = 1.0

    if layer_idx == 0:
        fan_out = curr.abs().sum(axis=1)
        degree = fan_out
    else:
        prev_map = maps[layer_idx - 1].copy()
        prev_map[prev_map != 0] = 1.0
        fan_in = prev_map.abs().sum(axis=0)
        fan_out = curr.abs().sum(axis=1)
        degree = fan_in.add(fan_out, fill_value=0)

    df = coef_df.copy()
    df.columns = ["coef"]
    df["coef_graph"] = degree.reindex(df.index).fillna(0)

    # diagnostics for layer 0
    if layer_idx == 0:
        zero_degree_genes = df[df["coef_graph"] == 0].index.tolist()
        nonzero_in_map = (maps[0].loc[maps[0].index.isin(zero_degree_genes)] != 0).sum(axis=1)
        print(f"  Non-zero connections for zero-degree genes: {nonzero_in_map.sum()}")
        print(f"  Zero-degree genes in maps[0]: {(~maps[0].index.isin(zero_degree_genes)).sum()} connected, {maps[0].index.isin(zero_degree_genes).sum()} unconnected")

    n_missing = df["coef_graph"].eq(0).sum()
    mean = df["coef_graph"].mean()
    std = df["coef_graph"].std()
    threshold = mean + 5 * std
    ind = df["coef_graph"] > threshold
    n_hubs = ind.sum()

    print(f"  Nodes: {len(df)}, degree mean={mean:.2f}, std={std:.2f}, threshold={threshold:.2f}")
    print(f"  Hub nodes penalised: {n_hubs}, zero degree (index mismatch): {n_missing}")

    divide = df["coef_graph"].copy()
    divide[~ind] = 1.0
    df["coef_combined"] = df["coef"] / divide

    z = df["coef_combined"]
    df["coef_combined_zscore"] = (z - z.mean()) / z.std(ddof=0)

    z1 = (df["coef_graph"] - df["coef_graph"].mean()) / df["coef_graph"].std(ddof=0)
    z2 = (df["coef"] - df["coef"].mean()) / df["coef"].std(ddof=0)
    z = z2 - z1
    df["coef_combined2"] = (z - z.mean()) / z.std(ddof=0)
    df["feature_importance_adjusted"] = df["coef_combined_zscore"]

    if n_hubs > 0:
        top_hubs = df[ind].sort_values("coef_graph", ascending=False).head(5)
        print(f"  Top hubs:\n{top_hubs[['coef_graph', 'coef', 'feature_importance_adjusted']]}")

    return df


def get_layer_weights(layer):
    from scipy.sparse import csr_matrix
    layer_type = type(layer).__name__

    if layer_type in ('Diagonal', 'SparseTF'):
        w = layer.get_weights()[0]  # flat non-zero values
        row_ind = layer.nonzero_ind[:, 0]
        col_ind = layer.nonzero_ind[:, 1]
        w_sparse = csr_matrix((w, (row_ind, col_ind)), shape=layer.kernel_shape)
        return np.array(w_sparse.todense())
    else:
        # standard Dense
        return layer.get_weights()[0]


def get_link_weights_df_(model, features, layer_names):
    link_weights_df = {}
    for i, layer_name in enumerate(layer_names[1:]):
        try:
            layer = model.get_layer(layer_name)
        except ValueError:
            print(f"  Layer {layer_name} not found in model, skipping")
            continue

        previous_layer_name = layer_names[i]
        rows = features[previous_layer_name]
        cols = features[layer_name]

        print(f"  Extracting {previous_layer_name} -> {layer_name}: "
              f"type={type(layer).__name__}, expected shape=({len(rows)}, {len(cols)})")

        w = get_layer_weights(layer)
        print(f"  Got weight matrix shape: {w.shape}")

        w_df = pd.DataFrame(w, index=rows, columns=cols)
        link_weights_df[layer_name] = w_df

    return link_weights_df


def get_deeplift_global(results, selected_genes, n_hidden_layers,
                        pathway_dataset, pp_relations, gp_relations):
    print("Computing DeepLIFT global importance scores")

    global_coefs, sample_coefs = get_coef_importance(
        results["model"].predictor, results["train_df"].xs, results["train_df"].ys, -1, "deepexplain_deeplift"
    )
    print(f"sample_coefs keys: {list(sample_coefs.keys())}")
    for k, v in sample_coefs.items():
        print(f"  sample_coefs['{k}'] type={type(v)}, shape={v.shape if hasattr(v, 'shape') else len(v)}")

    features = results["model"].feature_names
    features["inputs"] = [x[1] for x in features["inputs"]]

    reactome = PNetArchitectureGenerator()
    netx = reactome.get_networkx(pp_relations, pathway_dataset)
    maps = reactome.get_layers(netx, n_hidden_layers, gp_relations, selected_genes)
    maps = get_layer_maps(pd.Index(features["h0"]), maps, False)
    print(f"Built {len(maps)} layer maps from Reactome")

    # verify index alignment
    for i, m in enumerate(maps):
        k = f"h{i}"
        if k in global_coefs:
            map_idx = set(m.index)
            dlift_idx = set(features[k])
            overlap = map_idx & dlift_idx
            print(f"Layer {k}: maps={len(map_idx)}, deeplift={len(dlift_idx)}, overlap={len(overlap)}, "
                  f"missing from maps={len(dlift_idx - map_idx)}, missing from deeplift={len(map_idx - dlift_idx)}")

    # extract link weights from trained model
    print("\n--- Extracting link weights ---")
    layer_names = ['inputs', 'h0', 'h1', 'h2', 'h3', 'h4', 'h5']
    link_weights = get_link_weights_df_(results["model"].predictor, features, layer_names)
    for i, (layer_name, df) in enumerate(link_weights.items()):
        save_path = results["save_dir"] + f"/link_weights_{i}.csv"
        df.to_csv(save_path)
        print(f"  Saved link weights {layer_name} shape={df.shape} to {save_path}")

    # save deeplift importance scores
    for k, v in global_coefs.items():
        df = pd.DataFrame(np.abs(v), index=features[k], columns=["feature_importance"])

        if k.startswith("h"):
            layer_idx = int(k[1:])
            df = adjust_deeplift_for_degree(df, maps, layer_idx)
            print(f"Layer {k}: feature_importance_adjusted range="
                  f"[{df['feature_importance_adjusted'].min():.4f}, {df['feature_importance_adjusted'].max():.4f}]")
        else:
            print(f"Skipping adjustment for {k} (inputs layer)")

        save_path = results["save_dir"] + f"/feature_importance_{k}.csv"
        df.to_csv(save_path)
        print(f"Saved {k} importance to {save_path}")

    print("DeepLIFT global importance complete")