import os, sys, json
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, f1_score
from scipy.stats import ttest_rel

sys.path.insert(0, os.getcwd())
from architecture.data_utils import *
from architecture.pnet_config import *
from architecture.pipeline import *
from architecture.evaluation import *
from architecture.callbacks_custom import FixedEarlyStopping
from ovr import *
from statsmodels.stats.multitest import multipletests
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from scipy.stats import t  # for manual CI if needed



# Download data if not done so already and set up run directory
wd = "Tissue Type Classification"
download_dir = f"{wd}/data"
data_dir = f"{download_dir}/GTEx"
run_dir = f"{wd}/runs"

if not os.path.exists(download_dir):
    os.mkdir(download_dir)
    with open(f"{wd}/src/download_data.py") as file:
        exec(file.read())

if not os.path.exists(run_dir):
    os.mkdir(run_dir)

# Load hugo genes to use as feature selection
selected_genes = list(set(pd.read_csv(f"{download_dir}/hugo_genes.txt", sep="\t")["symbol"]))

# prepare config
config["data_dir"] = data_dir
config["run_dir"] = run_dir
config["run_id"] = "pnet"
config["views"] = [("gexpr", f"GTEx_gene_log2_tpm_0.csv", selected_genes, 0, lambda x : x, lambda x : x)]
config["view_alignment_method"] = "drop samples"
config["labels"] = [("tissue_classes.csv", 0)]
config["tv_split_seed"] = 42
config["inner_kfolds"] = 5
config["outer_kfolds"] = 10
config["stratified"] = False
config["use_validation_on_test"] = False
config["val_metric"] = {"f1" : lambda x : f1_score(x["val_df"].ys, (x["val_preds"] >= np.sort(x["val_preds"], axis=1)[:,[-1]]).astype(int), average="weighted")}
config["results_processors"] = [lambda x : save_results(x, save_supervised_result, {"auc" : lambda y, y_hat : roc_auc_score(y, y_hat, multi_class="ovr", average="micro"),
                                                                          "auprc" : lambda y, y_hat : average_precision_score(y, y_hat, average="micro"),
                                                                          "f1" : lambda ys, preds : f1_score(ys, (preds >= np.sort(preds, axis=1)[:,[-1]]).astype(int), average="weighted"),
                                                                          "accuracy" : lambda ys, preds : accuracy_score(ys, (preds >= np.sort(preds, axis=1)[:,[-1]]).astype(int))},
                                                                          "group")]

config["model_params"]["loss"] = [{"class_name" : "CategoricalCrossentropy", "config" : {"from_logits" : False}}] * (n_hidden_layers + 1)
config["model_params"]["o_activation"] = ["softmax"] * (n_hidden_layers + 1)
config["fitting_params"]["epochs"] = 200
config["drop_labels"] = True

# Run P-Net
config["run_id"] = "pnet"
gs_params = {"model_params" : {f"reg_{l}" : {
                                "pp_relations" : "architecture/Reactome/ReactomePathwaysRelation.txt",
                                "gp_relations" : "architecture/Reactome/ReactomePathways.gmt",
                                "n_hidden_layers" : n_hidden_layers,
                                "h_dropout" : [0.5] + [0.1] * n_hidden_layers,
                                "h_activation" : ["tanh"] * (n_hidden_layers + 1),
                                "o_activation" : ["sigmoid"] * (n_hidden_layers + 1),
                                "h_reg" : [(L2, {"l2" : 10 ** l})] * (n_hidden_layers + 1),
                                "o_reg" : [(L2, {"l2" : 10 ** l})] * (n_hidden_layers + 1),
                                "h_kernel_initializer" : ["lecun_uniform"] * (n_hidden_layers + 1),
                                "h_kernel_constraints" : [None] * (n_hidden_layers + 1),
                                "h_bias_initializer" : ["lecun_uniform"] * (n_hidden_layers + 1),
                                "h_bias_constraints" : [None] * (n_hidden_layers + 1),
                                "batch_normal" : False,
                                "sparse" : True,
                                "dropout_testing" : False,
                                "loss" : [{"class_name" : "BinaryCrossentropy", "config" : {"from_logits" : False}}] * (n_hidden_layers + 1),
                                "loss_weights" : [2, 7, 20, 54, 148, 400],
                                "optimizer" : {"class_name" : "Adam", "config" : {"learning_rate" : 1e-3}}
                            } for l in [-3, -4, -5, -6]}}
config["grid_search"] = construct_gs_params(gs_params)
pipeline = TFPipeline(config)
pipeline.run_crossvalidation()

# Run fully connected
config["run_id"] = "dense"
gs_params = {"model_params" : {f"reg_{l}" : {
                                "pp_relations" : "architecture/Reactome/ReactomePathwaysRelation.txt",
                                "gp_relations" : "architecture/Reactome/ReactomePathways.gmt",
                                "n_hidden_layers" : n_hidden_layers,
                                "h_dropout" : [0.5] + [0.1] * n_hidden_layers,
                                "h_activation" : ["tanh"] * (n_hidden_layers + 1),
                                "o_activation" : ["sigmoid"] * (n_hidden_layers + 1),
                                "h_reg" : [(L2, {"l2" : 10 ** l})] * (n_hidden_layers + 1),
                                "o_reg" : [(L2, {"l2" : 10 ** l})] * (n_hidden_layers + 1),
                                "h_kernel_initializer" : ["lecun_uniform"] * (n_hidden_layers + 1),
                                "h_kernel_constraints" : [None] * (n_hidden_layers + 1),
                                "h_bias_initializer" : ["lecun_uniform"] * (n_hidden_layers + 1),
                                "h_bias_constraints" : [None] * (n_hidden_layers + 1),
                                "batch_normal" : False,
                                "sparse" : False,
                                "dropout_testing" : False,
                                "loss" : [{"class_name" : "BinaryCrossentropy", "config" : {"from_logits" : False}}] * (n_hidden_layers + 1),
                                "loss_weights" : [2, 7, 20, 54, 148, 400],
                                "optimizer" : {"class_name" : "Adam", "config" : {"learning_rate" : 1e-3}}
                            } for l in [-3, -4, -5, -6]}}
config["grid_search"] = construct_gs_params(gs_params)
pipeline = TFPipeline(config)
pipeline.run_crossvalidation()

# Run base model
config["run_id"] = "svc"
config["model"] = OVRWrapper
config["task"] = "multiclass"
config["model_params"] = {"estimator" : LinearSVC, "args" : {"C" : 0.1}}
config["results_processors"] = [lambda x : save_results(x, save_supervised_result, {"auc" : lambda y, y_hat : roc_auc_score(y, y_hat, multi_class="ovr", average="micro"),
                                                                          "auprc" : lambda y, y_hat : average_precision_score(y, y_hat, average="micro"),
                                                                          "f1" : lambda ys, preds : f1_score(ys, ((preds >= np.sort(preds, axis=1)[:,[-1]]) & (preds > 0)).astype(int), average="weighted"),
                                                                          "accuracy" : lambda ys, preds : accuracy_score(ys, ((preds >= np.sort(preds, axis=1)[:,[-1]]) & (preds > 0)).astype(int))},
                                                                          "group")]
gs_params = {"model_params" : {f"c_{c}" : {"estimator" : LinearSVC, "args" : {"C" : 10 ** c}} for c in [1, 0, -1, -2, -3]}}
config["grid_search"] = construct_gs_params(gs_params)
pipeline = MLPipeline(config)
pipeline.run_crossvalidation()

# Compile results
metrics = ["auc", "auprc" ,"f1", "accuracy"]
results = {m : pd.read_csv(f"{run_dir}/{m}/results.csv", index_col=0) for m in os.listdir(run_dir)}
result_table = []
for k,v in results.items():
    df = v.groupby("index")[metrics].agg(["mean", "std"])
    df["model"] = k
    result_table.append(df)
result_table = pd.concat(result_table)
result_table.to_csv(f"{wd}/results.csv")

results_full = []

for metric in metrics:
    pnet_scores = results["pnet"].loc[results["pnet"]["index"] == "test", metric].values
    dense_scores = results["dense"].loc[results["dense"]["index"] == "test", metric].values
    t_stat, p_raw = stats.ttest_rel(dense_scores, pnet_scores, alternative='greater')

    diffs = dense_scores - pnet_scores
    n = len(diffs)
    df = n - 1
    mean_diff = np.mean(diffs)
    sem = stats.sem(diffs)
    cohens_d = mean_diff / np.std(diffs, ddof=1)

    # 95% CI for mean_diff
    t_crit = t.ppf(0.975, df)
    ci_low = mean_diff - t_crit * sem
    ci_high = mean_diff + t_crit * sem

    results_full.append({
        'comparison': f'dense_vs_pnet_{metric}',
        't': t_stat, 'df': df, 'mean_diff': mean_diff, 'sem': sem,
        'ci_low': ci_low, 'ci_high': ci_high, 'cohens_d': cohens_d, 'p_raw': p_raw
    })

# SVC > P-NET loop
for metric in ["auprc", "f1", "accuracy"]:
    pnet_scores = results["pnet"].loc[results["pnet"]["index"] == "test", metric].values
    svc_scores = results["svc"].loc[results["svc"]["index"] == "test", metric].values
    t_stat, p_raw = stats.ttest_rel(svc_scores, pnet_scores, alternative='greater')

    diffs = svc_scores - pnet_scores
    n = len(diffs)
    df = n - 1
    mean_diff = np.mean(diffs)
    sem = stats.sem(diffs)
    cohens_d = mean_diff / np.std(diffs, ddof=1)

    t_crit = t.ppf(0.975, df)
    ci_low = mean_diff - t_crit * sem
    ci_high = mean_diff + t_crit * sem

    results_full.append({
        'comparison': f'svc_vs_pnet_{metric}',
        't': t_stat, 'df': df, 'mean_diff': mean_diff, 'sem': sem,
        'ci_low': ci_low, 'ci_high': ci_high, 'cohens_d': cohens_d, 'p_raw': p_raw
    })

# FDR correction
rejected, p_fdr, _, _ = multipletests([r['p_raw'] for r in results_full], alpha=0.05, method='fdr_bh')

sigresults = pd.DataFrame(results_full)
sigresults['p_fdr'] = p_fdr
sigresults['significant'] = rejected

print(sigresults.round(4))
sigresults.to_csv(f"{wd}/significance_tests.csv", float_format='%.4f')

