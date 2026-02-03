import os, sys
import pandas as pd
import json
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score
from keras.activations import linear, relu, tanh, leaky_relu
from keras.losses import MeanSquaredError
from functools import partial
from sklearn.kernel_ridge import KernelRidge
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.getcwd())
from architecture.data_utils import *
from architecture.pnet_config import *
from architecture.pipeline import *
from architecture.evaluation import *
from architecture.callbacks_custom import step_decay, FixedEarlyStopping
from scipy.stats import ttest_rel, t

# Download data if not done so already and set up run directory
wd = "Radiosensitivity Prediction"
download_dir = f"{wd}/data"
data_dir = f"{download_dir}/Cleveland"
run_dir = f"{wd}/runs"

if not os.path.exists(download_dir):
    with open(f"{wd}/src/download_data.py") as file:
        exec(file.read())

if not os.path.exists(run_dir):
    os.mkdir(run_dir)

# Identify protein coding only genes and selected gene list based on P-Net paper
selected_genes = list(set(pd.read_csv(f"{download_dir}/hugo_genes.txt", sep="\t")["symbol"]))

# prepare config
config["data_dir"] = data_dir
config["run_dir"] = run_dir
config["run_id"] = "pnet"
config["views"] = [("gexpr", f"cleveland_gene_expression.csv", selected_genes, 0, lambda x : x, lambda x : x),
                   ("methylation", f"CCLE_Methylation_TSS1kb_20181022.csv", selected_genes, 0, lambda x : x, lambda x : x)]
config["view_alignment_method"] = "drop samples"
config["labels"] = [("cleveland_auc_only.csv", 0)]
config["tv_split_seed"] = 42
config["stratified"] = False
# This block is for nested CV
config["inner_kfolds"] = 5
config["outer_kfolds"] = 10

# This block is for standard kfold CV
#config["inner_kfolds"] = 5
#config["outer_kfolds"] = 1
#config["test_samples"] = 0.1 # Specifies proportion of data to be used as test set. Also accepts list of integers to specify specific samples for test set
# end of block
config["use_validation_on_test"] = False # Specifies whether to use validation sets for final testing after CV
config["val_metric"] = {"r2" : lambda x : r2_score(x["val_df"].ys, x["val_preds"])} # This line specifies metric to use for selecting best model from cv
config["results_processors"] = [lambda x : save_results(x, save_supervised_result, {"r2" : r2_score,
                                                                                    "explained_variance" : explained_variance_score,
                                                                                    "mse" : mean_squared_error,
                                                                                    "mae" : mean_absolute_error}, 
                                                                          "individual")]

n_hidden_layers = 5

step_decay_part = partial(
    step_decay,
    init_lr=0.001,
    drop=0.5,
    epochs_drop=25,
)

config["fitting_params"] = {
                                "epochs" : 200,
                                "batch" : 50,
                                "LRScheduler" : LearningRateScheduler(step_decay_part, verbose=0),
                                "early_stopping" : None,
                                "prediction_output" : "average",
                                "shuffle_samples" : True,
                                "class_weight" : None
                            }

gs_params = {"model_params" : {f"reg_{l}" : {
                            "pp_relations" : "architecture/Reactome/ReactomePathwaysRelation.txt",
                            "gp_relations" : "architecture/Reactome/ReactomePathways.gmt",
                            "n_hidden_layers" : n_hidden_layers,
                            "h_dropout" : [0.5] + [0.1] * n_hidden_layers,
                            "h_activation" : ["tanh"] * (n_hidden_layers + 1),
                            "o_activation" : ["linear"] * (n_hidden_layers + 1),
                            "h_reg" : [(L2, {"l2" : 10 ** l})] * (n_hidden_layers + 1),
                            "o_reg" : [(L2, {"l2" : 10 ** l})] * (n_hidden_layers + 1),
                            "h_kernel_initializer" : ["lecun_uniform"] * (n_hidden_layers + 1),
                            "h_kernel_constraints" : [None] * (n_hidden_layers + 1),
                            "h_bias_initializer" : ["lecun_uniform"] * (n_hidden_layers + 1),
                            "h_bias_constraints" : [None] * (n_hidden_layers + 1),
                            "batch_normal" : False,
                            "sparse" : True,
                            "dropout_testing" : False,
                            "loss" : ["MeanSquaredError"] * (n_hidden_layers + 1),
                            "loss_weights" : [2, 7, 20, 54, 148, 400],
                            "optimizer" : {"class_name" : "Adam", "config" : {"learning_rate" : 1e-3}}
                        } for l in [-1, -2, -3, -4]}}

config["grid_search"] = construct_gs_params(gs_params)

# Run pnet crossvalidation
pipeline = TFPipeline(config)
pipeline.run_crossvalidation()

# Run dense crossvalidation
gs_params = {"model_params" : {f"reg_{l}" : {
                            "pp_relations" : "architecture/Reactome/ReactomePathwaysRelation.txt",
                            "gp_relations" : "architecture/Reactome/ReactomePathways.gmt",
                            "n_hidden_layers" : n_hidden_layers,
                            "h_dropout" : [0.5] + [0.1] * n_hidden_layers,
                            "h_activation" : ["tanh"] * (n_hidden_layers + 1),
                            "o_activation" : ["linear"] * (n_hidden_layers + 1),
                            "h_reg" : [(L2, {"l2" : 10 ** l})] * (n_hidden_layers + 1),
                            "o_reg" : [(L2, {"l2" : 10 ** l})] * (n_hidden_layers + 1),
                            "h_kernel_initializer" : ["lecun_uniform"] * (n_hidden_layers + 1),
                            "h_kernel_constraints" : [None] * (n_hidden_layers + 1),
                            "h_bias_initializer" : ["lecun_uniform"] * (n_hidden_layers + 1),
                            "h_bias_constraints" : [None] * (n_hidden_layers + 1),
                            "batch_normal" : False,
                            "sparse" : False,
                            "dropout_testing" : False,
                            "loss" : ["MeanSquaredError"] * (n_hidden_layers + 1),
                            "loss_weights" : [2, 7, 20, 54, 148, 400],
                            "optimizer" : {"class_name" : "Adam", "config" : {"learning_rate" : 1e-3}}
                        } for l in [-1, -2, -3, -4]}}

config["grid_search"] = construct_gs_params(gs_params)
config["run_id"] = "dense"
pipeline = TFPipeline(config)
pipeline.run_crossvalidation()

# Run Kernel Regression
gs_params = {"model_params" : {f"degree_{d}_alpha_{a}" : {"kernel" : "poly", "degree" : d, "alpha" : a}
                               for d in [1, 2, 3] for a in [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]}}
config["model"] = KernelRidge
config["run_id"] = "krr"
config["task"] = "regression"
config["results_processors"] = config["results_processors"]
config["grid_search"] = construct_gs_params(gs_params)
pipeline = MLPipeline(config)
pipeline.run_crossvalidation()

metrics = ["auc_r2", "auc_explained_variance" ,"auc_mse", "auc_mae"]
results = {m : pd.read_csv(f"{run_dir}/{m}/results.csv", index_col=0) for m in os.listdir(run_dir)}
result_table = []
for k,v in results.items():
    df = v.groupby("index")[metrics].agg(["mean", "std"])
    df["model"] = k
    result_table.append(df)
result_table = pd.concat(result_table)
result_table.to_csv(f"{wd}/results.csv")

results_full = []

comparisons = [
    ("pnet", "dense", "pnet_vs_dense"),
    ("pnet", "krr", "pnet_vs_krr"),
    ("dense", "krr", "dense_vs_krr")
]

for metric in metrics:
    for model1, model2, comp_name in comparisons:
        scores1 = results[model1].loc[results[model1]["index"] == "test", metric].values
        scores2 = results[model2].loc[results[model2]["index"] == "test", metric].values

        # One-sided paired t-test (model2 > model1)
        t_stat, p_raw = ttest_rel(scores2, scores1, alternative='greater')

        diffs = scores2 - scores1
        n = len(diffs)
        df = n - 1
        mean_diff = np.mean(diffs)
        sem = np.std(diffs, ddof=1) / np.sqrt(n)
        cohens_d = mean_diff / np.std(diffs, ddof=1)

        # 95% CI for mean_diff
        t_crit = t.ppf(0.975, df)
        ci_low = mean_diff - t_crit * sem
        ci_high = mean_diff + t_crit * sem

        results_full.append({
            'comparison': f'{comp_name}_{metric}',
            'model1': model1, 'model2': model2, 'metric': metric,
            't': t_stat, 'df': df, 'mean_diff': mean_diff, 'sem': sem,
            'ci_low': ci_low, 'ci_high': ci_high, 'cohens_d': cohens_d, 'p_raw': p_raw
        })

# FDR correction across all 12 tests
rejected, p_fdr, _, _ = multipletests([r['p_raw'] for r in results_full], alpha=0.05, method='fdr_bh')

sigresults = pd.DataFrame(results_full)
sigresults['p_fdr'] = p_fdr
sigresults['significant'] = rejected

print("Radiosensitivity full results (t(df), CI, Cohen's d, FDR p-values):")
print(sigresults.round(4))
sigresults.to_csv(f"{wd}/significance_tests.csv", float_format='%.4f')
