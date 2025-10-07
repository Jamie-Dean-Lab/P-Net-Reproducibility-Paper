import os, sys, json
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, f1_score
from scipy.stats import ttest_ind

sys.path.insert(0, os.getcwd())
from architecture.data_utils import *
from architecture.pnet_config import *
from architecture.pipeline import *
from architecture.evaluation import *
from architecture.callbacks_custom import FixedEarlyStopping
from ovr import *

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

pvd = [ttest_ind(results["pnet"].loc[results["pnet"]["index"] == "test", x], results["dense"].loc[results["dense"]["index"] == "test", x]).pvalue for x in metrics]
pvs = [ttest_ind(results["pnet"].loc[results["pnet"]["index"] == "test", x], results["svc"].loc[results["pnet"]["index"] == "test", x]).pvalue for x in metrics]
svd = [ttest_ind(results["svc"].loc[results["pnet"]["index"] == "test", x], results["dense"].loc[results["pnet"]["index"] == "test", x]).pvalue for x in metrics]
sigresults = pd.DataFrame((pvd, pvs, svd), columns=metrics, index=["pnet_v_dense", "pnet_v_svc", "svc_v_dense"])
sigresults.to_csv(f"{wd}/significance_tests.csv")