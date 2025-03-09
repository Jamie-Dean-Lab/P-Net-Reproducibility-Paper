import os, sys, json
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, f1_score

sys.path.insert(0, os.getcwd())
from architecture.data_utils import *
from architecture.pnet_config import *
from architecture.pipeline import *
from architecture.evaluation import *
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
config["inner_kfolds"] = 1
config["outer_kfolds"] = 10
config["validation_prop"] = 0.1
config["results_processors"] = [lambda x : save_results(x, save_supervised_result, {"auc" : lambda y, y_hat : roc_auc_score(y, y_hat, multi_class="ovr", average="micro"),
                                                                          "auprc" : lambda y, y_hat : average_precision_score(y, y_hat, average="micro"),
                                                                          "f1" : lambda ys, preds : f1_score(ys, (preds >= np.sort(preds, axis=1)[:,[-1]]).astype(int), average="weighted"),
                                                                          "accuracy" : lambda ys, preds : accuracy_score(ys, (preds >= np.sort(preds, axis=1)[:,[-1]]).astype(int))},
                                                                          "group"),
                            plot_history]

config["model_params"]["loss"] = [{"class_name" : "CategoricalCrossentropy", "config" : {"from_logits" : False}}] * (n_hidden_layers + 1)
config["model_params"]["o_activation"] = ["softmax"] * (n_hidden_layers + 1)
config["fitting_params"]["epochs"] = 200
config["drop_labels"] = True

# Run P-Net
config["run_id"] = "pnet"
pipeline = TFPipeline(config)
#pipeline.run_crossvalidation()

# Run fully connected
config["run_id"] = "dense"
config["model_params"]["sparse"] = False
pipeline = TFPipeline(config)
#pipeline.run_crossvalidation()

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
gs_params = {"model_params" : {f"c_{c}" : {"estimator" : LinearSVC, "args" : {"C" : c}} for c in [0.001, 0.01, 0.1, 1, 10]}}
config["grid_search"] = construct_gs_params(gs_params)
pipeline = MLPipeline(config)
#pipeline.run_crossvalidation()

# Compile results
def compile_results(tag, gridsearch=None):
    results = []
    for i, fold in enumerate([x for x in os.listdir(f"{run_dir}/{tag}") if os.path.isdir(f"{run_dir}/{tag}/{x}")]):
        cvs = [x for x in os.listdir(f"{run_dir}/{tag}/{fold}") if x.find("cv") > -1 and os.path.isdir(f"{run_dir}/{tag}/{fold}/{x}")]
        if len(cvs) > 1:
            for cv in cvs:
                with open(f"{run_dir}/{tag}/{fold}/{cv}/config.txt") as f:
                    run_config = json.loads(f.read())
                    result = pd.read_csv(f"{run_dir}/{tag}/{fold}/{cv}/summary_results.csv")
                    result.columns = ["split"] + list(result.columns[1:])
                    result["test_fold"] = i
                    for k,v in gridsearch.items():
                        result[k] = run_config[v]
                    results.append(result)
        else:
            result = pd.read_csv(f"{run_dir}/{tag}/{fold}/cv_0/summary_results.csv")
            result.columns = ["split"] + list(result.columns[1:])
            result["test_fold"] = i
            results.append(result)
    results = pd.concat(results)
    if len(cvs) > 1:
        top = results.loc[results["split"] == "val"].groupby(list(gridsearch.keys()))["f1"].mean().reset_index().sort_values("f1", ascending=False)
        filt = None
        for k,v in gridsearch.items():
            if filt is None:
                filt = results[k] == top[k].iat[0]
            else:
                filt = (results[k] == top[k].iat[0]) & filt
        results = results.loc[filt]
        results = results.groupby("split")[["auc", "auprc", "f1", "accuracy"]].agg(["mean", "std"])
        hyperparams = "_".join([top[k].iat[0] for k in gridsearch.keys()])
        results.to_csv(f"{wd}/{tag}_{hyperparams}_results.csv")
    else:
        results = results.groupby("split")[["auc", "auprc", "f1", "accuracy"]].agg(["mean", "std"])
        results.to_csv(f"{wd}/{tag}_results.csv")

compile_results("pnet")
compile_results("dense")
compile_results("svc", {"c" : "model_params_choice"})