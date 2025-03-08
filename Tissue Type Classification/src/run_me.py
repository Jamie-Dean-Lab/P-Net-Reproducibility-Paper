import os, sys
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
pipeline = MLPipeline(config)
pipeline.run_crossvalidation()