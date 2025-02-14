import os, sys
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score
from keras.activations import linear
from keras.losses import MeanSquaredError

sys.path.insert(0, os.getcwd())
from architecture.data_utils import *
from architecture.pnet_config import *
from architecture.pipeline import *

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
config["views"] = [("gexpr", f"cleveland_gene_expression.csv", selected_genes, 0, lambda x : x, lambda x : x)]
config["view_alignment_method"] = "drop samples"
config["labels"] = [("cleveland_auc_only.csv", 0)]
config["tv_split_seed"] = 42
config["inner_kfolds"] = 1
config["outer_kfolds"] = 10
config["validation_prop"] = 0.1
config["results_processors"] = [lambda x : save_results(x, save_supervised_result, {"r2" : r2_score,
                                                                                    "explained_variance" : explained_variance_score,
                                                                                    "mse" : mean_squared_error,
                                                                                    "mae" : mean_absolute_error}, 
                                                                          "individual"),
                            plot_history]
n_hidden_layers = 5
config["model_params"] = {
                            "pp_relations" : "architecture/Reactome/ReactomePathwaysRelation.txt",
                            "gp_relations" : "architecture/Reactome/ReactomePathways.gmt",
                            "n_hidden_layers" : n_hidden_layers,
                            "h_dropout" : [0.5] + [0.1] * n_hidden_layers,
                            "h_activation" : [tanh] * (n_hidden_layers + 1),
                            "o_activation" : [linear] * (n_hidden_layers + 1),
                            "h_reg" : [(L2, {"l2" : 1e-3})] * (n_hidden_layers + 1),
                            "o_reg" : [(L2, {"l2" : 1e-2})] * (n_hidden_layers + 1),
                            "h_kernel_initializer" : ["lecun_uniform"] * (n_hidden_layers + 1),
                            "h_kernel_constraints" : [None] * (n_hidden_layers + 1),
                            "h_bias_initializer" : ["lecun_uniform"] * (n_hidden_layers + 1),
                            "h_bias_constraints" : [None] * (n_hidden_layers + 1),
                            "batch_normal" : False,
                            "sparse" : True,
                            "dropout_testing" : False,
                            "loss" : [MeanSquaredError()] * (n_hidden_layers + 1),
                            "loss_weights" : [2, 7, 20, 54, 148, 400],
                            "optimizer" : Adam(1e-3)
                        }

config["fitting_params"] = {
                                "epochs" : 300,
                                "batch" : 50,
                                "LRScheduler" : LearningRateScheduler(step_decay_part, verbose=0),
                                "early_stopping" : None,
                                "prediction_output" : "average",
                                "shuffle_samples" : True,
                                "class_weight" : None
                            }

# Run pnet crossvalidation
pipeline = TFPipeline(config)
pipeline.run_crossvalidation()

# Run dense crossvalidation
config["model_params"]["sparse"] = False
config["run_id"] = "dense"
pipeline = TFPipeline(config)
pipeline.run_crossvalidation()

# Compile results
dense_results = [] 
for i, fold in enumerate([x for x in os.listdir(f"{run_dir}/dense") if os.path.isdir(f"{run_dir}/dense/{x}")]):
    result = pd.read_csv(f"{run_dir}/dense/{fold}/cv_0/summary_results.csv")
    result.columns = ["split"] + list(result.columns[1:])
    result["test_fold"] = i
    dense_results.append(result)
dense_results = pd.concat(dense_results)
dense_results = dense_results.groupby("split")[["auc_r2", "auc_explained_variance", "auc_mae", "auc_mse"]].agg(["mean", "std"])
dense_results.to_csv(f"{wd}/dense_results.csv")

pnet_results = []
for i, fold in enumerate([x for x in os.listdir(f"{run_dir}/pnet") if os.path.isdir(f"{run_dir}/pnet/{x}")]):
    result = pd.read_csv(f"{run_dir}/pnet/{fold}/cv_0/summary_results.csv")
    result.columns = ["split"] + list(result.columns[1:])
    result["test_fold"] = i
    pnet_results.append(result)
pnet_results = pd.concat(pnet_results)
pnet_results = pnet_results.groupby("split")[["auc_r2", "auc_explained_variance", "auc_mae", "auc_mse"]].agg(["mean", "std"])
pnet_results.to_csv(f"{wd}/pnet_results.csv")


