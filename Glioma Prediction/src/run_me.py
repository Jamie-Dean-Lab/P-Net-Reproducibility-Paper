import os, sys
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, f1_score
from functools import partial
from keras.activations import tanh, relu, leaky_relu

sys.path.insert(0, os.getcwd())
from architecture.data_utils import *
from architecture.pnet_config import *
from architecture.pipeline import *
from architecture.evaluation import *
from preprocess import *
from architecture.callbacks_custom import step_decay


# Download data if not done so already and set up run directory
wd = "Glioma Prediction"
download_dir = f"{wd}/data"
run_dir = f"{wd}/runs"

if not os.path.exists(download_dir):
    with open(f"{wd}/src/download_data.py") as file:
        exec(file.read())
    with open(f"{wd}/src/preprocess.py") as file:
        exec(file.read())

if not os.path.exists(run_dir):
    os.mkdir(run_dir)

# Identify protein coding only genes and selected gene list based on P-Net paper
selected_genes = list(set(pd.read_csv(f"{download_dir}/hugo_genes.txt", sep="\t")["symbol"]))

# prepare config
config["data_dir"] = download_dir
config["run_dir"] = run_dir
config["run_id"] = "pnet"
config["views"] = [("mut", f"mutations.csv", selected_genes, 0, lambda x : x, lambda x : x)]

config["view_alignment_method"] = "drop samples"
config["labels"] = [("response.csv", 0)]
config["results_processors"] = [lambda x : save_results(x, save_supervised_result, {"auc" : roc_auc_score,
                                                                          "auprc" : average_precision_score,
                                                                          "f1" : lambda ys, preds : f1_score(ys, (preds > 0.5).astype(int)),
                                                                          "accuracy" : lambda ys, preds : accuracy_score(ys, (preds > 0.5).astype(int))}, 
                                                                          "individual"),
                            plot_history]
config["tv_split_seed"] = 42
config["inner_kfolds"] = 1
config["outer_kfolds"] = 5
config["validation_prop"] = 0.1

config["model_params"] = {
                            "pp_relations" : "architecture/Reactome/ReactomePathwaysRelation.txt",
                            "gp_relations" : "architecture/Reactome/ReactomePathways.gmt",
                            "n_hidden_layers" : n_hidden_layers,
                            "h_dropout" : [0.6] * (n_hidden_layers + 1),
                            "h_activation" : [relu, tanh, leaky_relu, tanh, leaky_relu, tanh],
                            "o_activation" : [relu] * (n_hidden_layers + 1),
                            "h_reg" : [(L2, {"l2" : 1})] * (n_hidden_layers + 1),
                            "o_reg" : [(L2, {"l2" : 1e-1})] * (n_hidden_layers + 1),
                            "h_kernel_initializer" : ["lecun_uniform"] * (n_hidden_layers + 1),
                            "h_kernel_constraints" : [None] * (n_hidden_layers + 1),
                            "h_bias_initializer" : ["lecun_uniform"] * (n_hidden_layers + 1),
                            "h_bias_constraints" : [None] * (n_hidden_layers + 1),
                            "batch_normal" : False,
                            "sparse" : True,
                            "dropout_testing" : False,
                            "loss" : [BinaryCrossentropy()] * (n_hidden_layers + 1),
                            "loss_weights" : [2, 7, 20, 54, 148, 400],
                            "optimizer" : Adam(1e-3)
                        }

config["fitting_params"] = {
                                "epochs" : 200,
                                "batch" : 100,
                                "LRScheduler" : None,
                                "early_stopping" : None,
                                "prediction_output" : "average",
                                "shuffle_samples" : True,
                                "class_weight" : None
                            }

# Run PNet
pipeline = TFPipeline(config)
pipeline.run_crossvalidation()

# Run Dense
config["run_id"] = "dense"
config["model_params"]["sparse"] = False
pipeline = TFPipeline(config)
pipeline.run_crossvalidation()
