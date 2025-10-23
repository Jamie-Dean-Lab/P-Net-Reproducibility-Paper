import os, sys, random, json
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, f1_score
from scipy.stats import ttest_ind

sys.path.insert(0, os.getcwd())
from architecture.data_utils import *
from architecture.pnet_config import *
from architecture.pipeline import *
from architecture.evaluation import *
from architecture.pnet_model import PNetArchitectureGenerator, get_layer_maps
from preprocess import *
from dense_config import *
from pnet_auprc import PlotAUPRC
from figure_pnet_vs_dense import ComparativeAnalysis
from sankey import SankeyDiagram

# Download data if not done so already and set up run directory
wd = "Prostate Cancer Prediction"
download_dir = f"{wd}/data"
data_dir = f"{download_dir}/_database"
run_dir = f"{wd}/runs"

if not os.path.exists(download_dir):
    with open("Prostate Cancer Prediction/src/download_data.py") as file:
        exec(file.read())

if not os.path.exists(run_dir):
    os.mkdir(run_dir)

data_dir = "Prostate Cancer Prediction/data/_database"
# Identify protein coding only genes and selected gene list based on P-Net paper
selected_genes = set(pd.read_csv(f"{data_dir}/genes/tcga_prostate_expressed_genes_and_cancer_genes.csv")["genes"])
hugo_genes = set(pd.read_csv(f"{data_dir}/genes/HUGO_genes/protein-coding_gene_with_coordinate_minimal.txt", sep="\t", header=None).iloc[:, 3].unique())
selected_genes = list(selected_genes.intersection(hugo_genes))

# prepare config
config["data_dir"] = f"{data_dir}/prostate/processed"
config["run_dir"] = run_dir
config["run_id"] = "pnet_specific_train_split"
config["views"] = [("mut_important", f"P1000_final_analysis_set_cross_important_only.csv", 
                         selected_genes, 0, mut_binary, lambda x : x),
                         ("cnv_amp", f"P1000_data_CNA_paper.csv", selected_genes, 0, cnv_amp, lambda x : x),
                         ("cnv_del", f"P1000_data_CNA_paper.csv", selected_genes, 0, cnv_del, lambda x : x)]

config["view_alignment_method"] = "drop samples"
config["labels"] = [("response_paper.csv", 0)]
config["train_samples"] = pd.read_csv(f"{data_dir}/prostate/splits/training_set.csv")["id"].to_list()
config["val_samples"] = pd.read_csv(f"{data_dir}/prostate/splits/validation_set.csv")["id"].to_list()
config["test_samples"] = pd.read_csv(f"{data_dir}/prostate/splits/test_set.csv")["id"].to_list()
config["results_processors"] = [lambda x : save_results(x, save_supervised_result, {"auc" : roc_auc_score,
                                                                          "auprc" : average_precision_score,
                                                                          "f1" : lambda ys, preds : f1_score(ys, (preds > 0.5).astype(int)),
                                                                          "accuracy" : lambda ys, preds : accuracy_score(ys, (preds > 0.5).astype(int))}, 
                                                                          "individual"),
                            plot_history, get_deeplift_global]
config["use_validation_on_test"] = True
f1_selection = lambda x : f1_score(x["val_df"].ys, (x["val_preds"] >= 0.5).astype(int))
auprc_selection = lambda x : average_precision_score(x["val_df"].ys, x["val_preds"])
auc_selection = lambda x : roc_auc_score(x["val_df"].ys, x["val_preds"])
config["val_metric"] = {"f1" : f1_selection, "auprc" : auprc_selection, "auc" : auc_selection}
config["rng_seed"] = 20080808
config["tt_split_seed"] = 20080808
config["tv_split_seed"] = 20080808

# Run PNet on specific training, validation, and test
pipeline = TFPipeline(config)
pipeline.run_single_split()

# Run gridsearch over params for the different ML pipelines

# Decision Tree
ml_config = copy.copy(config)
ml_config["task"] = "binary classification"
ml_config["model"] = DecisionTreeClassifier
gs_params = {"model_params" : {f"ssplit_{s}_depth_{d}" : {"min_samples_split" : s, "max_depth" : d,
                                                         "class_weight" : {0 : 0.75, 1 : 1.5}
                                                         } for s in range(10, 500, 20)
                                                         for d in range(1, 20, 2)}}
ml_config["grid_search"] = construct_gs_params(gs_params)
ml_config["run_id"] = "decision_tree"
ml_config["results_processors"] = ml_config["results_processors"][:-2]
pipeline = MLPipeline(ml_config)
pipeline.run_single_split()

ml_config["run_id"] = "decision_tree_elmarakeby"
gs_params = {"model_params" : {f"ssplit_{s}_depth_{d}" : {"min_samples_split" : s, "max_depth" : d,
                                                         "class_weight" : {0 : 0.75, 1 : 1.5}
                                                         } for s in [10]
                                                         for d in [10]}}
ml_config["val_metric"] = {"f1" : f1_selection}
ml_config["grid_search"] = construct_gs_params(gs_params)
pipeline = MLPipeline(ml_config)
pipeline.run_single_split()

# Linear SVC
ml_config["model"] = SVC
gs_params = {"model_params" : {f"c_{c}" : {"kernel" : "linear", "probability" : True,
                                           "C" : c, "class_weight" : {0 : 0.75, 1 : 1.5}}
                                           for c in [0.001, 0.01, 0.1, 1, 10, 100, 1000]}}
ml_config["val_metric"] = {"f1" : f1_selection, "auprc" : auprc_selection, "auc" : auc_selection}
ml_config["grid_search"] = construct_gs_params(gs_params)
ml_config["run_id"] = "linear_svm"
pipeline = MLPipeline(ml_config)
pipeline.run_single_split()
gs_params = {"model_params" : {f"c_{c}" : {"kernel" : "linear", "probability" : True,
                                           "C" : c, "class_weight" : {0 : 0.75, 1 : 1.5}}
                                           for c in [0.1]}}
ml_config["val_metric"] = {"f1" : f1_selection}
ml_config["grid_search"] = construct_gs_params(gs_params)
ml_config["run_id"] = "linear_svm_elmarakeby"
pipeline = MLPipeline(ml_config)
pipeline.run_single_split()

# RBF SVC
ml_config["model"] = SVC
gs_params = {"model_params" : {f"c_{c}_g_{g}" : {"kernel" : "rbf", "probability" : True,
                                           "C" : c, "class_weight" : {0 : 0.75, 1 : 1.5},
                                           "gamma" : g}
                                           for c in [0.001, 0.01, 0.1, 1, 10, 100, 1000]
                                           for g in [0.001, 0.01, 0.1, 1]}}
ml_config["grid_search"] = construct_gs_params(gs_params)
ml_config["val_metric"] = {"f1" : f1_selection, "auprc" : auprc_selection, "auc" : auc_selection}
ml_config["run_id"] = "rbf_svm"
pipeline = MLPipeline(ml_config)
pipeline.run_single_split()
gs_params = {"model_params" : {f"c_{c}_g_{g}" : {"kernel" : "rbf", "probability" : True,
                                           "C" : c, "class_weight" : {0 : 0.75, 1 : 1.5},
                                           "gamma" : g}
                                           for c in [100]
                                           for g in [0.001]}}
ml_config["grid_search"] = construct_gs_params(gs_params)
ml_config["val_metric"] = {"f1" : f1_selection}
ml_config["run_id"] = "rbf_svm_elmarakeby"
pipeline = MLPipeline(ml_config)
pipeline.run_single_split()

# Random Forest
ml_config["model"] = RandomForestClassifier
gs_params = {"model_params" : {f"bootstrap_{b}_depth_{d}_estimators_{n}" : 
                               {"bootstrap" : b, "max_depth" : d, "n_estimators" : n, "class_weight" : {0 : 0.75, 1 : 1.5}}
                               for b in [True, False] for d in [10, 30, 50, 70, None] for n in [10, 50, 100, 200]}}
ml_config["grid_search"] = construct_gs_params(gs_params)
ml_config["val_metric"] = {"f1" : f1_selection, "auprc" : auprc_selection, "auc" : auc_selection}
ml_config["run_id"] = "random_forest"
pipeline = MLPipeline(ml_config)
pipeline.run_single_split()
gs_params = {"model_params" : {f"bootstrap_{b}_depth_{d}_estimators_{n}" : 
                               {"bootstrap" : b, "max_depth" : d, "n_estimators" : n, "class_weight" : {0 : 0.75, 1 : 1.5}}
                               for b in [False] for d in [None] for n in [50]}}
ml_config["grid_search"] = construct_gs_params(gs_params)
ml_config["val_metric"] = {"f1" : f1_selection}
ml_config["run_id"] = "random_forest_elmarakeby"
pipeline = MLPipeline(ml_config)
pipeline.run_single_split()

# Adaboost
ml_config["model"] = AdaBoostClassifier
gs_params = {"model_params" : {f"lr_{l}_estimators_{n}" : 
                               {"learning_rate" : l, "n_estimators" : n}
                               for l in [0.01, 0.05, 0.1, 0.3, 1] for n in [50, 100]}}
ml_config["grid_search"] = construct_gs_params(gs_params)
ml_config["val_metric"] = {"f1" : f1_selection, "auprc" : auprc_selection, "auc" : auc_selection}
ml_config["run_id"] = "adaboost"
pipeline = MLPipeline(ml_config)
pipeline.run_single_split()
gs_params = {"model_params" : {f"lr_{l}_estimators_{n}" : 
                               {"learning_rate" : l, "n_estimators" : n}
                               for l in [0.1] for n in [50]}}
ml_config["grid_search"] = construct_gs_params(gs_params)
ml_config["run_id"] = "adaboost_elmarakeby"
ml_config["val_metric"] = {"f1" : f1_selection}
pipeline = MLPipeline(ml_config)
pipeline.run_single_split()

# Logistic Regression
ml_config["model"] = SGDClassifier
gs_params = {"model_params" : {f"alpha_{a}_penalty_{p}" : 
                               {"alpha" : a, "penalty" : p, "class_weight" : {0 : 0.75, 1 : 1.5},
                                "loss" : "log_loss"}
                               for a in [0.0001, 0.001, .009, 0.01, .09, 1, 5, 10]
                               for p in ["l1", "l2"]}}
ml_config["grid_search"] = construct_gs_params(gs_params)
ml_config["val_metric"] = {"f1" : f1_selection, "auprc" : auprc_selection, "auc" : auc_selection}
ml_config["run_id"] = "sgd_logistic_regression"
pipeline = MLPipeline(ml_config)
pipeline.run_single_split()
gs_params = {"model_params" : {f"alpha_{a}_penalty_{p}" : 
                               {"alpha" : a, "penalty" : p, "class_weight" : {0 : 0.75, 1 : 1.5},
                                "loss" : "log_loss"}
                               for a in [0.01]
                               for p in ["l2"]}}
ml_config["grid_search"] = construct_gs_params(gs_params)
ml_config["val_metric"] = {"f1" : f1_selection}
ml_config["run_id"] = "sgd_logistic_regression_elmarakeby"
pipeline = MLPipeline(ml_config)
pipeline.run_single_split()

# Create 5 different splits of data to get crossvalidated estimate of test performance on different train sizes

config["results_processors"] = config["results_processors"][:-1]
config["stratified"] = True
config["inner_kfolds"] = 5
config["val_samples"] = pd.read_csv(f"{data_dir}/prostate/splits/validation_set.csv")["id"].to_list()
config["test_samples"] = pd.read_csv(f"{data_dir}/prostate/splits/test_set.csv")["id"].to_list()
config["tv_split_seed"] = 20080808
config["fold_collators"].append(collate_folds)
train_samples = [pd.read_csv(f"{data_dir}/prostate/splits/training_set_{s}.csv")["id"].to_list() + config["val_samples"] for s in range(0, 20, 3)]
config["use_validation_on_test"] = False
config["val_metric"] = {}
config["grid_search_collators"] = []

for i, ts in enumerate(train_samples):
    config["train_samples"] = ts
    config["run_id"] = f"pnet_train_size_variation_{i}"
    pipeline = TFPipeline(config)
    pipeline.run_crossvalidation()

# Do the same for fully connected pnet
config["model_params"]["sparse"] = False

for i, ts in enumerate(train_samples):
    config["train_samples"] = ts
    config["run_id"] = f"pnetfc_train_size_variation_{i}"
    pipeline = TFPipeline(config)
    pipeline.run_crossvalidation()

# Do the same for dense net
dense_config["data_dir"] = f"{data_dir}/prostate/processed"
dense_config["run_dir"] = run_dir
dense_config["views"] = [("mut_important", f"P1000_final_analysis_set_cross_important_only.csv", 
                         selected_genes, 0, mut_binary, lambda x : x),
                         ("cnv_amp", f"P1000_data_CNA_paper.csv", selected_genes, 0, cnv_amp, lambda x : x),
                         ("cnv_del", f"P1000_data_CNA_paper.csv", selected_genes, 0, cnv_del, lambda x : x)]

dense_config["view_alignment_method"] = "drop samples"
dense_config["labels"] = [("response_paper.csv", 0)]
dense_config["results_processors"] = [lambda x : save_results(x, save_supervised_result, {"auc" : roc_auc_score,
                                                                          "auprc" : average_precision_score,
                                                                          "f1" : lambda ys, preds : f1_score(ys, (preds > 0.5).astype(int)),
                                                                          "accuracy" : lambda ys, preds : accuracy_score(ys, (preds > 0.5).astype(int))}, 
                                                                          "individual")]
dense_config["stratified"] = True
dense_config["inner_kfolds"] = 5
dense_config["tv_split_seed"] = 20080808
dense_config["fold_collators"].append(collate_folds)
dense_config["val_samples"] = pd.read_csv(f"{data_dir}/prostate/splits/validation_set.csv")["id"].to_list()
dense_config["test_samples"] = pd.read_csv(f"{data_dir}/prostate/splits/test_set.csv")["id"].to_list()
dense_config["val_metric"] = {}
dense_config["grid_search_collators"] = []
dense_config["use_validation_on_test"] = False

for i, ts in enumerate(train_samples):
    dense_config["train_samples"] = ts
    dense_config["run_id"] = f"dense_train_size_variation_{i}"
    pipeline = TFPipeline(dense_config)
    pipeline.run_crossvalidation()

# Plot results

fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(7, 14))
results = {}
tabular = []
for model in [x for x in os.listdir(run_dir) if x.find("elmarakeby") > -1 or x.find("specific_train_split") > -1]:
    if model.find("pnet") == -1:
        results[model] = pd.read_csv(f"{run_dir}/{model}/best_f1/test_results.csv", index_col=0)
        summary = pd.read_csv(f"{run_dir}/{model}/best_f1/summary_results.csv")
        summary["model"] = model
        summary.columns = ["split"] + summary.columns[1:].to_list()
        tabular.append(summary)
    else:
        results["pnet"] = pd.read_csv(f"{run_dir}/{model}/test_results.csv", index_col=0)
        summary = pd.read_csv(f"{run_dir}/{model}/summary_results.csv")
        summary["model"] = model
        summary.columns = ["split"] + summary.columns[1:].to_list()
        tabular.append(summary)

auprc = PlotAUPRC(results)
auprc.plot(ax[0], "A")
tabular = pd.concat(tabular)
tabular.to_csv(f"{wd}/specific_split_results.csv")

pnet_results = []
for exp_dir in [x for x in os.listdir(run_dir) if x.find("pnet_train_size_variation") > -1]:
    pnet_dir = f"{run_dir}/{exp_dir}"
    data = pd.read_csv(f"{run_dir}/{exp_dir}/test_0/cv_0/fold_summaries.csv", index_col=0)
    n_samples = pd.read_csv(f"{run_dir}/{exp_dir}/test_0/cv_0/fold_0/train_results.csv").shape[0] + pd.read_csv(f"{run_dir}/{exp_dir}/test_0/cv_0/fold_0/val_results.csv").shape[0]
    data = data.loc[data["split"] == "val", ["response_auc", "fold"]]
    data["n_samples"] = n_samples
    pnet_results.append(data)
pnet_results = pd.concat(pnet_results)

pnetfc_results = []
for exp_dir in [x for x in os.listdir(run_dir) if x.find("pnetfc_train_size_variation") > -1]:
    pnet_dir = f"{run_dir}/{exp_dir}"
    data = pd.read_csv(f"{run_dir}/{exp_dir}/test_0/cv_0/fold_summaries.csv", index_col=0)
    n_samples = pd.read_csv(f"{run_dir}/{exp_dir}/test_0/cv_0/fold_0/train_results.csv").shape[0] + pd.read_csv(f"{run_dir}/{exp_dir}/test_0/cv_0/fold_0/val_results.csv").shape[0]
    data = data.loc[data["split"] == "val", ["response_auc", "fold"]]
    data["n_samples"] = n_samples
    pnetfc_results.append(data)
pnetfc_results = pd.concat(pnetfc_results)

dense_results = []
for exp_dir in [x for x in os.listdir(run_dir) if x.find("dense_train_size_variation") > -1]:
    pnet_dir = f"{run_dir}/{exp_dir}"
    data = pd.read_csv(f"{run_dir}/{exp_dir}/test_0/cv_0/fold_summaries.csv", index_col=0)
    n_samples = pd.read_csv(f"{run_dir}/{exp_dir}/test_0/cv_0/fold_0/train_results.csv").shape[0] + pd.read_csv(f"{run_dir}/{exp_dir}/test_0/cv_0/fold_0/val_results.csv").shape[0]
    data = data.loc[data["split"] == "val", ["response_auc", "fold"]]
    data["n_samples"] = n_samples
    dense_results.append(data)
dense_results = pd.concat(dense_results)

pnet_dense_stats = [ttest_ind(pnet_results.loc[pnet_results["n_samples"] == n, "response_auc"].to_numpy(), 
                              dense_results.loc[dense_results["n_samples"] == n, "response_auc"].to_numpy()).pvalue < 0.05 
                              for n in pnet_results["n_samples"].unique()]
pnet_pnetfc_stats = [ttest_ind(pnet_results.loc[pnet_results["n_samples"] == n, "response_auc"].to_numpy(), 
                              pnetfc_results.loc[pnetfc_results["n_samples"] == n, "response_auc"].to_numpy()).pvalue < 0.05 
                              for n in pnet_results["n_samples"].unique()]
pnet_results = pnet_results.groupby("n_samples")["response_auc"].agg(["mean", "std"]).reset_index()
pnetfc_results = pnetfc_results.groupby("n_samples")["response_auc"].agg(["mean", "std"]).reset_index()
dense_results = dense_results.groupby("n_samples")["response_auc"].agg(["mean", "std"]).reset_index()
pnet_results = pnet_results.iloc[range(1, pnet_results.shape[0], 2)]
pnetfc_results = pnetfc_results.iloc[range(1, pnetfc_results.shape[0], 2)]
dense_results = dense_results.iloc[range(1, dense_results.shape[0], 2)]
pnet_dense_stats = [pnet_dense_stats[i] for i in range(1, len(pnet_dense_stats), 2)]
pnet_pnetfc_stats = [pnet_pnetfc_stats[i] for i in range(1, len(pnet_pnetfc_stats), 2)]
results = {"number_of_samples" : pnet_results["n_samples"], "pnet_auc" : pnet_results["mean"],
           "pnet_lower_bound" : pnet_results["mean"] - pnet_results["std"],
           "pnet_upper_bound" : pnet_results["mean"] + pnet_results["std"],
           "dense_auc" : dense_results["mean"],
           "dense_lower_bound" : dense_results["mean"] - dense_results["std"],
           "dense_upper_bound" : dense_results["mean"] + dense_results["std"],
           "statistically_significant" : np.array(pnet_dense_stats)}

compare = ComparativeAnalysis(results)
compare.plot(ax[1], "B", dense_label="Dense Single Layer")

results = {"number_of_samples" : pnet_results["n_samples"], "pnet_auc" : pnet_results["mean"],
           "pnet_lower_bound" : pnet_results["mean"] - pnet_results["std"],
           "pnet_upper_bound" : pnet_results["mean"] + pnet_results["std"],
           "dense_auc" : pnetfc_results["mean"],
           "dense_lower_bound" : pnetfc_results["mean"] - pnetfc_results["std"],
           "dense_upper_bound" : pnetfc_results["mean"] + pnetfc_results["std"],
           "statistically_significant" : np.array(pnet_pnetfc_stats)}

compare = ComparativeAnalysis(results)
compare.plot(ax[2], "C", dense_label="P-NET fully connected")
plt.tight_layout()
plt.savefig(f"{wd}/figure_1.jpg")
plt.close()

# Sankey plot
layer_order = ["inputs"] + [f"h{i}" for i in range(1, 6)]
node_values = {x.split("_")[-1].split(".")[0] : pd.read_csv(f"{wd}/runs/pnet_specific_train_split/{x}", index_col=0)
               for x in os.listdir(f"{wd}/runs/pnet_specific_train_split") if x.find("feature_importance") > -1}
reactome = PNetArchitectureGenerator()
netx = reactome.get_reactome_networkx("architecture/Reactome/ReactomePathwaysRelation.txt")
maps = reactome.get_layers(netx, n_hidden_layers, "architecture/Reactome/ReactomePathways.gmt", selected_genes)

deeplift = {}

for fn in [x for x in os.listdir(f"{wd}/runs/pnet_specific_train_split") if x.find("feature_importance") > -1]:
    name = fn.split("_")[-1].replace(".csv", "")
    deeplift[name] = pd.read_csv(f"{wd}/runs/pnet_specific_train_split/{fn}", index_col=0)

maps = get_layer_maps(deeplift["h0"].index, maps, False)
pathwaynames = pd.read_csv("architecture/Reactome/ReactomePathways.txt", sep="\t", index_col=0, header=None)
pathwaynames.columns = ["name", "species"]

layers = {}
weights = {}
for i in range(len(maps)):
    if i > 0:
        nodes = pathwaynames.loc[maps[i].index, "name"].to_numpy()
    else:
        nodes = maps[i].index.to_numpy()
    layers[f"layer_{i+1}"] = nodes
    weights[f"layer_{i+1}"] = maps[i].to_numpy() * deeplift[f"h{i}"].to_numpy()

diagram = SankeyDiagram(layers, weights)
diagram.plot([10, 10, 10, 10, 10, 6], wd)




