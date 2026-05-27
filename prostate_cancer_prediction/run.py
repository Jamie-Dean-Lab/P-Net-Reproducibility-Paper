import os, sys

from prostate_cancer_prediction.configs.adaboost_nested_CV import adaboost_nested_CV_config
from prostate_cancer_prediction.configs.adaboost_single_split import adaboost_single_split_config
from prostate_cancer_prediction.configs.adaboost_single_split_elmarakeby import adaboost_single_split_elmarakeby_config
from prostate_cancer_prediction.configs.adaboost_stratified_5_fold_CV import adaboost_stratified_5_fold_CV_config
from prostate_cancer_prediction.configs.decision_tree_nested_CV import decision_tree_nested_CV_config
from prostate_cancer_prediction.configs.decision_tree_single_split import decision_tree_single_split_config
from prostate_cancer_prediction.configs.decision_tree_single_split_elmarakeby import \
    decision_tree_single_split_elmarakeby_config
from prostate_cancer_prediction.configs.decision_tree_stratified_5_fold_CV import \
    decision_tree_stratified_5_fold_CV_config
from prostate_cancer_prediction.configs.dense_single_layer_train_size_variation import \
    dense_single_layer_train_size_variation_configs
from prostate_cancer_prediction.configs.linear_svm_nested_CV import linear_svm_nested_CV_config
from prostate_cancer_prediction.configs.linear_svm_single_split import linear_svm_single_split_config
from prostate_cancer_prediction.configs.linear_svm_single_split_elmarakeby import \
    linear_svm_single_split_elmarakeby_config
from prostate_cancer_prediction.configs.linear_svm_stratified_5_fold_CV import linear_svm_stratified_5_fold_CV_config
from prostate_cancer_prediction.configs.pnet_GO_nested_CV import pnet_GO_nested_CV_config
from prostate_cancer_prediction.configs.pnet_GO_single_split import pnet_GO_single_split_config
from prostate_cancer_prediction.configs.pnet_nested_CV import pnet_nested_CV_config
from prostate_cancer_prediction.configs.pnet_single_split import pnet_single_split_config
from prostate_cancer_prediction.configs.pnet_single_split_elmarakeby import pnet_single_split_elmarakeby_config
from prostate_cancer_prediction.configs.pnet_stratified_5_fold_CV import pnet_stratified_5_fold_CV_config
from prostate_cancer_prediction.configs.pnet_train_size_variation import pnet_train_size_variation_configs
from prostate_cancer_prediction.configs.pnetfc_train_size_variation import pnetfc_train_size_variation_configs
from prostate_cancer_prediction.configs.random_forest_nested_CV import random_forest_nested_CV_config
from prostate_cancer_prediction.configs.random_forest_single_split import random_forest_single_split_config
from prostate_cancer_prediction.configs.random_forest_single_split_elmarakeby import \
    random_forest_single_split_elmarakeby_config
from prostate_cancer_prediction.configs.random_forest_stratified_5_fold_CV import \
    random_forest_stratified_5_fold_CV_config
from prostate_cancer_prediction.configs.rbf_svm_nested_CV import rbf_svm_nested_CV_config
from prostate_cancer_prediction.configs.rbf_svm_single_split import rbf_svm_single_split_config
from prostate_cancer_prediction.configs.rbf_svm_single_split_elmarakeby import rbf_svm_single_split_elmarakeby_config
from prostate_cancer_prediction.configs.rbf_svm_stratified_5_fold_CV import rbf_svm_stratified_5_fold_CV_config
from prostate_cancer_prediction.configs.sgd_logistic_regression_nested_CV import \
    sgd_logistic_regression_nested_CV_config
from prostate_cancer_prediction.configs.sgd_logistic_regression_single_split import \
    sgd_logistic_regression_single_split_config
from prostate_cancer_prediction.configs.sgd_logistic_regression_single_split_elmarakeby import \
    sgd_logistic_regression_single_split_elmarakeby_config
from prostate_cancer_prediction.configs.sgd_logistic_regression_stratified_5_fold_CV import \
    sgd_logistic_regression_stratified_5_fold_CV_config

from prostate_cancer_prediction.configs.pnet_network_order_variation import pnet_network_order_variation_configs

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from prostate_cancer_prediction.configs.base_config import download_dir, run_dir, wd, selected_genes
from prostate_cancer_prediction.configs.pnet_single_split import n_hidden_layers
from architecture.train import train
from prostate_cancer_prediction.plot import plot

def run():
    if not os.path.exists(download_dir):
        with open("prostate_cancer_prediction/download_data.py") as file:
            exec(file.read())

    if not os.path.exists(run_dir):
        os.mkdir(run_dir)

    configs = [
        # Single split
        # pnet_single_split_config,
        # decision_tree_single_split_config,
        # linear_svm_single_split_config,
        # rbf_svm_single_split_config,
        # random_forest_single_split_config,
        # adaboost_single_split_config,
        # sgd_logistic_regression_single_split_config,
        # pnet_GO_single_split_config

        # Single split, using hyperparams from original paper.
        # pnet_single_split_elmarakeby_config,
        # decision_tree_single_split_elmarakeby_config,
        # linear_svm_single_split_elmarakeby_config,
        # rbf_svm_single_split_elmarakeby_config,
        # random_forest_single_split_elmarakeby_config,
        # adaboost_single_split_elmarakeby_config,
        # sgd_logistic_regression_single_split_elmarakeby_config,

        # Train size variation
        # *pnet_train_size_variation_configs,
        # *pnetfc_train_size_variation_configs,
        # *dense_single_layer_train_size_variation_configs,

        # Stratified 5 fold CV
        # pnet_stratified_5_fold_CV_config,
        # decision_tree_stratified_5_fold_CV_config,
        # adaboost_stratified_5_fold_CV_config,
        # linear_svm_stratified_5_fold_CV_config,
        # random_forest_stratified_5_fold_CV_config,
        # rbf_svm_stratified_5_fold_CV_config,
        # sgd_logistic_regression_stratified_5_fold_CV_config

        # Stratified nested CV
        # pnet_nested_CV_config,
        # pnet_GO_nested_CV_config,
        # adaboost_nested_CV_config,
        # decision_tree_nested_CV_config,
        # linear_svm_nested_CV_config,
         random_forest_nested_CV_config,
        # rbf_svm_nested_CV_config,
        # sgd_logistic_regression_nested_CV_config,



    ]

    for config in configs:
        train(config)

    plot(wd, run_dir, selected_genes, n_hidden_layers)

if __name__ == "__main__":
    run()