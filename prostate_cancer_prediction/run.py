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
from prostate_cancer_prediction.configs.pnetfc_nested_CV import pnetfc_nested_CV_config
from prostate_cancer_prediction.configs.pnet_network_order_fixed_seed import pnet_network_order_fixed_seed_configs
from prostate_cancer_prediction.configs.pnet_single_split import pnet_single_split_config
from prostate_cancer_prediction.configs.pnet_single_split_elmarakeby import pnet_single_split_elmarakeby_config
from prostate_cancer_prediction.configs.pnet_stratified_5_fold_CV import pnet_stratified_5_fold_CV_config
from prostate_cancer_prediction.configs.pnet_train_size_variation import pnet_train_size_variation_configs
from prostate_cancer_prediction.configs.pnetfc_stratified_5_fold_CV import pnetfc_stratified_5_fold_CV_config
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
from prostate_cancer_prediction.configs.pnet_external_validation_1 import \
    pnet_external_validation_1_elmarakeby_config
from prostate_cancer_prediction.configs.pnet_external_validation_2 import \
    pnet_external_validation_2_elmarakeby_config

from prostate_cancer_prediction.configs.pnet_network_order_variation import pnet_network_order_variation_configs
from prostate_cancer_prediction.configs.pnet_10_fold_CV_stability import pnet_10_fold_CV_stability_config
from prostate_cancer_prediction.configs.pnet_GO_10_fold_CV_stability import pnet_GO_10_fold_CV_stability_config
from prostate_cancer_prediction.plotting.plot_train_size_variations import plot_train_size_comparisons

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from prostate_cancer_prediction.configs.base_config import download_dir, run_dir, wd, selected_genes
from prostate_cancer_prediction.configs.pnet_single_split import n_hidden_layers
from architecture.train import train
from prostate_cancer_prediction.plotting.plot_nested_cv import plot_nested_CV
from prostate_cancer_prediction.plotting.plot_stratified_5_fold_cv import plot_stratified_5_fold_CV
from prostate_cancer_prediction.plotting.plot_external_validation import plot_external_validation
from prostate_cancer_prediction.plotting.plot_single_split import plot_single_split_curves
from prostate_cancer_prediction.plotting.plot_sankey import plot_sankey
from prostate_cancer_prediction.plotting.plot_importance_stability import analyse_importance_stability
from prostate_cancer_prediction.plotting.plot_network_order_variation import plot_network_order_variation
def run():
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
        #todo check pnetfc hyperparams
        # pnet_stratified_5_fold_CV_config,
        # pnetfc_stratified_5_fold_CV_config,
        # decision_tree_stratified_5_fold_CV_config,
        # adaboost_stratified_5_fold_CV_config,
        # linear_svm_stratified_5_fold_CV_config,
        # random_forest_stratified_5_fold_CV_config,
        # rbf_svm_stratified_5_fold_CV_config,
        # sgd_logistic_regression_stratified_5_fold_CV_config

        # Feature-importance stability: 10-fold CV, elmarakeby hyperparameters
        #pnet_10_fold_CV_stability_config,
        # Feature-importance stability: 10-fold CV, Our hyperparameters we found in nested CV
        #pnet_GO_10_fold_CV_stability_config,

        #*pnet_network_order_variation_configs,
        #*pnet_network_order_fixed_seed_configs,

        # Stratified nested CV
        # pnet_nested_CV_config,
        # pnetfc_nested_CV_config,
        # pnet_GO_nested_CV_config,
        # adaboost_nested_CV_config,
         decision_tree_nested_CV_config,
        # linear_svm_nested_CV_config,
        # random_forest_nested_CV_config,
        # rbf_svm_nested_CV_config,
        # sgd_logistic_regression_nested_CV_config,

         # pnet_external_validation_1_elmarakeby_config,
         # pnet_external_validation_2_elmarakeby_config
    ]

    for config in configs:
        train(config)

    figures_dir = os.path.join(wd, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    #plot_train_size_comparisons(run_dir, figures_dir)

    # Distribution of each test metric across the network-order variation runs
    # plot_network_order_variation(run_dir, figures_dir,
    #                              run_prefix="pnet_network_order_variation", split="test")

    #plot_nested_CV(run_dir, figures_dir)
    # plot_stratified_5_fold_CV(run_dir, figures_dir)
    #plot_external_validation(run_dir, figures_dir)

    # Feature-importance stability (each run writes to its own subdirectory under
    # figures/importance_stability/{run_id} so the two do not overwrite each other)
    # analyse_importance_stability(run_dir, figures_dir, n_hidden_layers,
    #                              run_id="pnet_10_fold_CV_stability",
    #                              pathway_names="architecture/Reactome/ReactomePathways.txt")
    # analyse_importance_stability(run_dir, figures_dir, n_hidden_layers,
    #                              run_id="pnet_GO_10_fold_CV_stability",
    #                              pathway_names="architecture/GO/go_id_name_map.tsv")

    # Single-split ROC/PRC curves (original hyperparams with test+val combined,
    # and our hyperparams with the splits kept separate):
    # models_elmarakeby = ["pnet_single_split_elmarakeby", "decision_tree_single_split_elmarakeby"]
    # models = ["pnet_single_split", "decision_tree_single_split"]
    # plot_single_split_curves(run_dir, figures_dir, models_elmarakeby, tag="elmarakeby", concat_val=True)
    # plot_single_split_curves(run_dir, figures_dir, models, concat_val=False)

    # Sankey diagram of the P-NET-GO hierarchy:
    # pnet_run_dir = f"{run_dir}/pnet_GO_single_split"
    # dataset_id_mappings = "architecture/GO/go_id_name_map.tsv"
    # plot_sankey(pnet_run_dir, n_hidden_layers, figures_dir, dataset_id_mappings)

if __name__ == "__main__":
    run()