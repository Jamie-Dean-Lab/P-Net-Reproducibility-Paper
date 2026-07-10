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
from prostate_cancer_prediction.configs.dense_single_layer_single_split import \
    dense_single_layer_single_split_config
from prostate_cancer_prediction.configs.dense_single_layer_nested_CV import \
    dense_single_layer_nested_CV_config
from prostate_cancer_prediction.configs.dense_single_layer_stratified_5_fold_CV import \
    dense_single_layer_stratified_5_fold_CV_config
from prostate_cancer_prediction.configs.dense_single_layer_single_split_elmarakeby import \
    dense_single_layer_single_split_elmarakeby_config
from prostate_cancer_prediction.configs.linear_svm_nested_CV import linear_svm_nested_CV_config
from prostate_cancer_prediction.configs.linear_svm_single_split import linear_svm_single_split_config
from prostate_cancer_prediction.configs.linear_svm_single_split_elmarakeby import \
    linear_svm_single_split_elmarakeby_config
from prostate_cancer_prediction.configs.linear_svm_stratified_5_fold_CV import linear_svm_stratified_5_fold_CV_config
from prostate_cancer_prediction.configs.pnet_GO_nested_CV import pnet_GO_nested_CV_config
from prostate_cancer_prediction.configs.pnet_GO_single_split import pnet_GO_single_split_config
from prostate_cancer_prediction.configs.pnet_GO_stratified_5_fold_CV import pnet_GO_stratified_5_fold_CV_config
from prostate_cancer_prediction.configs.pnet_nested_CV import pnet_nested_CV_config
from prostate_cancer_prediction.configs.pnetfc_nested_CV import pnetfc_nested_CV_config
from prostate_cancer_prediction.configs.pnet_network_order_fixed_seed import pnet_network_order_fixed_seed_configs
from prostate_cancer_prediction.configs.pnet_single_split import pnet_single_split_config
from prostate_cancer_prediction.configs.pnet_single_split_elmarakeby import pnet_single_split_elmarakeby_config
from prostate_cancer_prediction.configs.pnet_stratified_5_fold_CV import pnet_stratified_5_fold_CV_config
from prostate_cancer_prediction.configs.pnet_train_size_variation import pnet_train_size_variation_configs
from prostate_cancer_prediction.configs.pnet_train_size_variation_nested_CV import pnet_train_size_variation_nested_CV_configs
from prostate_cancer_prediction.configs.pnetfc_train_size_variation_nested_CV import pnetfc_train_size_variation_nested_CV_configs
from prostate_cancer_prediction.configs.dense_single_layer_train_size_variation_nested_CV import dense_single_layer_train_size_variation_nested_CV_configs
from prostate_cancer_prediction.configs.pnetfc_single_split import pnetfc_single_split_config
from prostate_cancer_prediction.configs.pnetfc_single_split_elmarakeby import pnetfc_single_split_elmarakeby_config
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
from prostate_cancer_prediction.configs.pnet_test_set_stability import pnet_test_set_stability_configs
from prostate_cancer_prediction.configs.pnet_GO_test_set_stability import pnet_GO_test_set_stability_configs
from prostate_cancer_prediction.plotting.plot_train_size_variations import plot_train_size_comparisons

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from prostate_cancer_prediction.configs.base_config import download_dir, run_dir, wd, selected_genes, figures_dir
from prostate_cancer_prediction.configs.pnet_single_split import n_hidden_layers
from architecture.run_pipeline import run_pipeline
from prostate_cancer_prediction.plotting.plot_nested_cv import plot_nested_CV
from prostate_cancer_prediction.plotting.plot_stratified_5_fold_cv import plot_stratified_5_fold_CV
from prostate_cancer_prediction.plotting.plot_external_validation import plot_external_validation
from prostate_cancer_prediction.plotting.plot_single_split import plot_single_split_curves
from architecture.plotting.plot_sankey import plot_sankey
from prostate_cancer_prediction.plotting.plot_importance_stability import analyse_importance_stability
from prostate_cancer_prediction.plotting.plot_network_order_variation import plot_network_order_variation
from prostate_cancer_prediction.significance_testing import significance_test


def run():
    configs = [
        # Single split, including grid search.
        # pnet_single_split_config,
        # decision_tree_single_split_config,
        # linear_svm_single_split_config,
        # rbf_svm_single_split_config,
        # random_forest_single_split_config,
        # adaboost_single_split_config,
        # sgd_logistic_regression_single_split_config,
        # dense_single_layer_single_split_config,
        # pnetfc_single_split_config,
        # pnet_GO_single_split_config,
        #
        # # Single split, using hyperparams from original paper.
        # pnet_single_split_elmarakeby_config,
        # decision_tree_single_split_elmarakeby_config,
        # linear_svm_single_split_elmarakeby_config,
        # rbf_svm_single_split_elmarakeby_config,
        # random_forest_single_split_elmarakeby_config,
        # adaboost_single_split_elmarakeby_config,
        # sgd_logistic_regression_single_split_elmarakeby_config,
        # dense_single_layer_single_split_elmarakeby_config,
        # pnetfc_single_split_elmarakeby_config,
        #
        # # Train size variation
        # *pnet_train_size_variation_configs,
        #*pnet_train_size_variation_nested_CV_configs,
        # *pnetfc_train_size_variation_configs,
        #*pnetfc_train_size_variation_nested_CV_configs,
        #*dense_single_layer_train_size_variation_configs,
        #*dense_single_layer_train_size_variation_nested_CV_configs,

        # # Stratified 5-fold CV
        # pnet_stratified_5_fold_CV_config,
        # pnetfc_stratified_5_fold_CV_config,
        # pnet_GO_stratified_5_fold_CV_config,
        # dense_single_layer_stratified_5_fold_CV_config,
        # decision_tree_stratified_5_fold_CV_config,
        # adaboost_stratified_5_fold_CV_config,
        # linear_svm_stratified_5_fold_CV_config,
        # random_forest_stratified_5_fold_CV_config,
        # rbf_svm_stratified_5_fold_CV_config,
        # sgd_logistic_regression_stratified_5_fold_CV_config,
        #
        # # Feature-importance test-train stability
        # *pnet_test_set_stability_configs,
        # *pnet_GO_test_set_stability_configs,
        #
        # # Feature-importance network order stability
        # *pnet_network_order_variation_configs,
        # *pnet_network_order_fixed_seed_configs,
        #
        #pnet_external_validation_1_elmarakeby_config,
        #pnet_external_validation_2_elmarakeby_config,
        #
        # # Stratified nested CV
        #  pnet_nested_CV_config,
        # pnetfc_nested_CV_config,
        # dense_single_layer_nested_CV_config,
        # pnet_GO_nested_CV_config,
        # adaboost_nested_CV_config,
        # decision_tree_nested_CV_config,
        # linear_svm_nested_CV_config,
        # random_forest_nested_CV_config,
        # rbf_svm_nested_CV_config,
        # sgd_logistic_regression_nested_CV_config,
    ]

    for config in configs:
        run_pipeline(config)

    # Single-split ROC/PRC curves (our hyperparameters, test and validation splits kept separate).
    #plot_single_split_curves(run_dir, figures_dir, concat_val=False)

    # Single-split ROC/PRC curves (Elmarakeby et al.'s hyperparameters, test and validation splits combined).
    #plot_single_split_curves(run_dir, figures_dir, tag="elmarakeby", concat_val=True)

    #plot_train_size_comparisons(run_dir, figures_dir)

    #plot_external_validation(run_dir, figures_dir)

    # Sankey diagram for P-NET
    # pnet_run_dir = f"{run_dir}/pnet_single_split_elmarakeby"
    # dataset_id_mappings = "architecture/Reactome/ReactomePathways.txt"
    # pathway_short_names = "architecture/plotting/reactome_short_names.csv"
    # plot_sankey(pnet_run_dir, n_hidden_layers, figures_dir, dataset_id_mappings,
    #            short_name_csv=pathway_short_names)

    # Sankey diagram for P-NET-GO
    # pnet_run_dir = f"{run_dir}/pnet_GO_single_split/best_auc"
    # dataset_id_mappings = "architecture/GO/go_id_name_map.tsv"
    # pathway_short_names = "architecture/plotting/go_short_names.csv"
    # plot_sankey(pnet_run_dir, n_hidden_layers, figures_dir, dataset_id_mappings,
    #            short_name_csv=pathway_short_names, format_pathway_names=True,
    #            output_prefix="pnet_GO_single_split")

    # plot_stratified_5_fold_CV(run_dir, figures_dir)

    #plot_nested_CV(run_dir, figures_dir)

    #plot_external_validation_elmarakeby(run_dir, figures_dir)

    #plot_external_validation(run_dir, figures_dir)

    significance_test(run_dir, wd)




    # Distribution of each test metric across the network-order variation runs
    #plot_network_order_variation(run_dir, figures_dir, run_prefix="pnet_network_order_variation", split="test")

    # Feature-importance stability across the network-order variation runs
    # analyse_importance_stability(
    #     run_dir, figures_dir, n_hidden_layers,
    #     run_id="pnet_network_order_variation",
    #     fold_dirs=[f"{run_dir}/pnet_network_order_variation_{i}"
    #                for i in range(len(pnet_network_order_variation_configs))],
    #     unit="run",
    #     pathway_names="architecture/Reactome/ReactomePathways.txt")

    # Feature-importance stability (each run writes to its own subdirectory under
    # analyse_importance_stability(
    #     run_dir, figures_dir, n_hidden_layers,
    #     run_id="pnet_test_set_stability",
    #     fold_dirs=[f"{run_dir}/pnet_test_set_stability_{i}"
    #                for i in range(len(pnet_test_set_stability_configs))],
    #     unit="run",
    #     pathway_names="architecture/Reactome/ReactomePathways.txt")
    # analyse_importance_stability(
    #     run_dir, figures_dir, n_hidden_layers,
    #     run_id="pnet_GO_test_set_stability",
    #     fold_dirs=[f"{run_dir}/pnet_GO_test_set_stability_{i}"
    #                for i in range(len(pnet_GO_test_set_stability_configs))],
    #     unit="run",
    #     pathway_names="architecture/GO/go_id_name_map.tsv")

if __name__ == "__main__":
    run()
