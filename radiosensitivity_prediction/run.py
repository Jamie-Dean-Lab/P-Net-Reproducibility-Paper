import os
import sys

from radiosensitivity_prediction.configs.adaboost import adaboost_config
from radiosensitivity_prediction.configs.decision_tree import decision_tree_config
from radiosensitivity_prediction.configs.lgbm import lgbm_config
from radiosensitivity_prediction.configs.linear_svm import linear_svm_config
from radiosensitivity_prediction.configs.pnet_GO import pnet_GO_config
from radiosensitivity_prediction.configs.pnet_hyperparameter_sensitivity import pnet_hyperparameter_sensitivity_config
from radiosensitivity_prediction.configs.random_forest import random_forest_config
from radiosensitivity_prediction.configs.rbf_svm import rbf_svm_config
from radiosensitivity_prediction.configs.sgd_logistic_regression import sgd_logistic_regression_config
from radiosensitivity_prediction.configs.xgb import xgb_config

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from radiosensitivity_prediction.significance_testing import significance_test
from radiosensitivity_prediction.configs.base_config import wd, run_dir, data_dir, figures_dir
from radiosensitivity_prediction.configs.pnet import pnet_config
from radiosensitivity_prediction.configs.dense import dense_config
from radiosensitivity_prediction.configs.kernel_regression import krr_config
from architecture.run_pipeline import run_pipeline
from architecture.plotting.plot_sankey import plot_sankey
from radiosensitivity_prediction.plotting.plot_hyperparameter_sensitivity import analyse as plot_hyperparameter_sensitivity
from radiosensitivity_prediction.plotting.plot_nested_cv import plot_nested_CV
from radiosensitivity_prediction.plotting.plot_external_validation import plot_external_validation


def run():
    configs = [
        #pnet_config,
        #dense_config,
        #pnet_GO_config,
        # krr_config,
        # lgbm_config,
        # xgb_config,
        # adaboost_config,
        # decision_tree_config,
        # linear_svm_config,
        # rbf_svm_config,
        # random_forest_config,
        # sgd_logistic_regression_config
        # pnet_hyperparameter_sensitivity_config,
    ]

    for config in configs:
        run_pipeline(config)

    # plot_hyperparameter_sensitivity()

    #significance_test(run_dir, wd)

    plot_nested_CV(run_dir, figures_dir)
    plot_external_validation(run_dir, figures_dir)

    # Sankey diagram for radiosensitivity P-NET
    # pnet_run_dir = os.path.join(run_dir, "pnet_test", "test_0", "best_r2")
    # plot_sankey(
    #     pnet_run_dir,
    #     n_hidden_layers=5,
    #     figures_dir=figures_dir,
    #     dataset_id_mappings="architecture/Reactome/ReactomePathways.txt",
    #     short_name_csv="architecture/plotting/reactome_short_names.csv",
    #     input_nodes=["gexpr", "methylation"],
    #     input_node_labels={"gexpr": "RNA-seq", "methylation": "DNA Methylation"},
    #     input_node_colors={
    #         "gexpr":       "rgba(60,180,75,0.7)",
    #         "methylation": "rgba(145,30,180,0.7)",
    #     },
    # )


if __name__ == "__main__":
    run()
