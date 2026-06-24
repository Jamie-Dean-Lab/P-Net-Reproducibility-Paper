import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from glioma_prediction.configs.pnet import pnet_config, n_hidden_layers
from glioma_prediction.configs.pnet_GO import pnet_GO_config
from glioma_prediction.configs.pnet_hyperparameter_sensitivity import pnet_hyperparameter_sensitivity_config
from glioma_prediction.configs.dense import dense_config
from glioma_prediction.configs.decision_tree import decision_tree_config
from glioma_prediction.configs.random_forest import random_forest_config
from glioma_prediction.configs.svc import svc_config
from glioma_prediction.configs.rbf_svm import rbf_svm_config
from glioma_prediction.configs.xgb import xgb_config
from glioma_prediction.configs.sgd_logistic_regression import sgd_logistic_regression_config
from glioma_prediction.configs.adaboost import adaboost_config
from glioma_prediction.configs.lgbm import lgbm_config
from architecture.run_pipeline import run_pipeline
from glioma_prediction.configs.base_config import run_dir, figures_dir, wd
from glioma_prediction.plotting.plot_nested_cv import plot_nested_CV
from glioma_prediction.plotting.plot_external_validation import plot_external_validation
from glioma_prediction.plotting.plot_hyperparameter_sensitivity import analyse as plot_hyperparameter_sensitivity
from architecture.plotting.plot_sankey import plot_sankey
from glioma_prediction.significance_testing import significance_test


def run():
    configs = [
        pnet_config,
        # pnet_GO_config,
        # dense_config,
        # decision_tree_config,
        # random_forest_config,
        # svc_config,
        # rbf_svm_config,
        # xgb_config,
        # sgd_logistic_regression_config,
        # adaboost_config,
        # lgbm_config,
        # pnet_hyperparameter_sensitivity_config,
    ]

    for config in configs:
        run_pipeline(config)

    # plot_nested_CV(run_dir, figures_dir)
    # plot_external_validation(run_dir, figures_dir)
    # plot_hyperparameter_sensitivity()

    # Sankey diagram of the first outer split's P-NET model.
    # pnet_run_dir = f"{run_dir}/pnet/test_0/best_auc"
    # dataset_id_mappings = "architecture/Reactome/ReactomePathways.txt"
    # pathway_short_names = "architecture/plotting/reactome_short_names.csv"
    # plot_sankey(pnet_run_dir, n_hidden_layers, figures_dir, dataset_id_mappings,
    #             pathway_short_names, output_prefix="pnet_single_split")

    significance_test(run_dir, wd)


if __name__ == "__main__":
    run()
