import os, sys

from tissue_type_classification.significance_testing import significance_test

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from tissue_type_classification.configs.base_config import wd, run_dir
from tissue_type_classification.configs.pnet import pnet_config
from tissue_type_classification.configs.dense import dense_config
from tissue_type_classification.configs.svc import svc_config
from tissue_type_classification.configs.adaboost import adaboost_config
from tissue_type_classification.configs.decision_tree import decision_tree_config
from tissue_type_classification.configs.lgbm import lgbm_config
from tissue_type_classification.configs.random_forest import random_forest_config
from tissue_type_classification.configs.rbf_svm import rbf_svm_config
from tissue_type_classification.configs.sgd_logistic_regression import sgd_logistic_regression_config
from tissue_type_classification.configs.xgb import xgb_config
from architecture.run_pipeline import run_pipeline


def run():
    configs = [
        pnet_config,
        dense_config,
        svc_config,
        adaboost_config,
        decision_tree_config,
        lgbm_config,
        random_forest_config,
        rbf_svm_config,
        sgd_logistic_regression_config,
        xgb_config,
    ]

    for config in configs:
        run_pipeline(config)

    significance_test(run_dir, wd)


if __name__ == "__main__":
    run()