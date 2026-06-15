import os
import sys

from radiosensitivity_prediction.configs.adaboost import adaboost_config
from radiosensitivity_prediction.configs.decision_tree import decision_tree_config
from radiosensitivity_prediction.configs.lgbm import lgbm_config
from radiosensitivity_prediction.configs.linear_svm import linear_svm_config
from radiosensitivity_prediction.configs.random_forest import random_forest_config
from radiosensitivity_prediction.configs.rbf_svm import rbf_svm_config
from radiosensitivity_prediction.configs.sgd_logistic_regression import sgd_logistic_regression_config
from radiosensitivity_prediction.configs.xgb import xgb_config
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from radiosensitivity_prediction.significance_testing import significance_test
from radiosensitivity_prediction.configs.base_config import wd, run_dir, data_dir
from radiosensitivity_prediction.configs.pnet import pnet_config
from radiosensitivity_prediction.configs.dense import dense_config
from radiosensitivity_prediction.configs.kernel_regression import krr_config
from architecture.train import run_pipeline


def run():
    configs = [
        pnet_config,
        dense_config,
        krr_config,
        lgbm_config,
        xgb_config,
        adaboost_config,
        decision_tree_config,
        linear_svm_config,
        rbf_svm_config,
        random_forest_config,
        sgd_logistic_regression_config
    ]

    for config in configs:
        run_pipeline(config)

    # significance_test(run_dir, wd)


if __name__ == "__main__":
    run()
