import os
import sys

from radiosensitivity_prediction.preprocess import Preprocessor

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from radiosensitivity_prediction.aggregate import aggregate_results
from radiosensitivity_prediction.significance_testing import significance_test
from radiosensitivity_prediction.configs.base_config import wd, run_dir, data_dir
from radiosensitivity_prediction.configs.pnet import pnet_config
from radiosensitivity_prediction.configs.dense import dense_config
from radiosensitivity_prediction.configs.kernel_regression import krr_config
from architecture.train import train

_PREPROCESSING_SENTINEL = "nci60_rnaseq_preprocessed.csv"


def run():
    if not os.path.exists(os.path.join(data_dir, _PREPROCESSING_SENTINEL)):
        print("Preprocessed data not found — running preprocessing pipeline...")
        Preprocessor(data_dir).run_all()

    if not os.path.exists(run_dir):
        os.mkdir(run_dir)

    configs = [
        #pnet_config,
        #dense_config,
        krr_config,
    ]

    for config in configs:
        train(config)

    aggregate_results(run_dir, wd)
    #significance_test(run_dir, wd)


if __name__ == "__main__":
    run()