"""
Extracts the train/test sample IDs for outer fold 0 of the main pnet crossvalidation
(outer_kfolds, tt_split_seed and stratified taken from base_config) and saves them to
data/sensitivity_split_fold0.json.
"""

from architecture.sensitivity_split import extract_sensitivity_split as _extract_sensitivity_split
from tissue_type_classification.configs.base_config import base_config, data_dir


def extract_sensitivity_split():
    return _extract_sensitivity_split(base_config, data_dir)


if __name__ == "__main__":
    extract_sensitivity_split()
