"""
Shared implementation for extracting a fixed train/test split from a task's main
P-NET crossvalidation.

The hyperparameter-sensitivity sweeps vary one hyperparameter at a time over a
single fixed split rather than running the full nested crossvalidation, so they
need the exact sample IDs of that split. Each task exposes a zero-argument wrapper
around this function in its own extract_sensitivity_split.py.
"""

import json
import os

from architecture.data_loading import outer_fold_ids


def extract_sensitivity_split(base_config, data_dir=None, fold=0):
    """
    Writes the sample IDs of one outer crossvalidation fold to JSON.

    outer_kfolds, stratified and tt_split_seed are taken from base_config, so the
    split produced here is the same one run_crossvalidation would generate.

    args:
        base_config (dict) : the task's base config, supplying views, labels, the
                             view alignment method and the split seeds
        data_dir (str)     : where the split JSON is written; defaults to the
                             config's own data_dir
        fold (int)         : which outer fold to extract

    returns:
        str : path of the JSON file written
    """
    if data_dir is None:
        data_dir = base_config["data_dir"]

    train_ids, test_ids = outer_fold_ids(base_config)[fold]

    out_path = os.path.join(data_dir, f"sensitivity_split_fold{fold}.json")
    with open(out_path, "w") as f:
        json.dump({"train": train_ids, "test": test_ids}, f, indent=2)

    print(f"Saved {len(train_ids)} train / {len(test_ids)} test IDs to {out_path}")
    return out_path
