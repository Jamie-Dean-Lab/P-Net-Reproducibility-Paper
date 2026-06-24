"""
Extracts the train/test sample IDs for outer fold 0 of the main pnet crossvalidation
(outer_kfolds, tt_split_seed and stratified taken from base_config) and saves them to
data/sensitivity_split_fold0.json.
"""

import json
import os

from tissue_type_classification.configs.base_config import base_config, data_dir
from architecture.data_utils import ConcatMultiViewDataset


def extract_sensitivity_split():
    data = ConcatMultiViewDataset()
    view_aligner = {}
    for view_name, data_fn, selected_columns, id_col, preprocessor, aligner in base_config["views"]:
        data.load_data_view(view_name, os.path.join(data_dir, data_fn), selected_columns, id_col, preprocessor)
        view_aligner[view_name] = aligner
    for label_fn, id_col in base_config["labels"]:
        data.load_data_label(os.path.join(data_dir, label_fn), id_col)
    data.align_views(
        base_config["view_alignment_method"],
        view_aligner,
        base_config["drop_labels"],
        base_config["shuffle_seed"],
    )

    folds = list(data.get_k_splits(base_config["outer_kfolds"], base_config["stratified"], base_config["tt_split_seed"]))
    train_df, test_df = folds[0]

    out_path = os.path.join(data_dir, "sensitivity_split_fold0.json")
    with open(out_path, "w") as f:
        json.dump({"train": list(train_df.ids), "test": list(test_df.ids)}, f, indent=2)

    print(f"Saved {len(train_df.ids)} train / {len(test_df.ids)} test IDs to {out_path}")


if __name__ == "__main__":
    extract_sensitivity_split()
