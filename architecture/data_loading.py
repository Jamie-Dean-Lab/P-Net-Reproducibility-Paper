"""
Single implementation of "turn a config's views and labels into an aligned dataset".

Used by Pipeline._load_data for training runs, and by the standalone scripts that
need to reproduce a run's exact sample set or fold composition without starting a
pipeline (see sensitivity_split.py and the fold-distribution plots). Keeping one
implementation means those scripts cannot silently drift from what the pipeline
actually does.
"""

import os


def load_dataset(config, shuffle_seed=None):
    """
    Builds the dataset described by a config, then aligns its views.

    args:
        config (dict)      : experiment config, supplying dataloader, views, labels,
                             data_dir, view_alignment_method, drop_labels and shuffle_seed
        shuffle_seed (int) : overrides config["shuffle_seed"] for the input gene
                             ordering; defaults to the config's own value

    returns:
        the aligned dataset object produced by config["dataloader"]
    """
    if shuffle_seed is None:
        shuffle_seed = config["shuffle_seed"]

    data = config["dataloader"]()
    view_aligner = {}
    for view_name, data_fn, selected_columns, id_col, preprocessor, aligner in config["views"]:
        data.load_data_view(view_name, os.path.join(config["data_dir"], data_fn),
                            selected_columns, id_col, preprocessor)
        view_aligner[view_name] = aligner
    for label_fn, id_col in config["labels"]:
        data.load_data_label(os.path.join(config["data_dir"], label_fn), id_col)

    data.align_views(config["view_alignment_method"], view_aligner,
                     drop_labels=config["drop_labels"],
                     shuffle_seed=shuffle_seed)
    return data


def outer_fold_ids(config):
    """
    Sample IDs of each outer crossvalidation fold, exactly as run_crossvalidation
    would generate them (same outer_kfolds, stratified flag and tt_split_seed).

    args:
        config (dict) : experiment config

    returns:
        list[tuple[list[str], list[str]]] : (train_ids, test_ids) per outer fold
    """
    data = load_dataset(config)
    return [
        (list(train_df.ids), list(test_df.ids))
        for train_df, test_df in data.get_k_splits(config["outer_kfolds"],
                                                   config["stratified"],
                                                   config["tt_split_seed"])
    ]
