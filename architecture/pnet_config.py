"""
Factory for the P-NET model_params block shared by every task's configs.
"""

# Pathway hierarchy backing the sparse layers. pathway_dataset, pp_relations and
# gp_relations always move together, so they are selected as a set.
PATHWAY_SOURCES = {
    "reactome": {
        "pathway_dataset": "reactome",
        "pp_relations": "architecture/Reactome/ReactomePathwaysRelation.txt",
        "gp_relations": "architecture/Reactome/ReactomePathways.gmt",
    },
    "go": {
        "pathway_dataset": "go",
        "pp_relations": "architecture/GO/go_hierarchy.tsv",
        "gp_relations": "architecture/GO/go_gene_sets.gmt",
    },
}

# Loss weights rise steeply with depth, so the deepest decision head dominates
# training. Unchanged from Elmarakeby et al.
LOSS_WEIGHTS = [2, 7, 20, 54, 148, 400]

BINARY_CROSSENTROPY = {"class_name": "BinaryCrossentropy", "config": {"from_logits": False}}
CATEGORICAL_CROSSENTROPY = {"class_name": "CategoricalCrossentropy", "config": {"from_logits": False}}
MEAN_SQUARED_ERROR = "MeanSquaredError"


def pnet_model_params(loss, o_activation, n_hidden_layers=5, pathway_dataset="reactome",
                      learning_rate=1e-3, h_dropout=(0.5, 0.1), sparse=True, map_seed=42,
                      **overrides):
    """
    Builds the model_params dict passed to compile_pnet.

    args:
        loss (str or dict)   : Keras loss applied to every decision head
        o_activation (str)   : activation on every decision head ("sigmoid" for binary,
                               "softmax" for multiclass, "linear" for regression)
        n_hidden_layers (int): depth of the P-NET hierarchy
        pathway_dataset (str): key into PATHWAY_SOURCES ("reactome" or "go")
        learning_rate (float): Adam learning rate
        h_dropout (tuple)    : (first-layer rate, rate for every later layer). Note these
                               rates are inert unless apply_training_dropout is also set —
                               see build_pnet in pnet_model.py.
        sparse (bool)        : True for P-NET, False for the fully-connected P-NET-FC
        map_seed (int)       : rng seed for the pathway/gene ordering of the network
        overrides            : any further model_params entries, applied last, e.g.
                               h_reg / o_reg or apply_training_dropout

    returns:
        dict : model_params suitable for compile_pnet
    """
    n = n_hidden_layers
    first_dropout, rest_dropout = h_dropout
    params = {
        **PATHWAY_SOURCES[pathway_dataset],
        "n_hidden_layers": n,
        "h_dropout": [first_dropout] + [rest_dropout] * n,
        "h_activation": ["tanh"] * (n + 1),
        "o_activation": [o_activation] * (n + 1),
        "h_kernel_initializer": ["lecun_uniform"] * (n + 1),
        "h_kernel_constraints": [None] * (n + 1),
        "h_bias_initializer": ["lecun_uniform"] * (n + 1),
        "h_bias_constraints": [None] * (n + 1),
        "batch_normal": False,
        "sparse": sparse,
        "dropout_testing": False,
        "loss": [loss] * (n + 1),
        "loss_weights": list(LOSS_WEIGHTS),
        "optimizer": {"class_name": "Adam", "config": {"learning_rate": learning_rate}},
        "map_seed": map_seed,
    }
    params.update(overrides)
    return params
