from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from keras.activations import tanh, sigmoid
from keras.regularizers import L2
from keras.callbacks import LearningRateScheduler

from architecture.data_utils import ConcatMultiViewDataset
from architecture.pipeline import IdentityProcessor
from architecture.callbacks_custom import step_decay_part
from dense_model import compile_dense
from architecture.evaluation import collate_grid_search

n_hidden_layers = 5

dense_config = {
    "dataloader" : ConcatMultiViewDataset,
    "feature_selector" : IdentityProcessor(),
    "feature_preprocessor" : IdentityProcessor(),
    "data_augmentor" : lambda x : x,
    "rng_seed" : 42,
    "tt_split_seed" : 42,
    "model" : compile_dense,
    "model_params" : {
        "h_activation" : "selu",
        "o_activation" : "sigmoid",
        "n_hidden_layers" : 0,
        "h_reg" : (L2, {"l2" : 1e-3}),
        "n_weights" : 71009,
        "loss" : {"class_name" : "BinaryCrossentropy", "config" : {"from_logits" : False}},
        "optimizer" : {"class_name" : "Adam", "config" : {"learning_rate" : 1e-3}}
    },
    "fitting_params" : {
        "epochs" : 300,
        "batch" : 50,
        "LRScheduler" : LearningRateScheduler(step_decay_part, verbose=0),
        "early_stopping" : None,
        "prediction_output" : "average",
        "shuffle_samples" : True,
        "class_weight" : [[0.75, 1.5]] * (n_hidden_layers + 1)
    },
    "grid_search" : [],
    "val_metric" : lambda x : x,
    "use_validation_on_test" : False,
    "results_processors" : [],
    "fold_collators" : [],
    "grid_search_collators" : [collate_grid_search],
    "drop_labels" : True
}