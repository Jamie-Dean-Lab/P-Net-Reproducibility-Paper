import copy
from functools import partial

from keras.regularizers import L2
from keras.callbacks import LearningRateScheduler

from architecture.pipeline import TFPipeline
from architecture.pnet_model import compile_pnet
from architecture.callbacks_custom import step_decay
from architecture.evaluation import plot_history, get_deeplift_global
from .base_config import (base_config, data_dir, f1_selection, auprc_selection,
                          auc_selection, save_processor, train_samples, val_samples, test_samples, selected_genes)

from prostate_cancer_prediction.feature_encoders import mut_binary, cnv_amp, cnv_del
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, precision_score, \
    recall_score

n_hidden_layers = 5

learning_rate = 1e-3
step_decay_part = partial(step_decay, init_lr=learning_rate, drop=0.25, epochs_drop=10)

_model_params = {
    "pathway_dataset": "reactome",
    "pp_relations": "architecture/Reactome/ReactomePathwaysRelation.txt",
    "gp_relations": "architecture/Reactome/ReactomePathways.gmt",
    "n_hidden_layers": n_hidden_layers,
    "h_dropout": [0.5] + [0.1] * n_hidden_layers,
    "h_activation": ["tanh"] * (n_hidden_layers + 1),
    "o_activation": ["sigmoid"] * (n_hidden_layers + 1),
    "h_kernel_initializer": ["lecun_uniform"] * (n_hidden_layers + 1),
    "h_kernel_constraints": [None] * (n_hidden_layers + 1),
    "h_bias_initializer": ["lecun_uniform"] * (n_hidden_layers + 1),
    "h_bias_constraints": [None] * (n_hidden_layers + 1),
    "batch_normal": False,
    "sparse": True,
    "dropout_testing": False,
    "loss": [{"class_name": "BinaryCrossentropy", "config": {"from_logits": False}}] * (n_hidden_layers + 1),
    "loss_weights": [2, 7, 20, 54, 148, 400],
    "optimizer": {"class_name": "Adam", "config": {"learning_rate": learning_rate}},
    "map_seed": 42
}

pnet_external_validation_1_elmarakeby_config = {
    **copy.deepcopy(base_config),
    "run_id": "pnet_external_validation_1_elmarakeby",
    "model": compile_pnet,
    "model_params": _model_params,
    "labels": [("response_paper_external_validation_1.csv", 0)],
    "stratified": False,
    "fitting_params": {
        "epochs": 35,
        "batch": 50,
        "LRScheduler": LearningRateScheduler(step_decay_part, verbose=0),
        "early_stopping": None,
        "prediction_output": "average",
        "shuffle_samples": True,
        "class_weight": [[1, 1]] * (n_hidden_layers + 1),
    },
    "results_processors": [
        save_processor,
        plot_history
    ],
    "val_metric": {"auc": auc_selection},
    "pipeline_class": TFPipeline,
    "run_method": "run_single_split",
    "grid_search": {
        "model_params": {
            f"h_reg_{h}_o_reg_{o}": {**_model_params,
                                     "h_reg": [(L2, {"l2": h})] * (n_hidden_layers + 1),
                                     "o_reg": [(L2, {"l2": o})] * (n_hidden_layers + 1)}
            for h in [1e-3]
            for o in [1e-2]
        },
    },
    "ext_validation_hyperparam_selection_metric": "auc",
    "external_datasets": [
        {
            "tag": "Met500",
            "views": [
                ("mut_important", "../external_validation/Met500/Met500_mut_matrix_processed.csv", selected_genes, 0,
                 mut_binary, lambda x: x),
                ("cnv_amp", "../external_validation/Met500/Met500_cnv_processed.csv", selected_genes, 0, cnv_amp,
                 lambda x: x),
                ("cnv_del", "../external_validation/Met500/Met500_cnv_processed.csv", selected_genes, 0, cnv_del,
                 lambda x: x),
            ],
            "labels": [("../external_validation/Met500/Met500_labels.csv", 0)],
        },
        {
            "tag": "PRAD",
            "views": [
                ("mut_important", "../external_validation/PRAD/mut_matrix.csv", selected_genes, 0, mut_binary,
                 lambda x: x),
                ("cnv_amp", "../external_validation/PRAD/cnv_matrix.csv", selected_genes, 0, cnv_amp, lambda x: x),
                ("cnv_del", "../external_validation/PRAD/cnv_matrix.csv", selected_genes, 0, cnv_del, lambda x: x),
            ],
            "labels": [("../external_validation/PRAD/PRAD_labels.csv", 0)],
        }
    ],
    "external_validation_metrics": {
        "auc": roc_auc_score,
        "auprc": average_precision_score,
        "f1": lambda ys, preds: f1_score(ys, (preds > 0.5).astype(int)),
        "accuracy": lambda ys, preds: accuracy_score(ys, (preds > 0.5).astype(int)),
        "precision": lambda ys, preds: precision_score(ys, (preds > 0.5).astype(int)),
        "recall": lambda ys, preds: recall_score(ys, (preds > 0.5).astype(int)),
    },
}
