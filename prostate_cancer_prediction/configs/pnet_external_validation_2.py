import copy
from functools import partial

from keras.regularizers import L2
from keras.callbacks import LearningRateScheduler

from architecture.pipeline import TFPipeline
from architecture.pnet_config import BINARY_CROSSENTROPY, pnet_model_params
from architecture.pnet_model import compile_pnet
from architecture.callbacks_custom import step_decay
from architecture.evaluation import plot_history, get_deeplift_global
from .base_config import (base_config, data_dir, f1_selection, auprc_selection,
                          auc_selection, save_processor, train_samples, val_samples, test_samples, selected_genes)

from prostate_cancer_prediction.feature_encoders import mut_binary, cnv
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, precision_score, \
    recall_score

n_hidden_layers = 5

learning_rate = 1e-3
step_decay_part = partial(step_decay, init_lr=learning_rate, drop=0.25, epochs_drop=10)

_views = [
    ("mut_important", "P1000_final_analysis_set_cross_important_only.csv", selected_genes, 0, mut_binary, lambda x: x),
    ("cnv", "P1000_data_CNA_paper.csv", selected_genes, 0, cnv, lambda x: x),
]

_model_params = pnet_model_params(
    loss=BINARY_CROSSENTROPY,
    o_activation="sigmoid",
    n_hidden_layers=n_hidden_layers,
    learning_rate=learning_rate,
)

pnet_external_validation_2_elmarakeby_config = {
    **copy.deepcopy(base_config),
    "views": _views,
    "run_id": "pnet_external_validation_2_elmarakeby",
    "model": compile_pnet,
    "model_params": _model_params,
    "labels": [("response_paper_external_validation_2.csv", 0)],
    "stratified": False,
    "fitting_params": {
        "epochs": 35,
        "batch": 50,
        "LRScheduler": LearningRateScheduler(step_decay_part, verbose=0),
        "early_stopping": None,
        "prediction_output": "average",
        "shuffle_samples": True,
        "class_weight": [{0: 1, 1: 1}] * (n_hidden_layers + 1),
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
                # Met500_cnv_processed.csv contains single and double amplification/deletion, so treat it identically to primary dataset.
                # (collapse to 3 values (+1,0,-1).
                ("cnv", "../external_validation/Met500/Met500_cnv_processed.csv", selected_genes, 0, cnv,
                 lambda x: x)
            ],
            "labels": [("../external_validation/Met500/Met500_labels.csv", 0)],
        },
        {
            "tag": "PRAD",
            "views": [
                ("mut_important", "../external_validation/PRAD/mut_matrix.csv", selected_genes, 0, mut_binary,
                 lambda x: x),
                # cnv_matrix.csv has already been collapsed to 3 levels (+1,0,-1), no preprocessing necessary.
                ("cnv", "../external_validation/PRAD/cnv_matrix.csv", selected_genes, 0, lambda x: x, lambda x: x),
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
