import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import confusion_matrix


def plot_external_validation(run_dir, figures_dir):
    met1 = pd.read_csv(f"{run_dir}/pnet_external_validation_1_elmarakeby/external_validation/Met500/predictions.csv", index_col=0)
    primary1 = pd.read_csv(f"{run_dir}/pnet_external_validation_1_elmarakeby/external_validation/PRAD/predictions.csv", index_col=0)

    met2 = pd.read_csv(f"{run_dir}/pnet_external_validation_2_elmarakeby/external_validation/Met500/predictions.csv", index_col=0)
    primary2 = pd.read_csv(f"{run_dir}/pnet_external_validation_2_elmarakeby/external_validation/PRAD/predictions.csv", index_col=0)

    # Average prediction scores before thresholding, then concatenate datasets
    met_preds  = (met1["metastatic_pred"] + met2["metastatic_pred"]) / 2
    prad_preds = (primary1["metastatic_pred"] + primary2["metastatic_pred"]) / 2
    all_preds  = pd.concat([met_preds, prad_preds])
    all_true   = pd.concat([met1["metastatic"], primary1["metastatic"]])

    conf_mat = confusion_matrix(all_true, (all_preds > 0.5).astype(int), normalize="true") * 100

    fig = plt.figure(figsize=(4, 4))
    ax = fig.subplots(1, 1)
    _plot_confusion_matrix(ax, conf_mat)
    plt.savefig(f"{figures_dir}/pnet_external_validation.jpg", dpi=200)
    plt.close()


def _plot_confusion_matrix(ax, conf_mat):
    cmap = plt.cm.Reds

    # Columns = true class (dataset), rows = predicted, so each dataset column
    # sums to 100%. conf_mat is row=actual, col=predicted, so transpose.
    cm = conf_mat.T
    labels = np.array([["TN", "FN"], ["FP", "TP"]])
    classes = ["localized", "metastatic"]

    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    fig = ax.get_figure()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="10%", pad=0.1)
    cb = fig.colorbar(im, cax=cax, orientation="vertical")
    cax.yaxis.set_ticks_position("right")
    cb.outline.set_visible(False)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, "{}: {:.2f}%".format(labels[i, j], cm[i, j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black", fontsize=12)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=0)
    ax.set_yticks([])

    ax2 = divider.append_axes("bottom", size="5%", pad=0.6)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)
    ax2.set_yticks([])
    ax2.set_xticks([])
