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

    # Average prediction scores before thresholding
    met_preds = (met1["metastatic_pred"] + met2["metastatic_pred"]) / 2
    prad_preds = (primary1["metastatic_pred"] + primary2["metastatic_pred"]) / 2

    met_binary = (met_preds > 0.5).astype(int)
    prad_binary = (prad_preds > 0.5).astype(int)

    # Reproduce their approach: one row per dataset, [pred==False count, pred==True count]
    primary_row = np.array([sum(prad_binary == 0), sum(prad_binary == 1)])
    mets_row = np.array([sum(met_binary == 0), sum(met_binary == 1)])

    counts = np.array([primary_row, mets_row])
    # Normalise each dataset independently (row-wise)
    cm = 100. * counts.astype('float') / counts.sum(axis=1)[:, np.newaxis]

    fig = plt.figure(figsize=(4, 4))
    ax = fig.subplots(1, 1)
    _plot_confusion_matrix(ax, cm)
    plt.savefig(f"{figures_dir}/pnet_external_validation.pdf")
    plt.close()

    _confusion_source_data(counts, cm).to_csv(
        f"{figures_dir}/pnet_external_validation_source_data.csv", index=False)


# Row order of the matrix built above: the dataset each sample came from, paired
# with the class it stands for in the figure.
_DATASETS = [("PRAD", "localised"), ("Met500", "metastatic")]
# Column order of the matrix: the predicted class (metastatic_pred > 0.5).
_PREDICTED_CLASSES = ["localised", "metastatic"]


def _confusion_source_data(counts, percentages):
    """Return the plotted confusion-matrix cells as a dataframe.

    One row per cell of the figure: the sample count behind it and the
    row-normalised percentage annotated on it. Both arrays are indexed
    [dataset, predicted class]; _plot_confusion_matrix transposes them for
    display, and its TN/FN/FP/TP labels are reproduced in the ``cell`` column so
    a row can be matched to the cell it was drawn in.
    """
    rows = []
    for j, (dataset, dataset_class) in enumerate(_DATASETS):
        for i, predicted in enumerate(_PREDICTED_CLASSES):
            rows.append({
                "dataset": dataset,
                "dataset_class": dataset_class,
                "predicted_class": predicted,
                "cell": _CELL_LABELS[i, j],
                "n_samples": int(counts[j, i]),
                "dataset_n_samples": int(counts[j].sum()),
                "percent_of_dataset": float(percentages[j, i]),
            })
    return pd.DataFrame(rows)


# Cell labels as drawn, indexed [predicted class, dataset].
_CELL_LABELS = np.array([["TN", "FN"], ["FP", "TP"]])


def _plot_confusion_matrix(ax, conf_mat):
    cmap = plt.cm.Reds

    # Columns = true class (dataset), rows = predicted, so each dataset column
    # sums to 100%. conf_mat is row=actual, col=predicted, so transpose.
    cm = conf_mat.T
    labels = _CELL_LABELS
    classes = ["localised", "metastatic"]

    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    fig = ax.get_figure()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="10%", pad=0.1)
    cb = fig.colorbar(im, cax=cax, orientation="vertical")
    cax.yaxis.set_ticks_position("right")
    cb.outline.set_visible(False)

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        # Pick text colour from the actual cell luminance so it stays legible
        # regardless of how imshow autoscales the colormap (white text on a pale
        # cell was the previous problem).
        r, g, b, _ = im.cmap(im.norm(cm[i, j]))
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        ax.text(j, i, "{}: {:.2f}%".format(labels[i, j], cm[i, j]),
                horizontalalignment="center",
                color="white" if luminance < 0.5 else "black", fontsize=12)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=0)
    ax.set_yticks([])
