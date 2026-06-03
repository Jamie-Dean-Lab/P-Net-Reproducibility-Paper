import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

    # Fix label indexing: conf_mat[row=actual, col=predicted], so labels[row, col]
    labels = np.array([["TN: ", "FP: "], ["FN: ", "TP: "]])
    plt.imshow(conf_mat)
    plt.xticks(ticks=[0, 1], labels=["Localised", "Metastatic"])
    plt.yticks(ticks=[], labels=[])
    for i in range(2):
        for j in range(2):
            plt.text(j - 0.25, i, f"{labels[i, j]}{round(conf_mat[i, j], 2)}%")
    plt.colorbar()
    plt.savefig(f"{figures_dir}/pnet_external_validation.jpg")
    plt.close()