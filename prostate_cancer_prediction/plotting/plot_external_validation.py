import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_external_validation(run_dir, figures_dir):
    met1 = pd.read_csv(f"{run_dir}/pnet_external_validation_1_elmarakeby/external_validation/Met500/predictions.csv", index_col=0)
    primary1 = pd.read_csv(f"{run_dir}/pnet_external_validation_1_elmarakeby/external_validation/PRAD/predictions.csv", index_col=0)
    combined1 = pd.concat([met1, primary1])
    conf_mat1 = confusion_matrix(combined1["metastatic"], round(combined1["metastatic_pred"]), normalize="true")

    met2 = pd.read_csv(f"{run_dir}/pnet_external_validation_2_elmarakeby/external_validation/Met500/predictions.csv", index_col=0)
    primary2 = pd.read_csv(f"{run_dir}/pnet_external_validation_2_elmarakeby/external_validation/PRAD/predictions.csv", index_col=0)
    combined2 = pd.concat([met2, primary2])
    conf_mat2 = confusion_matrix(combined2["metastatic"], round(combined2["metastatic_pred"]), normalize="true")

    conf_mat = (conf_mat1 + conf_mat2) / 2 * 100
    labels = np.array([["TN: ", "FP: "],["FN: ", "TP: "]])
    plt.imshow(conf_mat)
    plt.xticks(ticks=[0,1], labels=["Localised", "Metastatic"])
    plt.yticks(ticks=[], labels=[])
    for i in range(2):
        for j in range(2):
            plt.text(i-0.25, j, f"{labels[i, j]}{round(conf_mat[i,j], 2)}%")
    plt.colorbar()
    plt.savefig(f"{figures_dir}/pnet_external_validation.jpg")
    plt.close()
