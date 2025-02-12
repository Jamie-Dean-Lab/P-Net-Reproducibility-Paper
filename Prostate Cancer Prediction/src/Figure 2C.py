import matplotlib.pyplot as plt
import os
import pandas as pd
import json

run_dir = "Prostate Cancer Prediction/runs"

"""
dense_auprcs = []
for f in os.listdir(f"{run_dir}/dense_train_size_trend"):
    wd = f"{run_dir}/dense_train_size_trend/{f}"
    if os.path.isdir(wd):
      with open(f"{wd}/config.txt") as fr:
        config = json.loads(fr.read())
      df = pd.read_csv(f"{wd}/summary_results.csv", index_col=0)
      dense_auprcs.append({"seed" : config["tt_split_seed"], "n_samples" : config["train_samples"] * 1013,
                          "auprc" : df.loc["test", "response_auprc"]})
pnet_auprcs = []
for f in os.listdir(f"{run_dir}/pnet_train_size_trend"):
    wd = f"{run_dir}/pnet_train_size_trend/{f}"
    if os.path.isdir(wd):
      with open(f"{wd}/config.txt") as fr:
        config = json.loads(fr.read())
      df = pd.read_csv(f"{wd}/summary_results.csv", index_col=0)
      pnet_auprcs.append({"seed" : config["tt_split_seed"], "n_samples" : config["train_samples"] * 1013,
                          "auprc" : df.loc["test", "response_auprc"]})
"""
dense_auprcs = [{"n_samples" : int(k.split("_")[1]), 
                 "auprc" : pd.read_csv(f"{run_dir}/{k}/summary_results.csv", index_col=0).loc["test", "response_auprc"]} 
                 for k in os.listdir(run_dir) if k.find("dense") > -1]
pnet_auprcs = [{"n_samples" : int(k.split("_")[1]), 
                 "auprc" : pd.read_csv(f"{run_dir}/{k}/summary_results.csv", index_col=0).loc["test", "response_auprc"]} 
                 for k in os.listdir(run_dir) if k.find("pnet") > -1]
dense_auprcs = pd.DataFrame(dense_auprcs)
pnet_auprcs = pd.DataFrame(pnet_auprcs)

plt.plot(dense_auprcs["n_samples"], dense_auprcs["auprc"], label="dense")
plt.plot(pnet_auprcs["n_samples"], pnet_auprcs["auprc"], label="pnet")
plt.xlabel("n_samples")
plt.ylabel("AUPRC")
plt.legend()
plt.show()
pass