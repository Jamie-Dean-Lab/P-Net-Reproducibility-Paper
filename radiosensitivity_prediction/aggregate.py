import os
import pandas as pd

def aggregate_results(run_dir, wd):
    run_dir = run_dir + "/krr"
    metrics = ["AUC_log1p_r2", "AUC_log1p_explained_variance", "AUC_log1p_mse", "AUC_log1p_mae"]

    results = pd.read_csv(f"{run_dir}/results.csv", index_col=0)

    result_table = []
    for hyperparams, group in results.groupby("hyperparams"):
        df = group.groupby("index")[metrics].agg(["mean", "std"])
        df["hyperparams"] = hyperparams
        result_table.append(df)
    pd.concat(result_table).to_csv(f"{wd}/results.csv")