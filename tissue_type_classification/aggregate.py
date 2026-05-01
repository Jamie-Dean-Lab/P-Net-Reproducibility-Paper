import numpy as np
import pandas as pd


def aggregate_results(run_dir, wd):
    metrics = ["auc", "auprc", "f1", "accuracy"]
    results = {m: pd.read_csv(f"{run_dir}/{m}/results.csv", index_col=0) for m in __import__('os').listdir(run_dir)}

    result_table = []
    for k, v in results.items():
        df = v.groupby("index")[metrics].agg(["mean", "std"])
        df["model"] = k
        result_table.append(df)
    pd.concat(result_table).to_csv(f"{wd}/results.csv")
    return results