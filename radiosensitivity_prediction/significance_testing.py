import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, t
from statsmodels.stats.multitest import multipletests

def significance_test(run_dir, wd):
    metrics = ["auc_r2", "auc_explained_variance", "auc_mse", "auc_mae"]

    results = {m: pd.read_csv(f"{run_dir}/{m}/results.csv", index_col=0) for m in os.listdir(run_dir)}

    comparisons = [
        ("pnet", "dense", "pnet_vs_dense"),
        ("pnet", "krr",   "pnet_vs_krr"),
        ("dense", "krr",  "dense_vs_krr"),
    ]

    results_full = []
    for metric in metrics:
        for model1, model2, comp_name in comparisons:
            scores1 = results[model1].loc[results[model1]["index"] == "test", metric].values
            scores2 = results[model2].loc[results[model2]["index"] == "test", metric].values

            t_stat, p_raw = ttest_rel(scores2, scores1, alternative="greater")

            diffs = scores2 - scores1
            n = len(diffs)
            df = n - 1
            mean_diff = np.mean(diffs)
            sem = np.std(diffs, ddof=1) / np.sqrt(n)
            cohens_d = mean_diff / np.std(diffs, ddof=1)
            t_crit = t.ppf(0.975, df)

            results_full.append({
                "comparison": f"{comp_name}_{metric}",
                "model1":     model1,
                "model2":     model2,
                "metric":     metric,
                "t":          t_stat,
                "df":         df,
                "mean_diff":  mean_diff,
                "sem":        sem,
                "ci_low":     mean_diff - t_crit * sem,
                "ci_high":    mean_diff + t_crit * sem,
                "cohens_d":   cohens_d,
                "p_raw":      p_raw,
            })

    rejected, p_fdr, _, _ = multipletests([r["p_raw"] for r in results_full], alpha=0.05, method="fdr_bh")

    sigresults = pd.DataFrame(results_full)
    sigresults["p_fdr"] = p_fdr
    sigresults["significant"] = rejected
    sigresults.to_csv(f"{wd}/significance_tests.csv", float_format="%.4f")
    print("Radiosensitivity full results (t(df), CI, Cohen's d, FDR p-values):")
    print(sigresults.round(4))