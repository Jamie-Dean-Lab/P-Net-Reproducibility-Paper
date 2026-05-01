import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import t
from statsmodels.stats.multitest import multipletests

def _paired_ttest(scores1, scores2, label):
    t_stat, p_raw = stats.ttest_rel(scores2, scores1, alternative="greater")
    diffs = scores2 - scores1
    n = len(diffs)
    df = n - 1
    mean_diff = np.mean(diffs)
    sem = stats.sem(diffs)
    cohens_d = mean_diff / np.std(diffs, ddof=1)
    t_crit = t.ppf(0.975, df)
    return {
        "comparison": label,
        "t":          t_stat,
        "df":         df,
        "mean_diff":  mean_diff,
        "sem":        sem,
        "ci_low":     mean_diff - t_crit * sem,
        "ci_high":    mean_diff + t_crit * sem,
        "cohens_d":   cohens_d,
        "p_raw":      p_raw,
    }


def significance_test(run_dir, wd):
    results = {m: pd.read_csv(f"{run_dir}/{m}/results.csv", index_col=0) for m in __import__('os').listdir(run_dir)}

    def test_scores(model):
        return lambda metric: results[model].loc[results[model]["index"] == "test", metric].values

    pnet_scores  = test_scores("pnet")
    dense_scores = test_scores("dense")
    svc_scores   = test_scores("svc")

    results_full = []

    for metric in ["auc", "auprc", "f1", "accuracy"]:
        results_full.append(_paired_ttest(pnet_scores(metric), dense_scores(metric), f"dense_vs_pnet_{metric}"))

    for metric in ["auprc", "f1", "accuracy"]:
        results_full.append(_paired_ttest(pnet_scores(metric), svc_scores(metric), f"svc_vs_pnet_{metric}"))

    rejected, p_fdr, _, _ = multipletests([r["p_raw"] for r in results_full], alpha=0.05, method="fdr_bh")

    sigresults = pd.DataFrame(results_full)
    sigresults["p_fdr"] = p_fdr
    sigresults["significant"] = rejected
    sigresults.to_csv(f"{wd}/significance_tests.csv", float_format="%.4f")
    print(sigresults.round(4))