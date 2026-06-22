import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, t
from statsmodels.stats.multitest import multipletests


def significance_test(run_dir, wd, selection_metric="auc"):
    metrics = ["auc", "auprc", "f1", "accuracy", "precision", "recall"]

    model_names = [
        "pnet",
        "pnet_GO",
        "dense",
        "decision_tree",
        "adaboost",
        "svc",
        "random_forest",
        "rbf_svm",
        "xgb",
        "lgbm",
        "sgd_logistic_regression",
    ]

    # Load per-fold test scores for each model
    model_scores = {}
    for model_name in model_names:
        model_dir = os.path.join(run_dir, model_name)
        if not os.path.isdir(model_dir):
            continue
        fold_scores = []
        for test_dir in sorted([d for d in os.listdir(model_dir) if d.startswith("test_")]):
            path = os.path.join(model_dir, test_dir, f"best_{selection_metric}", "summary_results.csv")
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path, index_col=0)
            df.columns = [c.replace("response_", "") for c in df.columns]
            fold_scores.append(df[df.index == "test"][metrics])
        if fold_scores:
            model_scores[model_name] = pd.concat(fold_scores).reset_index(drop=True)

    # P-NET vs every other model, two-sided paired t-test
    comparisons = [
        ("pnet", other, f"pnet_vs_{other}")
        for other in model_names
        if other != "pnet" and other in model_scores
    ]

    results_full = []
    for model1, model2, comp_name in comparisons:
        for metric in metrics:
            scores1 = model_scores[model1][metric].values
            scores2 = model_scores[model2][metric].values

            t_stat, p_raw = ttest_rel(scores1, scores2, alternative="two-sided")

            diffs = scores1 - scores2
            n = len(diffs)
            df_deg = n - 1
            mean_diff = np.mean(diffs)
            sem = np.std(diffs, ddof=1) / np.sqrt(n)
            cohens_d = mean_diff / np.std(diffs, ddof=1)
            t_crit = t.ppf(0.975, df_deg)

            results_full.append({
                "comparison": f"{comp_name}_{metric}",
                "model1":     model1,
                "model2":     model2,
                "metric":     metric,
                "t":          t_stat,
                "df":         df_deg,
                "mean_diff":  mean_diff,
                "sem":        sem,
                "ci_low":     mean_diff - t_crit * sem,
                "ci_high":    mean_diff + t_crit * sem,
                "cohens_d":   cohens_d,
                "p_raw":      p_raw,
            })

    rejected, p_fdr, _, _ = multipletests(
        [r["p_raw"] for r in results_full], alpha=0.05, method="fdr_bh"
    )

    sigresults = pd.DataFrame(results_full)
    sigresults["p_fdr"] = p_fdr
    sigresults["significant"] = rejected

    out_path = os.path.join(wd, "significance_tests.csv")
    sigresults.to_csv(out_path, float_format="%.4f")
    print(f"Glioma significance test results saved to {out_path}")
    print(sigresults.round(4).to_string())
