import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, t
from statsmodels.stats.multitest import multipletests


def significance_test(run_dir, wd, selection_metric="auc"):
    metric = "auc"

    # P-NET is compared against every other model we ran nested CV on;
    # names match plot_nested_cv
    reference = "pnet_nested_CV"
    baseline_models = [
        "pnet_GO_nested_CV",
        "pnetfc_nested_CV",
        "dense_single_layer_nested_CV",
        "decision_tree_nested_CV",
        "adaboost_nested_CV",
        "linear_svm_nested_CV",
        "random_forest_nested_CV",
        "rbf_svm_nested_CV",
        "sgd_logistic_regression_nested_CV",
    ]
    comparisons = [
        (reference, m, f"pnet_vs_{m.replace('_nested_CV', '')}")
        for m in baseline_models
    ]
    model_names = [reference] + baseline_models

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
            fold_scores.append(df[df.index == "test"][[metric]])
        if fold_scores:
            model_scores[model_name] = pd.concat(fold_scores).reset_index(drop=True)

    comparisons = [(m1, m2, name) for m1, m2, name in comparisons
                   if m1 in model_scores and m2 in model_scores]

    results_full = []
    for model1, model2, comp_name in comparisons:
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
            "comparison": comp_name,
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
    print(f"Prostate significance test results saved to {out_path}")
    print(sigresults.round(4).to_string())
