import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, t
from statsmodels.stats.multitest import multipletests


def significance_test(run_dir, wd):
    metric = "f1"  # weighted F1

    # P-NET is compared against every other model; display names match plot_nested_cv
    baseline_models = [
        "dense",                    # P-NET-FC
        "decision_tree",
        "adaboost",
        "sgd_logistic_regression",
        "svc",                      # Linear SVM
        "rbf_svm",
        "lgbm",
        "xgb",
        "random_forest",
    ]

    # Standardised display names, kept in sync with plot_nested_cv.models_display
    # so a model reads identically across figures and the significance table.
    # Note "dense" is the fully-connected P-NET (P-NET-FC), not a dense single layer.
    models_display = {
        "pnet":                    "P-NET",
        "dense":                   "P-NET-FC",
        "decision_tree":           "Decision Tree",
        "adaboost":                "Ada. Boosting",
        "sgd_logistic_regression": "Logistic Reg.",
        "svc":                     "Linear SVM",
        "rbf_svm":                 "RBF SVM",
        "lgbm":                    "LightGBM",
        "xgb":                     "XGBoost",
        "random_forest":           "Random Forest",
    }

    def disp(m):
        return models_display.get(m, m)

    comparisons = [("pnet", m, f"{disp('pnet')} vs {disp(m)}") for m in baseline_models]
    model_names = ["pnet"] + baseline_models

    # each model stores one results.csv with per-outer-fold rows; the "test" rows
    # give the held-out fold scores we compare across models
    model_scores = {}
    for model_name in model_names:
        path = os.path.join(run_dir, model_name, "results.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, index_col=0)
        test_rows = df[df["index"] == "test"][[metric]]
        model_scores[model_name] = test_rows.reset_index(drop=True)

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
            "model1":     disp(model1),
            "model2":     disp(model2),
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
    print(f"Tissue type significance test results saved to {out_path}")
    print(sigresults.round(4).to_string())
