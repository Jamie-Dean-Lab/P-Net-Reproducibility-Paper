"""Shared significance-testing logic for the nested cross-validation model
comparisons (P-NET, versus each baseline).

Model scores come from k-fold cross validation, so the fold-wise differences
between two models are NOT independent: the k training sets overlap heavily,
which makes the naive paired t-test underestimate the variance and produce
anti-conservative p-values. We therefore use the Nadeau & Bengio (2003)
corrected resampled t-test, which replaces the ``1/n`` variance factor with
``(1/n + n_test/n_train)``. For standard k-fold CV ``n_test/n_train = 1/(k-1)``,
which is the default here (``rho=None``).
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import t
from statsmodels.stats.multitest import multipletests


def corrected_paired_stats(scores1, scores2, rho=None, alpha=0.05):
    """Nadeau & Bengio corrected resampled paired t-test for two models scored
    on the same cross-validation folds.

    args:
        scores1, scores2 : per-fold scores for the two models (same fold order)
        rho : n_test/n_train ratio for the variance correction. When None,
              assumes standard k-fold CV and uses 1/(k-1) with k = n folds.
        alpha : two-sided significance level used for the confidence interval.

    returns:
        dict with the test statistic, degrees of freedom, mean difference,
        corrected standard error, confidence interval, Cohen's d, and raw
        (uncorrected for multiplicity) p-value.
    """
    diffs = np.asarray(scores1, dtype=float) - np.asarray(scores2, dtype=float)
    n = len(diffs)
    df_deg = n - 1
    if rho is None:
        rho = 1.0 / (n - 1)  # standard k-fold CV: (1/k) test vs (1 - 1/k) train

    mean_diff = float(np.mean(diffs))
    sd = float(np.std(diffs, ddof=1))
    corrected_se = float(np.sqrt((1.0 / n + rho) * sd ** 2))
    t_stat = mean_diff / corrected_se
    p_raw = float(2.0 * t.sf(abs(t_stat), df_deg))
    t_crit = float(t.ppf(1 - alpha / 2, df_deg))

    return {
        "t":         t_stat,
        "df":        df_deg,
        "mean_diff": mean_diff,
        "sem":       corrected_se,
        "ci_low":    mean_diff - t_crit * corrected_se,
        "ci_high":   mean_diff + t_crit * corrected_se,
        "cohens_d":  mean_diff / sd,
        "p_raw":     p_raw,
    }


def build_comparisons(reference, baseline_models, disp):
    """Build the (reference vs baseline) comparison list and the ordered list of
    model directory names to load."""
    comparisons = [(reference, m, f"{disp(reference)} vs {disp(m)}") for m in baseline_models]
    model_names = [reference] + baseline_models
    return comparisons, model_names


def load_scores_from_summary(run_dir, model_names, selection_metric, metric, col_prefix=""):
    """Load per-fold test scores from ``<model>/test_*/best_<selection_metric>/
    summary_results.csv`` (used by the glioma, prostate, and radiosensitivity
    tasks). ``col_prefix`` is stripped from column names before selecting
    ``metric``."""
    model_scores = {}
    for model_name in model_names:
        model_dir = os.path.join(run_dir, model_name)
        if not os.path.isdir(model_dir):
            continue
        fold_scores = []
        for test_dir in sorted(d for d in os.listdir(model_dir) if d.startswith("test_")):
            path = os.path.join(model_dir, test_dir, f"best_{selection_metric}", "summary_results.csv")
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path, index_col=0)
            if col_prefix:
                df.columns = [c.replace(col_prefix, "") for c in df.columns]
            fold_scores.append(df[df.index == "test"][[metric]])
        if fold_scores:
            model_scores[model_name] = pd.concat(fold_scores).reset_index(drop=True)
    return model_scores


def load_scores_from_results(run_dir, model_names, metric):
    """Load per-fold test scores from a single ``<model>/results.csv`` whose
    ``index`` column marks the held-out ``test`` rows (used by the tissue-type
    task)."""
    model_scores = {}
    for model_name in model_names:
        path = os.path.join(run_dir, model_name, "results.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, index_col=0)
        model_scores[model_name] = df[df["index"] == "test"][[metric]].reset_index(drop=True)
    return model_scores


def run_significance_tests(model_scores, comparisons, metric, dataset_name, wd, disp,
                           rho=None, alpha=0.05):
    """Run the corrected paired test for each available comparison, apply
    Benjamini-Hochberg FDR across comparisons, then save and print the table."""
    comparisons = [(m1, m2, name) for m1, m2, name in comparisons
                   if m1 in model_scores and m2 in model_scores]

    results_full = []
    for model1, model2, comp_name in comparisons:
        stats = corrected_paired_stats(
            model_scores[model1][metric].values,
            model_scores[model2][metric].values,
            rho=rho, alpha=alpha,
        )
        results_full.append({
            "comparison": comp_name,
            "model1":     disp(model1),
            "model2":     disp(model2),
            "metric":     metric,
            **stats,
        })

    rejected, p_fdr, _, _ = multipletests(
        [r["p_raw"] for r in results_full], alpha=alpha, method="fdr_bh"
    )

    sigresults = pd.DataFrame(results_full)
    sigresults["p_fdr"] = p_fdr
    sigresults["significant"] = rejected

    out_path = os.path.join(wd, "significance_tests.csv")
    sigresults.to_csv(out_path, float_format="%.4f")
    print(f"{dataset_name} significance test results saved to {out_path}")
    print(sigresults.round(4).to_string())
    return sigresults
