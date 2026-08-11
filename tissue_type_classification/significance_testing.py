from architecture.significance_utils import (
    build_comparisons,
    load_scores_from_results,
    run_significance_tests,
)


def significance_test(run_dir, wd):
    metric = "f1"  # weighted F1

    # P-NET is compared against every other model; display names match plot_nested_cv
    reference = "pnet"
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

    comparisons, model_names = build_comparisons(reference, baseline_models, disp)
    model_scores = load_scores_from_results(run_dir, model_names, metric)
    run_significance_tests(model_scores, comparisons, metric, "Tissue type", wd, disp)
