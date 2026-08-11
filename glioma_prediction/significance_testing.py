from architecture.significance_utils import (
    build_comparisons,
    load_scores_from_summary,
    run_significance_tests,
)


def significance_test(run_dir, wd, selection_metric="auc"):
    metric = "auc"

    # P-NET is compared against every other model; names match plot_nested_cv
    reference = "pnet"
    baseline_models = [
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

    # Standardised display names, kept in sync with plot_nested_cv.models_display
    # so a model reads identically across figures and the significance table.
    # Note "dense" is the fully-connected P-NET (P-NET-FC), not a dense single layer.
    models_display = {
        "pnet":                    "P-NET",
        "pnet_GO":                 "P-NET-GO",
        "dense":                   "P-NET-FC",
        "decision_tree":           "Decision Tree",
        "adaboost":                "Ada. Boosting",
        "svc":                     "Linear SVM",
        "random_forest":           "Random Forest",
        "rbf_svm":                 "RBF SVM",
        "xgb":                     "XGBoost",
        "lgbm":                    "LightGBM",
        "sgd_logistic_regression": "Logistic Regression",
    }

    def disp(m):
        return models_display.get(m, m)

    comparisons, model_names = build_comparisons(reference, baseline_models, disp)
    model_scores = load_scores_from_summary(
        run_dir, model_names, selection_metric, metric, col_prefix="response_"
    )
    run_significance_tests(model_scores, comparisons, metric, "Glioma", wd, disp)
