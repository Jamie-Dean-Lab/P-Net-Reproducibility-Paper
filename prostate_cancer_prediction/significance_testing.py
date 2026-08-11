from architecture.significance_utils import (
    build_comparisons,
    load_scores_from_summary,
    run_significance_tests,
)


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

    # Standardised display names, kept in sync with plot_nested_cv.models_display
    # so a model reads identically across figures and the significance table.
    # Here "dense_single_layer" IS a genuine dense single layer (distinct from the
    # fully-connected P-NET, "pnetfc" -> P-NET-FC).
    models_display = {
        "pnet":                    "P-NET",
        "pnet_GO":                 "P-NET-GO",
        "pnetfc":                  "P-NET-FC",
        "dense_single_layer":      "Dense Single Layer",
        "decision_tree":           "Decision Tree",
        "adaboost":                "Ada. Boosting",
        "linear_svm":              "Linear SVM",
        "random_forest":           "Random Forest",
        "rbf_svm":                 "RBF SVM",
        "sgd_logistic_regression": "Logistic Regression",
    }

    def disp(m):
        return models_display.get(m.replace("_nested_CV", ""), m)

    comparisons, model_names = build_comparisons(reference, baseline_models, disp)
    model_scores = load_scores_from_summary(
        run_dir, model_names, selection_metric, metric, col_prefix="response_"
    )
    run_significance_tests(model_scores, comparisons, metric, "Prostate", wd, disp)
