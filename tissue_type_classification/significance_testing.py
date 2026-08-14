"""
P-NET versus every baseline on the nested-crossvalidation test folds.

Models and display names come from plotting/models.py — the same registry the
figures use — so a model reads identically in the table and in the plots.
"""

from architecture.significance_utils import (
    build_comparisons,
    load_scores_from_results,
    run_significance_tests,
)

from .plotting.models import MODELS, REFERENCE

# The metric the comparison is reported on.
METRIC = 'f1'


def significance_test(run_dir, wd):
    baselines = [m for m in MODELS.names if m != REFERENCE]
    comparisons, model_names = build_comparisons(REFERENCE, baselines, MODELS.label)
    model_scores = load_scores_from_results(run_dir, model_names, METRIC)
    run_significance_tests(model_scores, comparisons, METRIC, 'Tissue type', wd, MODELS.label)
