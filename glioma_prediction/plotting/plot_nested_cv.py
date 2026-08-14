"""
Per-fold distribution of each test metric across models, from the nested
crossvalidation. Models, display names and ordering come from models.py; the
drawing is shared with the other tasks in architecture/plotting/model_comparison.py.
"""

from architecture.plotting.model_comparison import load_per_fold_scores, plot_fold_distribution

from .models import BOUNDED_METRICS, METRICS, MODELS


def _filename(metric, prefix):
    # the ROC-AUC figure has always been named "auroc" rather than "auc"
    slug = "auroc" if metric == "auc" else metric
    return f"{prefix}_{slug}.pdf"


def plot_nested_CV(run_dir, figures_dir, selection_metric="auc"):
    combined = load_per_fold_scores(run_dir, MODELS, list(METRICS),
                                   selection_metric, strict=True)
    plot_fold_distribution(combined, MODELS, METRICS, figures_dir,
                           lambda m: _filename(m, "nested_CV"),
                           bounded_metrics=BOUNDED_METRICS)
