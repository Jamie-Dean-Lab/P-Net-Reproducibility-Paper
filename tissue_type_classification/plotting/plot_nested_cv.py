"""
Per-fold distribution of each test metric across models, from the nested
crossvalidation. Models, display names and ordering come from models.py; the
drawing is shared with the other tasks in architecture/plotting/model_comparison.py.
"""

from architecture.plotting.model_comparison import load_run_scores, plot_fold_distribution

from .models import BOUNDED_METRICS, METRICS, MODELS


def _filename(metric, prefix):
    return f"{prefix}_{metric}.pdf"


def plot_nested_CV(run_dir, figures_dir):
    combined = load_run_scores(run_dir, MODELS, list(METRICS))
    plot_fold_distribution(combined, MODELS, METRICS, figures_dir,
                           lambda m: _filename(m, "nested_CV"),
                           bounded_metrics=BOUNDED_METRICS)
