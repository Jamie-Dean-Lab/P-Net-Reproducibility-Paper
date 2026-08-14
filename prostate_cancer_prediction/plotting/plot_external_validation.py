"""
External-validation scores, one bar per model.

Each bar is a single unreplicated fit, so there is no spread to show — see
Pipeline._run_external_validation, which trains one model at one seed.
"""

from architecture.plotting.model_comparison import load_single_values, plot_single_values

from .models import BOUNDED_METRICS, EXTERNAL, METRICS


def _filename(metric, prefix):
    # the ROC-AUC figure has always been named "auroc" rather than "auc"
    slug = "auroc" if metric == "auc" else metric
    return f"{prefix}_{slug}.pdf"


def plot_external_validation(run_dir, figures_dir, dataset_tag="combined"):
    scores = load_single_values(run_dir, EXTERNAL, list(METRICS), dataset_tag)
    plot_single_values(scores, EXTERNAL, METRICS, figures_dir,
                       lambda m: _filename(m, "external_validation"),
                       bounded_metrics=BOUNDED_METRICS)
