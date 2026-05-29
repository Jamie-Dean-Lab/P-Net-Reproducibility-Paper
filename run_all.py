from radiosensitivity_prediction.run import run as run_radiosensitivity
from tissue_type_classification.run import run as run_tissue
from prostate_cancer_prediction.run import run as run_prostate
from glioma_prediction.run import run as run_glioma


def run_all():
    """
    WARNING: Running this function will execute all models across all tasks
    (prostate cancer, radiosensitivity, tissue type, and glioma prediction)
    and is expected to take an extremely long time to complete.

    It is strongly advised to run individual task configs separately instead,
    e.g. via prostate_cancer_prediction/run.py, radiosensitivity_prediction/run.py,
    tissue_type_classification/run.py, or glioma_prediction/run.py.
    """
    run_prostate()
    run_radiosensitivity()
    run_tissue()
    run_glioma()


if __name__ == "__main__":
    run_all()
