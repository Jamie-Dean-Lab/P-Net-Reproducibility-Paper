from radiosensitivity_prediction.run import run as run_radiosensitivity
from tissue_type_classification.run import run as run_tissue
from prostate_cancer_prediction.run import run as run_prostate


def run_all():
    run_prostate()
    run_radiosensitivity()
    run_tissue()


if __name__ == "__main__":
    run_all()