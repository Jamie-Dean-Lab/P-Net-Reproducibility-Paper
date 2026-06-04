"""Preprocessing for the prostate external-validation cohorts.

Builds the processed feature matrices and ``metastatic`` label files for the
Met500 (metastatic, label 1) and PRAD (primary, label 0) cohorts, derives a
combined dataset spanning both, and produces two class-balanced training splits
from the primary response labels.
"""

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import requests

ZENODO_DATABASE_URL = "https://zenodo.org/records/10775529/files/_database.zip?download=1"

# Met500 prostate samples that also appear in the primary cohort. This overlap
# is removed from the metastatic validation set to avoid train/test leakage.
COMMON_SAMPLES = [
    "MO_1008", "MO_1012", "MO_1013", "MO_1014", "MO_1015", "MO_1020", "MO_1040", "MO_1074",
    "MO_1084", "MO_1094", "MO_1095", "MO_1096", "MO_1114", "MO_1118", "MO_1124", "MO_1128",
    "MO_1130", "MO_1132", "MO_1139", "MO_1161", "MO_1162", "MO_1176", "MO_1179", "MO_1184",
    "MO_1192", "MO_1202", "MO_1215", "MO_1219", "MO_1232", "MO_1241", "MO_1244", "MO_1249",
    "MO_1262", "MO_1277", "MO_1316", "MO_1337", "MO_1339", "MO_1410", "MO_1421", "MO_1447",
    "MO_1460", "MO_1473",
    "TP_2001", "TP_2010", "TP_2020", "TP_2032", "TP_2034", "TP_2054", "TP_2060", "TP_2061",
    "TP_2064", "TP_2069", "TP_2077", "TP_2078", "TP_2079",
]

# All prostate samples present in the Met500 cohort.
MET500_PROSTATE_SAMPLES = [
    "MO_1008", "MO_1012", "MO_1013", "MO_1014", "MO_1015", "MO_1020", "MO_1040", "MO_1066",
    "MO_1074", "MO_1084", "MO_1093", "MO_1094", "MO_1095", "MO_1096", "MO_1112", "MO_1114",
    "MO_1118", "MO_1124", "MO_1128", "MO_1130", "MO_1132", "MO_1139", "MO_1161", "MO_1162",
    "MO_1176", "MO_1179", "MO_1184", "MO_1192", "MO_1200", "MO_1201", "MO_1202", "MO_1214",
    "MO_1215", "MO_1219", "MO_1221", "MO_1232", "MO_1240", "MO_1241", "MO_1244", "MO_1249",
    "MO_1260", "MO_1262", "MO_1263", "MO_1277", "MO_1307", "MO_1316", "MO_1336", "MO_1337",
    "MO_1339", "MO_1410", "MO_1420", "MO_1421", "MO_1437", "MO_1443", "MO_1446", "MO_1447",
    "MO_1460", "MO_1469", "MO_1472", "MO_1473", "MO_1482", "MO_1490", "MO_1492", "MO_1496",
    "MO_1499", "MO_1510", "MO_1511", "MO_1514", "MO_1517", "MO_1541", "MO_1543", "MO_1553",
    "MO_1556",
    "TP_2001", "TP_2009", "TP_2010", "TP_2020", "TP_2032", "TP_2034", "TP_2037", "TP_2043",
    "TP_2054", "TP_2060", "TP_2061", "TP_2064", "TP_2069", "TP_2077", "TP_2078", "TP_2079",
    "TP_2080", "TP_2081", "TP_2090", "TP_2093", "TP_2096", "TP_2156",
]


class Preprocessor:
    """Generates the external-validation datasets under a prostate data directory.

    ``data_dir`` is the prostate database directory containing the
    ``external_validation`` (Met500 / PRAD raw inputs) and ``processed``
    (``response_paper.csv``) subdirectories.
    """

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        # The _database archive (containing this prostate dir) extracts into the
        # task's top-level data directory, two levels up from data_dir.
        self.database_dir = self.data_dir.parent
        self.download_dir = self.database_dir.parent
        self.external_dir = self.data_dir / "external_validation"
        self.processed_dir = self.data_dir / "processed"
        self.met500_dir = self.external_dir / "Met500"
        self.prad_dir = self.external_dir / "PRAD"
        self.combined_dir = self.external_dir / "Combined"

    def download_data(self):
        """Download and extract the _database archive from Zenodo if not present."""
        if self.database_dir.exists():
            return
        self.download_dir.mkdir(parents=True, exist_ok=True)
        print("Downloading prostate _database archive from Zenodo...")
        response = requests.get(ZENODO_DATABASE_URL)
        response.raise_for_status()
        with ZipFile(BytesIO(response.content)) as archive:
            archive.extractall(self.download_dir)

    @staticmethod
    def _read_matrix(path, sep=","):
        """Read a feature matrix indexed by sample, filling missing values with 0."""
        return pd.read_csv(path, sep=sep, index_col=0).fillna(0)

    def generate_external_validation_labels(self):
        """Write processed feature matrices and metastatic labels for both cohorts."""
        # Met500: metastatic samples (label 1), excluding those shared with the primary cohort.
        met500_samples = sorted(set(MET500_PROSTATE_SAMPLES) - set(COMMON_SAMPLES))
        cnv = self._read_matrix(self.met500_dir / "Met500_cnv.txt", sep="\t")
        mut = self._read_matrix(self.met500_dir / "Met500_mut_matrix.csv")
        mut.index = mut.index.str.split(".").str[0]

        cnv.T.to_csv(self.met500_dir / "Met500_cnv_processed.csv")
        mut.to_csv(self.met500_dir / "Met500_mut_matrix_processed.csv")
        pd.DataFrame({"metastatic": 1}, index=met500_samples).to_csv(self.met500_dir / "Met500_labels.csv")

        # PRAD: primary samples (label 0).
        cnv = self._read_matrix(self.prad_dir / "cnv_matrix.csv")
        mut = self._read_matrix(self.prad_dir / "mut_matrix.csv")
        prad_samples = sorted(set(cnv.index) | set(mut.index))
        pd.DataFrame({"metastatic": 0}, index=prad_samples).to_csv(self.prad_dir / "PRAD_labels.csv")

    def generate_combined_external_validation_dataset(self):
        """Combine the Met500 (metastatic) and PRAD (primary) cohorts into single
        cnv/mut/labels files spanning both external validation datasets.

        Requires generate_external_validation_labels() to have run first, since it
        reads the processed per-cohort feature matrices and label files. The two
        cohorts use disjoint sample IDs, so rows are simply concatenated.
        """
        self.combined_dir.mkdir(parents=True, exist_ok=True)

        cnv = self._combine_matrices(
            self.met500_dir / "Met500_cnv_processed.csv",
            self.prad_dir / "cnv_matrix.csv",
        )
        mut = self._combine_matrices(
            self.met500_dir / "Met500_mut_matrix_processed.csv",
            self.prad_dir / "mut_matrix.csv",
        )
        labels = pd.concat([
            pd.read_csv(self.met500_dir / "Met500_labels.csv", index_col=0),
            pd.read_csv(self.prad_dir / "PRAD_labels.csv", index_col=0),
        ], axis=0)

        cnv.to_csv(self.combined_dir / "cnv_combined.csv")
        mut.to_csv(self.combined_dir / "mut_combined.csv")
        labels.to_csv(self.combined_dir / "labels_combined.csv")

    def _combine_matrices(self, met500_path, prad_path):
        """Stack two sample-indexed matrices row-wise, taking the union of columns.

        Genes present in only one cohort are filled with 0 for the other's samples.
        """
        met500 = self._read_matrix(met500_path)
        prad = self._read_matrix(prad_path)
        return pd.concat([met500, prad], axis=0).fillna(0)

    def get_balanced_training_sets(self):
        """Split the response labels into two class-balanced training sets.

        All positives are paired with the first and second halves of the negatives
        respectively, producing two complementary balanced cohorts.
        """
        df = pd.read_csv(self.processed_dir / "response_paper.csv", index_col=0)
        pos = np.where(df["response"] == 1)[0]
        neg = np.where(df["response"] == 0)[0]
        n_pos = pos.shape[0]
        neg_first, neg_second = neg[:n_pos], neg[n_pos:]

        df.iloc[np.concatenate([pos, neg_first])].to_csv(
            self.processed_dir / "response_paper_external_validation_1.csv")
        df.iloc[np.concatenate([pos, neg_second])].to_csv(
            self.processed_dir / "response_paper_external_validation_2.csv")

    def run_all(self):
        """Download source data (if needed) and run the full preprocessing pipeline."""
        self.download_data()
        self.generate_external_validation_labels()
        self.generate_combined_external_validation_dataset()
        self.get_balanced_training_sets()


if __name__ == "__main__":
    Preprocessor("prostate_cancer_prediction/data/_database/prostate").run_all()
