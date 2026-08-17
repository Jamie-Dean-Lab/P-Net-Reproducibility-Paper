import subprocess
import traceback
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.integrate import quad

_ZENODO_URL = "https://zenodo.org/records/21979483/files/rs_data.zip"

# HCC56_LARGE_INTESTINE is contaminated and has been intentionally excluded from the model list.

# IDs missing or mis-formatted in the DepMap model list; verified against Cellosaurus.
_CCLE_ID_PATCHES = {
    "SIDM00037": "ISHIKAWAHERAKLIO02ER_ENDOMETRIUM",
    "SIDM01095": "RH30_SOFT_TISSUE",
    "SIDM01424": "SEM_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE",
    "SIDM00969": "DOV13_OVARY",
    "SIDM00096": "SR786_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE",
    "SIDM00035": "S117_THYROID",
    # Tissue-of-origin ambiguity; assigned manually
    "SIDM02102": "NCIH854_LUNG",
    "SIDM01665": "NCIH684_LIVER",
    "SIDM01505": "DM3_FIBROBLAST",
    "SIDM01592": "HS274T_FIBROBLAST",
}

_BROAD_ID_PATCHES = {
    "SIDM00104": "ACH-002123",
    "SIDM00517": "ACH-002239",
    "SIDM00069": "ACH-002024",
    "SIDM01845": "ACH-002014",
    "SIDM00096": "ACH-000338",
    "SIDM01095": "ACH-000833",
    "SIDM00246": "ACH-002084",
    "SIDM00619": "ACH-002397",
    "SIDM00518": "ACH-002136",
    "SIDM00969": "ACH-001063",
    "SIDM00603": "ACH-001543",
}


class Preprocessor:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.remove_overlapping_cell_lines = True
        self._model_annotations = None

    @property
    def model_annotations(self):
        if self._model_annotations is None:
            self._model_annotations = self.get_annotations()
        return self._model_annotations

    @property
    def ccle_id_to_sanger_mappings(self):
        return self.model_annotations.set_index('CCLE_ID')['model_id'].to_dict()

    @property
    def broad_id_to_sanger_mappings(self):
        return self.model_annotations.set_index('BROAD_ID')['model_id'].to_dict()

    # --- Shared utilities ---

    def get_annotations(self):
        df = pd.read_csv(self.data_dir / "model_list_20250630.csv")
        df = df[df["model_type"] == "Cell Line"]

        for model_id, ccle_id in _CCLE_ID_PATCHES.items():
            df.loc[df["model_id"] == model_id, "CCLE_ID"] = ccle_id
        for model_id, broad_id in _BROAD_ID_PATCHES.items():
            df.loc[df["model_id"] == model_id, "BROAD_ID"] = broad_id

        return df[["model_id", "model_name", "synonyms", "tissue", "CCLE_ID", "RRID", "BROAD_ID"]].copy()

    @staticmethod
    def clean_string(text):
        return ''.join(char.upper() for char in text if char.isalnum())

    def map_to_model_id(self, df, model_annotations):
        model_annotations = model_annotations.copy()
        model_annotations["model_name_clean"] = model_annotations["model_name"].apply(self.clean_string)

        def clean_synonyms(syn_str):
            if pd.isna(syn_str):
                return []
            return [self.clean_string(s.strip()) for s in syn_str.split(";")]

        model_annotations["synonyms_clean"] = model_annotations["synonyms"].apply(clean_synonyms)

        def find_model_id(value):
            value_clean = self.clean_string(value)
            if not value_clean:
                return value

            matches = []
            name_matches = model_annotations[model_annotations["model_name_clean"] == value_clean]
            matches.extend(name_matches["model_id"].tolist())

            for _, row in model_annotations.iterrows():
                if value_clean in row["synonyms_clean"]:
                    matches.append(row["model_id"])

            matches = list(dict.fromkeys(matches))

            if len(matches) == 0:
                print(f"No match found for: {value}")
                return value
            elif len(matches) == 1:
                return matches[0]
            else:
                print(f"Multiple matches found for '{value}': {matches}")
                return value

        df = df.copy()
        df.index = [find_model_id(idx) for idx in df.index]
        return df

    def map_ids_to_sanger(self, df, mapping):
        idx = df.index.to_series()
        mapped = idx.map(mapping)
        unmapped_ids = idx[mapped.isna()]
        print(f"ID mapping: {mapped.notna().sum()}/{len(idx)} mapped successfully")
        if not unmapped_ids.empty:
            print("Unmapped IDs (will be dropped):")
            print(unmapped_ids.unique())

        df = df[mapped.notna()]
        mapped = mapped[mapped.notna()]
        df.index = mapped
        df.index.name = "cell_line"
        return df

    def sanitise(self, df):
        df = df.dropna(axis=0, how='all')
        df = df.dropna(axis=1, how='any')

        nunique = df.nunique(dropna=False)
        df = df.loc[:, nunique > 1]

        dup_rows = df.index.duplicated().any()
        dup_cols = df.columns.duplicated().any()
        if dup_rows or dup_cols:
            msg_parts = []
            if dup_rows:
                msg_parts.append("duplicate row (index) names")
            if dup_cols:
                msg_parts.append("duplicate column names")
            print("Sanitised dataframe still has " + " and ".join(msg_parts))

        n_rows, n_cols = df.shape
        print(f"Dataframe contains {n_rows} cell lines and {n_cols} features")
        return df

    def get_cleveland_cell_lines(self):
        df = pd.read_csv(self.data_dir / "cleveland_auc_preprocessed.csv")
        return set(df[df.columns[0]])

    def read_gencode_gtf(self) -> pd.DataFrame:
        """Parse gene_id and gene_name from a Gencode GTF; returns one row per gene."""
        rows = []
        with open(self.data_dir / "gencode.v19.genes.v7_model.patched_contigs.gtf") as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                fields = line.rstrip("\n").split("\t")
                if fields[2] != "gene":
                    continue
                attrs = fields[8]
                gene_id = gene_name = None
                for token in attrs.split(";"):
                    token = token.strip()
                    if token.startswith("gene_id"):
                        gene_id = token.split('"')[1]
                    elif token.startswith("gene_name"):
                        gene_name = token.split('"')[1]
                rows.append({"gene_id": gene_id, "gene_name": gene_name})

        df = pd.DataFrame(rows).drop_duplicates()
        # Strip version numbers so IDs match the CCLE RNA-seq index
        df["gene_id_base"] = df["gene_id"].str.split(".").str[0]
        print(f"GTF: {len(df)} genes, {df['gene_name'].nunique()} unique gene names")
        return df

    # --- Data download ---

    def download_zenodo(self):
        zip_path = self.data_dir / "rs_data.zip"
        if not zip_path.exists():
            print(f"Downloading {_ZENODO_URL} ...")
            urllib.request.urlretrieve(_ZENODO_URL, zip_path)
            print("Download complete.")
        else:
            print(f"  {zip_path.name} already exists, skipping download.")
        print("Extracting rs_data.zip ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.infolist():
                # strip the top-level directory from each path before extracting
                parts = Path(member.filename).parts
                stripped = Path(*parts[1:]) if len(parts) > 1 else None
                if stripped is None or str(stripped) == ".":
                    continue
                dest = self.data_dir / stripped
                if member.is_dir():
                    dest.mkdir(parents=True, exist_ok=True)
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(member) as src, open(dest, "wb") as out:
                        out.write(src.read())
        print("Extraction complete.")

    # --- Radiation response (target variable) ---

    def download_cleveland(self):
        print("Downloading Cleveland radiation response data...")
        r_script = Path(__file__).parent / "download_cleveland.R"
        result = subprocess.call(["Rscript", str(r_script), str(self.data_dir)], shell=False)
        if result != 0:
            raise RuntimeError(f"Rscript exited with code {result}")

    def cleveland(self):
        print("Preprocessing Cleveland data...")
        df = pd.read_csv(self.data_dir / "cleveland.csv")

        # Drop cell lines absent from the DepMap model list
        df = df[~df['cell_line'].isin(["BT112", "BT145", "BT181", "BT189", "BT216", "BT228", "HCC-56", "NCI-H322"])]
        df = df.set_index('cell_line')
        df = self.map_to_model_id(df, self.model_annotations)

        # Manually map synonyms not present in the model list
        print("Manually mapping failed matches...")
        df.index = df.index.to_series().replace({
            "Hs 729": "SIDM01560",
            "ML-1":   "SIDM00442",
            "T.T":    "SIDM00322",
            "WM-793": "SIDM00973",
        })

        df = df[df["Rsquare"] >= 0.6]
        df['AUC_log1p'] = np.log1p(df['AUC'])
        df = df[['AUC_log1p']]
        print(f"Final dataset has {df.shape[0]} samples (cell lines)")
        df.to_csv(self.data_dir / "cleveland_auc_preprocessed.csv", index=True)

    @staticmethod
    def LQModel(D, alpha, beta):
        return np.exp(-alpha * D - beta * D**2)

    @staticmethod
    def compute_lq_auc(alpha, beta, lower, upper):
        result, _ = quad(Preprocessor.LQModel, lower, upper, args=(alpha, beta))
        return result

    def nci60_auc(self):
        print("Preprocessing NCI-60 data...")
        df = pd.read_csv(self.data_dir / "nci60_auc.csv")

        # Drop cell lines absent from the DepMap model list
        df = df[~df["cell_line"].isin(["NH32", "TK6", "MDA-N"])]
        df = df.set_index('cell_line')
        df = self.map_to_model_id(df, self.model_annotations)

        # Manually map synonyms not present in the model list
        df.index = df.index.to_series().replace({
            "ML-1":     "SIDM00440",
            "LOX-IVMI": "SIDM00149",
            "NCI-ADR":  "SIDM00089",
        })

        df["auc_dose_range_2_8"] = df.apply(
            lambda row: self.compute_lq_auc(row["alpha"], row["beta"], lower=2, upper=8), axis=1
        )
        df["auc_dose_range_1_10"] = df.apply(
            lambda row: self.compute_lq_auc(row["alpha"], row["beta"], lower=1, upper=10), axis=1
        )
        df['auc_dose_range_1_10_log1p'] = np.log1p(df['auc_dose_range_1_10'])

        df = df[df["Rsquare"] >= 0.6]

        if self.remove_overlapping_cell_lines:
            cleveland_cell_lines = self.get_cleveland_cell_lines()
            overlap = sorted(set(df.index) & cleveland_cell_lines)
            print(f"Removing {len(overlap)} overlapping cell lines with Cleveland: {overlap}")
            df = df[~df.index.isin(overlap)]

        df = df[['auc_dose_range_1_10_log1p']]
        print(f"Final dataset has {df.shape[0]} samples (cell lines)")
        df.to_csv(self.data_dir / "nci60_auc_preprocessed.csv", index=True)

    # --- Omics features ---

    def ccle_dna_methylation(self):
        print("Preprocessing methylation data...")
        df = pd.read_csv(
            self.data_dir / 'CCLE_RRBS_TSS1kb_20181022.txt',
            sep="\t",
            low_memory=False
        )

        df["locus_id"] = df["locus_id"].apply(lambda x: x.split("_")[0])
        df = df.drop(df.columns[1:3], axis=1)

        value_cols = df.columns.difference(["locus_id"])
        df[value_cols] = df[value_cols].apply(pd.to_numeric, errors="coerce")
        df = df.groupby("locus_id")[value_cols].mean()
        df = df.T

        df = self.map_ids_to_sanger(df, self.ccle_id_to_sanger_mappings)
        df = self.sanitise(df)
        df.to_csv(self.data_dir / 'methylation_preprocessed.csv', index=True)

    def ccle_rna_seq(self):
        print("Preprocessing CCLE RNA-seq data...")
        df = pd.read_csv(
            self.data_dir / "CCLE_RNAseq_rsem_genes_tpm_20180929.txt",
            sep="\t",
            index_col=0,
            low_memory=False,
        )

        df = df.drop(columns=["transcript_ids"])

        # Strip Ensembl version numbers and map to gene symbols via GTF
        df.index = df.index.str.split(".").str[0]
        gtf = self.read_gencode_gtf()
        ensembl_to_gene = gtf.set_index("gene_id_base")["gene_name"].to_dict()
        df.index = df.index.map(ensembl_to_gene)

        unmapped = df.index.isna().sum()
        if unmapped:
            print(f"Dropping {unmapped} genes with no gene name in GTF")
        df = df[df.index.notna()]

        # Melt to long format and check for duplicates, mirroring pivot_data
        df.index.name = "gene_symbol"
        df_long = df.reset_index().melt(id_vars="gene_symbol", var_name="model_id", value_name="tpm")

        duplicates = df_long.groupby(['model_id', 'gene_symbol']).size()
        num_duplicates = (duplicates > 1).sum()

        if num_duplicates > 0:
            print(f"  Found {num_duplicates} duplicate model_id + gene_symbol combinations")
            print(f"  Aggregating duplicates by taking the SUM of TPM values")
            df_long = (
                df_long
                .groupby(['model_id', 'gene_symbol'], as_index=False)['tpm']
                .sum()
            )

        df = df_long.pivot(index='model_id', columns='gene_symbol', values='tpm')
        df.index.name = 'cell_line'

        # Log-transform raw TPM values
        df = np.log1p(df)
        print(f"TPM value range after log1p: min={df.values.min():.3f}, max={df.values.max():.3f}")

        df = self.map_ids_to_sanger(df, self.ccle_id_to_sanger_mappings)
        df = self.sanitise(df)
        df.to_csv(self.data_dir / "ccle_gene_expression_preprocessed.csv", index=True)

    def nci60_rnaseq(self):
        print("Preprocessing NCI60 RNA-seq data...")
        df = pd.read_excel(
            self.data_dir / "RNA__RNA_seq_composite_expression.xls",
            sheet_name=0,
            skiprows=10
        )
        df = df.drop(df.columns[[1, 2, 3, 4, 5]], axis=1)
        df = df.T

        df.columns = df.iloc[0]
        df = df.iloc[1:]
        df.index = [idx.split(":")[1] for idx in df.index]
        df.index.name = 'cell_line'

        df = df.drop(['HL-60(TB)', 'MDA-N'], axis=0)
        df = self.map_to_model_id(df, self.model_annotations)
        df.index.name = 'cell_line'

        numeric = df.apply(pd.to_numeric, errors='coerce')
        print(f"NCI-60 RNA-seq value range: min={numeric.values.min():.3f}, max={numeric.values.max():.3f}, mean={numeric.values.mean():.3f}")

        # Note no log transform, these are already log2(FPKM + 1)
        df = self.sanitise(df)
        df.to_csv(self.data_dir / "nci60_rnaseq_preprocessed.csv")

    def nci60_methylation(self):
        print("Preprocessing NCI60 methylation data...")
        df = pd.read_excel(
            self.data_dir / "DNA__Illumina_450K_methylation_Gene_average.xls",
            sheet_name=0,
            skiprows=10,
            na_values=['-']
        )
        df = df.drop(df.columns[[1, 2, 3, 4, 5]], axis=1)

        # Aggregate at gene level — no protein-level mapping available between datasets
        df = df.groupby("Gene name d", as_index=False).mean()
        df = df.T
        df.columns = df.iloc[0]
        df = df.iloc[1:]
        df.index = [idx.split(":")[1] for idx in df.index]
        df.index.name = 'cell_line'

        df = df.drop(['HL-60(TB)', 'MDA-N'], axis=0)
        df = self.map_to_model_id(df, self.model_annotations)
        df.index.name = 'cell_line'
        df = self.sanitise(df)
        df.to_csv(self.data_dir / "nci60_methylation_preprocessed.csv")

    # --- Feature alignment ---

    def align_rnaseq(self):
        self.align_datasets("ccle_gene_expression_preprocessed.csv", "nci60_rnaseq_preprocessed.csv")

    def align_methylation(self):
        self.align_datasets("methylation_preprocessed.csv", "nci60_methylation_preprocessed.csv")

    def align_datasets(self, discovery_name: str, validation_name: str):
        print(f"Aligning {discovery_name} and {validation_name}")

        df_disc = pd.read_csv(self.data_dir / discovery_name)
        df_valid = pd.read_csv(self.data_dir / validation_name)

        cell_col_disc = df_disc.columns[0]
        cell_col_valid = df_valid.columns[0]

        def split_synonyms(col: str) -> set[str]:
            return {c.strip() for c in str(col).split(";") if c.strip()}

        disc_ids = {col: split_synonyms(col) for col in df_disc.columns[1:]}
        valid_ids = {col: split_synonyms(col) for col in df_valid.columns[1:]}

        disc_to_canon = {}
        valid_to_canon = {}
        for dcol, dids in disc_ids.items():
            for vcol, vids in valid_ids.items():
                overlap = dids & vids
                if overlap:
                    canon = sorted(overlap)[0]
                    disc_to_canon[dcol] = canon
                    valid_to_canon[vcol] = canon
                    break

        common_genes = sorted(set(disc_to_canon.values()) & set(valid_to_canon.values()))
        print(f"Found {len(common_genes)} common features")

        disc_series = {}
        valid_series = {}
        for canon in common_genes:
            dcols = [c for c, c_id in disc_to_canon.items() if c_id == canon]
            vcols = [c for c, c_id in valid_to_canon.items() if c_id == canon]
            if dcols:
                disc_series[canon] = df_disc[dcols[0]]
            if vcols:
                valid_series[canon] = df_valid[vcols[0]]

        df_disc_h = pd.concat([df_disc[[cell_col_disc]], pd.DataFrame(disc_series)], axis=1)
        df_valid_h = pd.concat([df_valid[[cell_col_valid]], pd.DataFrame(valid_series)], axis=1)

        if self.remove_overlapping_cell_lines:
            cleveland_cell_lines = self.get_cleveland_cell_lines()
            overlap = sorted(set(df_valid_h[cell_col_valid]) & cleveland_cell_lines)
            print(f"Removing {len(overlap)} overlapping cell lines from NCI60: {overlap}")
            df_valid_h = df_valid_h[~df_valid_h[cell_col_valid].isin(overlap)]

        print(f"  Discovery after alignment:  {df_disc_h.shape[0]} cell lines, {df_disc_h.shape[1] - 1} features")
        print(f"  Validation after alignment: {df_valid_h.shape[0]} cell lines, {df_valid_h.shape[1] - 1} features")

        # Overwrite the preprocessed files in place — the preprocessed files are the final output
        df_disc_h.to_csv(self.data_dir / discovery_name, index=False)
        df_valid_h.to_csv(self.data_dir / validation_name, index=False)

    def print_preprocessing_summary(self):
        files = {
            "cleveland_auc":    "cleveland_auc_preprocessed.csv",
            "gene_expression":  "ccle_gene_expression_preprocessed.csv",
            "methylation":      "methylation_preprocessed.csv",
        }

        cell_line_sets = {}
        print("\n--- Preprocessing summary ---")
        for name, filename in files.items():
            path = self.data_dir / filename
            if not path.exists():
                print(f"  {name}: file not found ({filename})")
                continue
            df = pd.read_csv(path, index_col=0)
            n_cell_lines = df.shape[0]
            n_features = df.shape[1]
            print(f"  {name}: {n_cell_lines} cell lines, {n_features} features")
            cell_line_sets[name] = set(df.index)

        if len(cell_line_sets) == len(files):
            intersection = set.intersection(*cell_line_sets.values())
            print(f"  Intersection of all 3 files: {len(intersection)} cell lines")
        print("-----------------------------\n")

    # --- Pipeline ---

    def run_all(self):
        print("Starting preprocessing...")
        steps = [
            ("download_zenodo",       self.download_zenodo),
            ("download_cleveland",    self.download_cleveland),
            ("cleveland",             self.cleveland),
            ("ccle_dna_methylation",  self.ccle_dna_methylation),
            ("ccle_rna_seq",          self.ccle_rna_seq),
            ("nci60_auc",             self.nci60_auc),
            ("nci60_methylation",     self.nci60_methylation),
            ("nci60_rnaseq",          self.nci60_rnaseq),
            ("align_methylation", self.align_methylation),
            ("align_rnaseq",      self.align_rnaseq),
        ]
        for name, func in steps:
            print(f"Running {name}...")
            try:
                func()
                print(f"{name}: finished.")
            except Exception as e:
                print(f"{name}: FAILED – {e}")
                traceback.print_exc()
        print("Preprocessing pipeline finished.")
        self.print_preprocessing_summary()


if __name__ == "__main__":
    preprocessor = Preprocessor("data")
    preprocessor.run_all()
