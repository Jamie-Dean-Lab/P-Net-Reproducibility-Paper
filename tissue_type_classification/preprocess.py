import joblib
import traceback
import zipfile

import numpy as np
import pandas as pd
import requests
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder

_ZENODO_URL = "https://zenodo.org/records/20829764/files/data_tt.zip"

_EXPRESSION_GCT = "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct"
_METADATA_TSV = "GTEx_Analysis_v8_Annotations_SampleAttributesDS.tsv"
_HPA_TSV = "rna_tissue_hpa.tsv"
_TRANSCRIPT_RNA_TSV = "transcript_rna_tissue.tsv"


class Preprocessor:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # --- Data download ---

    def download_zenodo(self):
        zip_path = self.data_dir / "data_tt.zip"
        if not zip_path.exists():
            print(f"Downloading {_ZENODO_URL} ...")
            with open(zip_path, "wb") as f:
                f.write(requests.get(_ZENODO_URL).content)
            print("Download complete.")
        else:
            print(f"  {zip_path.name} already exists, skipping download.")
        print("Extracting data_tt.zip ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.infolist():
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

    # --- GTEx ---

    def process_gtex_expression(self):
        print("Processing GTEx expression data...")
        df = pd.read_csv(
            self.data_dir / _EXPRESSION_GCT,
            sep="\t", skiprows=2, index_col=0,
        )
        df = df.drop("Description", axis=1).T
        df.columns = df.columns.str.split(".").str[0]
        df = np.log2(df + 1)
        df = df.sample(df.shape[0], replace=False, random_state=42)

        chunk = df.iloc[:1000]
        chunk.to_csv(self.data_dir / "GTEx_gene_log2_tpm_0.csv", index=True)

        # Save sample order for the saved chunk only so process_gtex_tissue_labels can align to it
        pd.Series(chunk.index, name="sample_id").to_csv(
            self.data_dir / "sample_order.csv", index=False
        )
        print(f"Saved first 1000 samples, {df.shape[1]} genes.")

    def process_gtex_tissue_labels(self):
        print("Processing GTEx tissue labels...")
        sample_order = pd.read_csv(self.data_dir / "sample_order.csv")["sample_id"]

        info = pd.read_csv(self.data_dir / _METADATA_TSV, sep="\t", index_col=0)
        info = info[["SMTSD"]].loc[sample_order]

        encoder = OneHotEncoder()
        encoded = encoder.fit_transform(info)
        info = pd.DataFrame(encoded.todense(), index=info.index, columns=encoder.categories_[0])
        info.to_csv(self.data_dir / "GTEx_tissue_classes_encoded.csv", index=True)
        joblib.dump(encoder, self.data_dir / "tissue_encoder.pkl")
        pd.Series(encoder.categories_[0], name="tissue").to_csv(
            self.data_dir / "GTEx_tissue_classes.csv", index=False
        )
        print(f"Tissue labels saved: {info.shape[0]} samples, {info.shape[1]} classes.")

    # --- Human Protein Atlas ---

    def process_transcript_rna_tissue(self):
        print("Processing transcript RNA tissue data...")
        df = pd.read_csv(self.data_dir / _TRANSCRIPT_RNA_TSV)

        tpm_cols = [c for c in df.columns if c.startswith("TPM.")]

        # Extract numeric sample ID and tissue name from column ("TPM.adipose tissue.86" → id="86", tissue="adipose tissue")
        sample_ids = [col.rsplit(".", 1)[1] for col in tpm_cols]
        tissues = [col[4:].rsplit(".", 1)[0] for col in tpm_cols]
        sample_tissue = pd.Series(tissues, index=sample_ids, name="tissue")
        sample_tissue.index.name = "sample_id"
        sample_tissue.to_csv(self.data_dir / "hpa_sample_tissue.csv")
        print(f"Sample-tissue mapping saved: {len(sample_tissue)} samples.")

        # Sum transcript TPMs to gene level per sample, then log2-transform
        gene_df = df.groupby("ensgid")[tpm_cols].sum()
        gene_df = np.log2(gene_df + 1)

        # Pivot to (sample × gene) matrix, using numeric sample IDs as index
        gene_df = gene_df.T
        gene_df.index = sample_ids
        gene_df.index.name = "sample_id"
        gene_df.columns.name = None

        gene_df.to_csv(self.data_dir / "hpa_expression_preprocessed.csv", index=True)
        print(f"Transcript RNA expression saved: {gene_df.shape[0]} samples, {gene_df.shape[1]} genes.")

    def process_hpa_tissue_labels(self):
        mapping = pd.read_csv(self.data_dir / "tissue_mappings.csv")
        hpa_to_gtex = dict(zip(mapping["hpa_tissue"], mapping["gtex_tissue"]))

        hpa = pd.read_csv(self.data_dir / "hpa_expression_preprocessed.csv", index_col=0)
        sample_tissue = pd.read_csv(self.data_dir / "hpa_sample_tissue.csv", index_col=0)

        sample_tissue["gtex_tissue"] = sample_tissue["tissue"].map(hpa_to_gtex)
        keep = sample_tissue["gtex_tissue"].notna()
        dropped_tissues = sorted(sample_tissue.loc[~keep, "tissue"].unique())
        if dropped_tissues:
            print(f"Dropping samples from {len(dropped_tissues)} unmapped tissues: {dropped_tissues}")

        kept_samples = sample_tissue[keep].index
        hpa = hpa.loc[kept_samples].copy()
        hpa.to_csv(self.data_dir / "hpa_expression_preprocessed.csv", index=True)

        gtex_tissues = sample_tissue.loc[kept_samples, "gtex_tissue"]
        encoder = joblib.load(self.data_dir / "tissue_encoder.pkl")
        encoded = encoder.transform(gtex_tissues.to_frame().values)
        labels = pd.DataFrame(encoded.todense(), index=hpa.index, columns=encoder.categories_[0])
        labels.to_csv(self.data_dir / "hpa_tissue_classes_encoded.csv", index=True)
        print(f"HPA tissue labels saved: {labels.shape[0]} samples, {labels.shape[1]} classes.")
        print(f"Samples per tissue:\n{gtex_tissues.value_counts().sort_index()}")

    # --- Shared ---

    def intersect_genes(self):
        print("Intersecting genes between GTEx and HPA...")
        gtex = pd.read_csv(self.data_dir / "GTEx_gene_log2_tpm_0.csv", index_col=0)
        hpa = pd.read_csv(self.data_dir / "hpa_expression_preprocessed.csv", index_col=0)

        common_ensg = sorted(set(gtex.columns) & set(hpa.columns))
        print(f"GTEx genes: {len(gtex.columns)}, HPA genes: {len(hpa.columns)}, common: {len(common_ensg)}")

        ensg_to_symbol = (
            pd.read_csv(self.data_dir / _HPA_TSV)[["Gene", "Gene name"]]
            .drop_duplicates("Gene")
            .set_index("Gene")["Gene name"]
            .to_dict()
        )
        unmapped = [g for g in common_ensg if g not in ensg_to_symbol]
        if unmapped:
            print(f"Warning: {len(unmapped)} ENSG IDs have no symbol mapping and will be dropped.")
        common_ensg = [g for g in common_ensg if g in ensg_to_symbol]
        symbol_columns = [ensg_to_symbol[g] for g in common_ensg]

        gtex[common_ensg].set_axis(symbol_columns, axis=1).to_csv(self.data_dir / "GTEx_gene_expression_preprocessed.csv",
                                                                  index=True)
        hpa[common_ensg].set_axis(symbol_columns, axis=1).to_csv(self.data_dir / "hpa_gene_expression_preprocessed.csv",
                                                                 index=True)

        print(f"Both datasets saved with {len(symbol_columns)} gene symbol columns.")

    # --- Pipeline ---

    def run_all(self):
        print("Starting preprocessing...")
        steps = [
            ("download_zenodo", self.download_zenodo),
            ("process_gtex_expression", self.process_gtex_expression),
            ("process_gtex_tissue_labels", self.process_gtex_tissue_labels),
            ("process_transcript_rna_tissue", self.process_transcript_rna_tissue),
            ("process_hpa_tissue_labels", self.process_hpa_tissue_labels),
            ("intersect_genes", self.intersect_genes),
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


if __name__ == "__main__":
    preprocessor = Preprocessor("data")
    preprocessor.run_all()
