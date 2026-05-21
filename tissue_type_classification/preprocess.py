import gzip
import io
import joblib
import traceback
import zipfile

import numpy as np
import pandas as pd
import requests
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder

# GTEx bulk RNAseq data (~18k samples, 300 people, 50+ tissue types)
# https://www.gtexportal.org/home/downloads/adult-gtex/bulk_tissue_expression
_DATA_URL = "https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz"
_META_URL = "https://storage.googleapis.com/adult-gtex/annotations/v8/metadata-files/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt"
_HUGO_URL = "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/locus_types/gene_with_protein_product.txt"

_EXPRESSION_GCT = "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct"
_METADATA_TSV = "GTEx_Analysis_v8_Annotations_SampleAttributesDS.tsv"
_HPA_URL = "https://www.proteinatlas.org/download/tsv/rna_tissue_hpa.tsv.zip"
_HPA_TSV = "rna_tissue_hpa.tsv"


class Preprocessor:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # --- GTEx ---

    def download_hugo_genes(self):
        print("Downloading HUGO protein-coding gene list...")
        response = requests.get(_HUGO_URL)
        with open(self.data_dir / "hugo_genes.txt", "wb") as f:
            f.write(response.content)

    def download_gtex_metadata(self):
        out = self.data_dir / _METADATA_TSV
        if out.exists():
            print("GTEx metadata already downloaded, skipping.")
            return
        print("Downloading GTEx sample metadata...")
        response = requests.get(_META_URL)
        with open(out, "w") as f:
            f.write(response.content.decode("utf-8"))

    def download_gtex_expression(self):
        out = self.data_dir / _EXPRESSION_GCT
        if out.exists():
            print("GTEx expression data already downloaded, skipping.")
            return
        print("Downloading GTEx expression data...")
        response = requests.get(_DATA_URL)
        with open(out, "wb") as f:
            f.write(gzip.decompress(response.content))

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
        info.to_csv(self.data_dir / "tissue_classes.csv", index=True)
        joblib.dump(encoder, self.data_dir / "tissue_encoder.pkl")
        pd.Series(encoder.categories_[0], name="tissue").to_csv(
            self.data_dir / "GTEx_tissue_classes.csv", index=False
        )
        print(f"Tissue labels saved: {info.shape[0]} samples, {info.shape[1]} classes.")

    # --- Human Protein Atlas ---

    def download_hpa_expression(self):
        out = self.data_dir / _HPA_TSV
        if out.exists():
            print("Human Protein Atlas tissue RNA data already downloaded, skipping.")
            return
        print("Downloading Human Protein Atlas tissue RNA data...")
        response = requests.get(_HPA_URL)
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            with zf.open(_HPA_TSV) as tsv_file:
                df = pd.read_csv(tsv_file, sep="\t")
        df.to_csv(out, index=False)
        print(f"HPA tissue RNA data saved: {df.shape[0]} rows, {df.shape[1]} columns.")

    def process_hpa_expression(self):
        print("Processing HPA expression data...")
        df = pd.read_csv(self.data_dir / _HPA_TSV)

        df["log2_tpm"] = np.log2(df["TPM"] + 1)

        duplicates = df.groupby(["Tissue", "Gene"]).size()
        n_duplicates = (duplicates > 1).sum()
        if n_duplicates > 0:
            print(f"Found {n_duplicates} duplicate Tissue/Gene pairs — aggregating by mean.")
            df = df.pivot_table(index="Tissue", columns="Gene", values="log2_tpm", aggfunc="mean")
        else:
            df = df.pivot(index="Tissue", columns="Gene", values="log2_tpm")
        df.index.name = "Tissue"
        df.columns.name = None

        df.to_csv(self.data_dir / "hpa_expression_preprocessed.csv", index=True)
        pd.Series(df.index, name="tissue").to_csv(
            self.data_dir / "hpa_tissue_classes.csv", index=False
        )
        print(f"HPA expression saved: {df.shape[0]} tissues, {df.shape[1]} genes.")

    def process_hpa_tissue_labels(self):
        print("Processing HPA tissue labels...")
        mapping = pd.read_csv(self.data_dir / "tissue_mappings.csv")
        hpa_to_gtex = dict(zip(mapping["hpa_tissue"], mapping["gtex_tissue"]))

        hpa = pd.read_csv(self.data_dir / "hpa_expression_preprocessed.csv", index_col=0)

        keep = hpa.index.map(hpa_to_gtex).notna()
        dropped = sorted(hpa.index[~keep])
        if dropped:
            print(f"Dropping {len(dropped)} unmapped HPA tissues: {dropped}")

        hpa = hpa[keep].copy()
        hpa.index = hpa.index.map(hpa_to_gtex)
        hpa.index.name = "Tissue"
        hpa.to_csv(self.data_dir / "hpa_expression_preprocessed.csv", index=True)

        encoder = joblib.load(self.data_dir / "tissue_encoder.pkl")
        encoded = encoder.transform(hpa.index.to_frame().values)
        labels = pd.DataFrame(encoded.todense(), index=hpa.index, columns=encoder.categories_[0])
        labels.to_csv(self.data_dir / "hpa_tissue_classes.csv", index=True)
        print(f"HPA tissue labels saved: {labels.shape[0]} tissues, {labels.shape[1]} classes.")

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

        gtex[common_ensg].set_axis(symbol_columns, axis=1).to_csv(self.data_dir / "GTEx_gene_log2_tpm_0.csv",
                                                                  index=True)
        hpa[common_ensg].set_axis(symbol_columns, axis=1).to_csv(self.data_dir / "hpa_expression_preprocessed.csv",
                                                                 index=True)

        print(f"Both datasets saved with {len(symbol_columns)} gene symbol columns.")

    # --- Pipeline ---

    def run_all(self):
        print("Starting preprocessing...")
        steps = [
            ("download_hugo_genes", self.download_hugo_genes),
            ("download_gtex_metadata", self.download_gtex_metadata),
            ("download_gtex_expression", self.download_gtex_expression),
            ("download_hpa_expression", self.download_hpa_expression),
            ("process_gtex_expression", self.process_gtex_expression),
            ("process_gtex_tissue_labels", self.process_gtex_tissue_labels),
            ("process_hpa_expression", self.process_hpa_expression),
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
    preprocessor = Preprocessor("data2")
    preprocessor.run_all()
