import os
import shutil
import tarfile

import numpy as np
import pandas as pd
import requests
from pathlib import Path

_GDAC = "http://gdac.broadinstitute.org/runs/stddata__2016_01_28/data/GBMLGG/20160128"

CGGA_EXCLUDED_MUTATION_TYPES = {
    'synonymous_variant',               # silent
    'intron_variant',
    '3_prime_UTR_variant',
    '5_prime_UTR_variant',
    'non_coding_transcript_exon_variant',  # RNA / lincRNA
    'upstream_gene_variant',
    'downstream_gene_variant',
}

TCGA_EXCLUDED_VARIANT_CLASSIFICATIONS = {
    'Silent',
    'Intron',
    "3'UTR",
    "5'UTR",
    'RNA',
    'lincRNA',
}


class Preprocessor:
    def __init__(self, data_dir):
        self.cgga_data_dir = Path(data_dir)
        self.tcga_data_dir = Path(data_dir)

    # --- Shared utilities ---

    @staticmethod
    def _download_and_extract(url, tar_path, extracted_name, dest):
        if dest.exists():
            print(f"{dest} already exists, skipping download.")
            return
        print(f"Downloading {url} ...")
        with open(tar_path, 'wb') as f:
            f.write(requests.get(url).content)
        with tarfile.open(tar_path) as tf:
            tf.extractall(path=tar_path.parent)
        shutil.move(tar_path.parent / extracted_name, dest)
        os.remove(tar_path)

    # --- CGGA ---

    def cgga_mutations(self):
        print("Preprocessing CGGA mutations data...")
        df = pd.read_csv(
            self.cgga_data_dir / 'CGGA.WEseq_286.20200506.txt',
            sep='\t',
            index_col=0,
            low_memory=False,
        )
        df = df.map(lambda v: v if (pd.notna(v) and v not in CGGA_EXCLUDED_MUTATION_TYPES) else None)
        df = df.T
        df.index.name = None
        df = (df.notna() & (df != '')).astype(float)
        df.to_csv(self.cgga_data_dir / 'cgga_mutations_preprocessed.csv')
        print(f"Saved {df.shape[0]} samples x {df.shape[1]} genes to cgga_mutations_preprocessed.csv")

    def cgga_cna(self):
        print("Preprocessing CGGA copy number data...")
        df = pd.read_csv(
            self.cgga_data_dir / 'cgga_all_thresholded.by_genes.txt',
            sep='\t',
            index_col=0,
            low_memory=False,
        )
        df = df.drop(columns=['Locus ID', 'Cytoband'])
        df.index = df.index.str.split('|').str[0]
        df.index.name = None
        df = df.T
        df.index.name = None
        df.to_csv(self.cgga_data_dir / 'cgga_cna_preprocessed.csv')
        print(f"Saved {df.shape[0]} samples x {df.shape[1]} genes to cgga_cna_preprocessed.csv")

    def cgga_labels(self):
        print("Preprocessing CGGA labels...")
        df = pd.read_csv(
            self.cgga_data_dir / 'CGGA.WEseq_286_clinical.20200506.txt',
            sep='\t',
            index_col=0,
        )
        labels = df['Subtype'].map(lambda v: 1 if 'GBM' in v else 0)
        labels.index.name = None
        labels.name = 'response'
        labels.to_csv(self.cgga_data_dir / 'cgga_labels_preprocessed.csv', header=True)
        print(f"Saved {len(labels)} labels (LGG=0: {(labels==0).sum()}, GBM=1: {(labels==1).sum()}) to cgga_labels_preprocessed.csv")

    # --- TCGA ---

    def download_tcga_mutations(self):
        name = 'gdac.broadinstitute.org_GBMLGG.Mutation_Packager_Calls.Level_3.2016012800.0.0'
        self._download_and_extract(
            url=f"{_GDAC}/{name}.tar.gz",
            tar_path=self.tcga_data_dir / 'mut.tar.gz',
            extracted_name=name,
            dest=self.tcga_data_dir / 'mut',
        )

    def tcga_mutations(self):
        print("Preprocessing TCGA mutations data...")
        self.download_tcga_mutations()

        mut_dir = self.tcga_data_dir / 'mut'
        samples = {}
        all_genes = set()
        for filename in os.listdir(mut_dir):
            if filename == 'MANIFEST.txt':
                continue
            sample_id = '-'.join(filename.split('-')[:-1])
            df = pd.read_csv(
                mut_dir / filename,
                sep='\t',
                usecols=['Hugo_Symbol', 'Variant_Classification'],
            )
            df = df[~df['Variant_Classification'].isin(TCGA_EXCLUDED_VARIANT_CLASSIFICATIONS)]
            mutated_genes = set(df['Hugo_Symbol'].unique())
            all_genes.update(mutated_genes)
            samples[sample_id] = samples.get(sample_id, set()) | mutated_genes

        genes = np.array(sorted(all_genes))
        sample_ids = sorted(samples)
        mut_mat = pd.DataFrame(
            [[1.0 if g in samples[s] else 0.0 for g in genes] for s in sample_ids],
            index=sample_ids,
            columns=genes,
        )
        mut_mat.index.name = None
        mut_mat.to_csv(self.tcga_data_dir / 'tcga_mutations_preprocessed.csv')
        print(f"Saved {mut_mat.shape[0]} samples x {mut_mat.shape[1]} genes to tcga_mutations_preprocessed.csv")

    def tcga_labels(self):
        print("Preprocessing TCGA labels...")
        name = 'gdac.broadinstitute.org_GBMLGG.Merge_Clinical.Level_1.2016012800.0.0'
        self._download_and_extract(
            url=f"{_GDAC}/{name}.tar.gz",
            tar_path=self.tcga_data_dir / 'clin.tar.gz',
            extracted_name=name,
            dest=self.tcga_data_dir / 'clin',
        )
        df = pd.read_csv(
            self.tcga_data_dir / 'clin' / 'GBMLGG.merged_only_clinical_clin_format.txt',
            sep='\t',
            index_col=0,
        )
        df = df.loc[['admin.disease_code', 'patient.bcr_patient_barcode']].T.set_index('patient.bcr_patient_barcode')
        df.columns = ['response']
        df.index = [x.upper() for x in df.index]
        df['response'] = [0 if x == 'lgg' else 1 for x in df['response']]
        df.index.name = None
        df.to_csv(self.tcga_data_dir / 'tcga_labels_preprocessed.csv')
        print(f"Saved {len(df)} labels (LGG=0: {(df['response']==0).sum()}, GBM=1: {(df['response']==1).sum()}) to tcga_labels_preprocessed.csv")

    def tcga_cna(self):
        print("Preprocessing TCGA copy number data...")
        cached = self.tcga_data_dir / 'Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes.gz'
        url = "https://tcga.xenahubs.net/download/TCGA.GBMLGG.sampleMap/Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes.gz"
        if not cached.exists():
            print(f"Downloading {url} ...")
            with open(cached, 'wb') as f:
                f.write(requests.get(url).content)
        cna = pd.read_csv(cached, sep='\t', index_col=0, compression='gzip')
        cna = cna.T
        cna.index = ['-'.join(x.split('-')[:3]) for x in cna.index]
        cna.index.name = None
        cna.to_csv(self.tcga_data_dir / 'tcga_cna_preprocessed.csv')
        print(f"Saved {cna.shape[0]} samples x {cna.shape[1]} genes to tcga_cna_preprocessed.csv")

    # --- Alignment ---

    def _align_pair(self, path_a, path_b):
        df_a = pd.read_csv(path_a, index_col=0)
        df_b = pd.read_csv(path_b, index_col=0)
        common = sorted(set(df_a.columns) & set(df_b.columns))
        print(f"  {path_a.name}: {df_a.shape[1]} genes -> {len(common)} common")
        print(f"  {path_b.name}: {df_b.shape[1]} genes -> {len(common)} common")
        df_a[common].to_csv(path_a)
        df_b[common].to_csv(path_b)

    def align_mutations(self):
        print("Aligning mutation gene sets...")
        self._align_pair(
            self.tcga_data_dir / 'tcga_mutations_preprocessed.csv',
            self.cgga_data_dir / 'cgga_mutations_preprocessed.csv',
        )

    def align_cna(self):
        print("Aligning CNA gene sets...")
        self._align_pair(
            self.tcga_data_dir / 'tcga_cna_preprocessed.csv',
            self.cgga_data_dir / 'cgga_cna_preprocessed.csv',
        )

    # --- Pipeline ---

    def run_all(self):
        self.cgga_mutations()
        self.cgga_cna()
        self.cgga_labels()
        self.tcga_mutations()
        self.tcga_labels()
        self.tcga_cna()
        self.align_mutations()
        self.align_cna()


if __name__ == '__main__':
    Preprocessor(Path(__file__).parent / 'data').run_all()
