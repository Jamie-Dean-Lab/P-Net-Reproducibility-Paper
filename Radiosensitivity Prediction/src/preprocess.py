import pandas as pd
import os, itertools
import numpy as np

# Generate target labels
wd = "Glioma Prediction/data"
df = pd.read_csv(f"{wd}/clin/GBMLGG.clin.merged.txt", sep="\t", header=None, index_col=0)
labels = pd.DataFrame({"response" : [0 if x == "lgg" else 1 for x in df.loc["admin.disease_code"]],
                       "sample_id" : df.loc["patient.bcr_patient_barcode"].str.upper()}).set_index("sample_id")
labels.to_csv(f"{wd}/response.csv", index=True)

# Get mutations
mutations = {}
for mut_file in os.listdir(f"{wd}/mut"):
    if mut_file != "MANIFEST.txt":
        id = "-".join(mut_file.split("-")[:3])
        df = pd.read_csv(f"{wd}/mut/{mut_file}", sep="\t")
        mutations[id] = df.groupby("Hugo_Symbol")["Variant_Classification"].count()
gene_set = np.array(sorted(list(set(itertools.chain.from_iterable([v.index.to_list() for k,v in mutations.items()])))))
mut_mat = np.zeros((len(mutations), len(gene_set)))
indices = []
for i, (k,v) in enumerate(mutations.items()):
    indices.append(k)
    cols = np.argwhere(v.index.to_numpy().reshape(1,-1) == gene_set.reshape(-1,1))[:, 0]
    mut_mat[i, cols] = v.to_numpy()
mut_mat = pd.DataFrame(mut_mat, index=indices, columns=gene_set)
mut_mat.to_csv(f"{wd}/mutations.csv", index=True)

# get CNAs
df = pd.read_csv(f"{wd}/cna/GBMLGG.snp__humanhap550__hudsonalpha_org__Level_3__segmented_cna__seg.seg.txt", sep="\t")
pass

# get gene expression