import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import gzip
import requests

# Downloads GTEx bulk RNAseq data containing approximately 18k samples from 300 people across 50+ tissue types
# https://www.gtexportal.org/home/downloads/adult-gtex/bulk_tissue_expression

data_url = "https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz"
meta_data_url = "https://storage.googleapis.com/adult-gtex/annotations/v8/metadata-files/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt"
stem = "Tissue Type Classification/data"

if not os.path.exists(f"{stem}/GTEx"):
    os.mkdir(f"{stem}/GTEx")

# Download gene list
temp = requests.get("https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/locus_types/gene_with_protein_product.txt")
with open(f"{stem}/hugo_genes.txt", "wb") as f:
    f.write(temp.content)

# Download metadata
if not os.path.exists(f"{stem}/GTEx/GTEx_Analysis_v8_Annotations_SampleAttributesDS.tsv"):
    with open(f"{stem}/GTEx/GTEx_Analysis_v8_Annotations_SampleAttributesDS.tsv", "w") as f:
        response = requests.get(meta_data_url)
        f.write(response.content.decode("utf-8"))

# Download Expression data
if not os.path.exists(f"{stem}/GTEx/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct"):
    with open(f"{stem}/GTEx/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct", "wb") as f:
        response = requests.get(data_url)
        f.write(gzip.decompress(response.content))

# Process the gct file to format that is easier to use
# Transform the data using log2 transform
# Chunk into sets of 1000
df = pd.read_csv(f"{stem}/GTEx/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct", sep="\t", 
                 skiprows=2, index_col=1)
df = df.drop("Name", axis=1).T
df = np.log2(df + 1)
df = df.sample(df.shape[0], replace=False, random_state=42)
for i in range(0, df.shape[0], 1000):
    if i + 1000 < df.shape[0]:
        subset = df.iloc[i:i+1000, :]
    else:
        subset = df.iloc[i:, :]
    subset.to_csv("{}/GTEx/GTEx_gene_log2_tpm_{}.csv".format(stem, i // 1000), index=True)

# Extract the tissue labels to create labels
info = pd.read_csv(f"{stem}/GTEx/GTEx_Analysis_v8_Annotations_SampleAttributesDS.tsv", sep="\t", index_col=0)
info = info[["SMTSD"]].loc[df.index]
tissueonehot = OneHotEncoder()
tissuedata = tissueonehot.fit_transform(info)
info = pd.DataFrame(tissuedata.todense(), index=info.index, columns=tissueonehot.categories_)
info.to_csv(f"{stem}/GTEx/tissue_classes.csv", index=True)