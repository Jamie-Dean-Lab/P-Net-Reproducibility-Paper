import os, subprocess, requests

if not os.path.exists("radiosensitivity_prediction/data"):
    os.mkdir("radiosensitivity_prediction/data")

subprocess.call('"C:/Program Files/R/R-4.4.1/bin/x64/Rscript" "radiosensitivity_prediction/src/download_R_datasets.R',
                shell=True)

with open("radiosensitivity_prediction/data/hugo_genes.txt", "wb") as f:
    text = requests.get("https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/locus_types/gene_with_protein_product.txt").content
    f.write(text)
    