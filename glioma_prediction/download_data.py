import requests, os
import tarfile, shutil

DATA_DIR = "glioma_prediction/data"

if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)


def _download_and_extract(url, tar_path, extracted_name, dest):
    if os.path.exists(dest):
        print(f"{dest} already exists, skipping.")
        return
    print(f"Downloading {url} ...")
    with open(tar_path, "wb") as f:
        f.write(requests.get(url).content)
    with tarfile.open(tar_path) as tf:
        tf.extractall()
    shutil.move(extracted_name, dest)
    os.remove(tar_path)


if not os.path.exists(f"{DATA_DIR}/hugo_genes.txt"):
    print("Downloading HUGO gene list...")
    with open(f"{DATA_DIR}/hugo_genes.txt", "wb") as f:
        f.write(requests.get("https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/locus_types/gene_with_protein_product.txt").content)

_GDAC = "http://gdac.broadinstitute.org/runs/stddata__2016_01_28/data/GBMLGG/20160128"

_download_and_extract(
    url=f"{_GDAC}/gdac.broadinstitute.org_GBMLGG.Merge_snp__humanhap550__hudsonalpha_org__Level_3__segmented_cna__seg.Level_3.2016012800.0.0.tar.gz",
    tar_path=f"{DATA_DIR}/cna.tar.gz",
    extracted_name="gdac.broadinstitute.org_GBMLGG.Merge_snp__humanhap550__hudsonalpha_org__Level_3__segmented_cna__seg.Level_3.2016012800.0.0",
    dest=f"{DATA_DIR}/cna",
)

_download_and_extract(
    url=f"{_GDAC}/gdac.broadinstitute.org_GBMLGG.Mutation_Packager_Calls.Level_3.2016012800.0.0.tar.gz",
    tar_path=f"{DATA_DIR}/mut.tar.gz",
    extracted_name="gdac.broadinstitute.org_GBMLGG.Mutation_Packager_Calls.Level_3.2016012800.0.0",
    dest=f"{DATA_DIR}/mut",
)

_download_and_extract(
    url=f"{_GDAC}/gdac.broadinstitute.org_GBMLGG.Merge_Clinical.Level_1.2016012800.0.0.tar.gz",
    tar_path=f"{DATA_DIR}/clin.tar.gz",
    extracted_name="gdac.broadinstitute.org_GBMLGG.Merge_Clinical.Level_1.2016012800.0.0",
    dest=f"{DATA_DIR}/clin",
)

_download_and_extract(
    url=f"{_GDAC}/gdac.broadinstitute.org_GBMLGG.Merge_rnaseqv2__illuminahiseq_rnaseqv2__unc_edu__Level_3__RSEM_genes_normalized__data.Level_3.2016012800.0.0.tar.gz",
    tar_path=f"{DATA_DIR}/gexpr.tar.gz",
    extracted_name="gdac.broadinstitute.org_GBMLGG.Merge_rnaseqv2__illuminahiseq_rnaseqv2__unc_edu__Level_3__RSEM_genes_normalized__data.Level_3.2016012800.0.0",
    dest=f"{DATA_DIR}/gexpr",
)
