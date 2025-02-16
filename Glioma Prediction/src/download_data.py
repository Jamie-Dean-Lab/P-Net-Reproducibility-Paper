import requests, os
import tarfile, shutil

if not os.path.exists("Glioma Prediction/data"):
    os.mkdir("Glioma Prediction/data")

with open("Glioma Prediction/data/hugo_genes.txt", "wb") as f:
    text = requests.get("https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/locus_types/gene_with_protein_product.txt").content
    f.write(text)

with open("Glioma Prediction/data/cna.tar.gz", "wb") as f:
    data = requests.get("http://gdac.broadinstitute.org/runs/stddata__2016_01_28/data/GBMLGG/20160128/gdac.broadinstitute.org_GBMLGG.Merge_snp__humanhap550__hudsonalpha_org__Level_3__segmented_cna__seg.Level_3.2016012800.0.0.tar.gz").content
    f.write(data)
file = tarfile.open("Glioma Prediction/data/cna.tar.gz")
file.extractall()
file.close()
shutil.move("gdac.broadinstitute.org_GBMLGG.Merge_snp__humanhap550__hudsonalpha_org__Level_3__segmented_cna__seg.Level_3.2016012800.0.0", "Glioma Prediction/data/cna")

with open("Glioma Prediction/data/mut.tar.gz", "wb") as f:
    data = requests.get("http://gdac.broadinstitute.org/runs/stddata__2016_01_28/data/GBMLGG/20160128/gdac.broadinstitute.org_GBMLGG.Mutation_Packager_Calls.Level_3.2016012800.0.0.tar.gz").content
    f.write(data)
file = tarfile.open("Glioma Prediction/data/mut.tar.gz")
file.extractall()
file.close()
shutil.move("gdac.broadinstitute.org_GBMLGG.Mutation_Packager_Calls.Level_3.2016012800.0.0", "Glioma Prediction/data/mut")

with open("Glioma Prediction/data/clin.tar.gz", "wb") as f:
    data = requests.get("http://gdac.broadinstitute.org/runs/stddata__2016_01_28/data/GBMLGG/20160128/gdac.broadinstitute.org_GBMLGG.Merge_Clinical.Level_1.2016012800.0.0.tar.gz").content
    f.write(data)
file = tarfile.open("Glioma Prediction/data/clin.tar.gz")
file.extractall()
file.close()
shutil.move("gdac.broadinstitute.org_GBMLGG.Merge_Clinical.Level_1.2016012800.0.0", "Glioma Prediction/data/clin")

with open("Glioma Prediction/data/gexpr.tar.gz", "wb") as f:
    data = requests.get("http://gdac.broadinstitute.org/runs/stddata__2016_01_28/data/GBMLGG/20160128/gdac.broadinstitute.org_GBMLGG.Merge_rnaseqv2__illuminahiseq_rnaseqv2__unc_edu__Level_3__RSEM_genes__data.Level_3.2016012800.0.0.tar.gz").content
    f.write(data)
file = tarfile.open("Glioma Prediction/data/gexpr.tar.gz")
file.extract("gdac.broadinstitute.org_GBMLGG.Merge_rnaseqv2__illuminahiseq_rnaseqv2__unc_edu__Level_3__RSEM_genes__data.Level_3.2016012800.0.0/gdac.broadinstitute.org_GBMLGG.Merge_rnaseqv2__illuminahiseq_rnaseqv2__unc_edu__Level_3__RSEM_genes__data.Level_3.2016012800.0.0/GBMLGG.rnaseqv2__illuminahiseq_rnaseqv2__unc_edu__Level_3__RSEM_genes__data.data.txt")
file.close()
shutil.move("gdac.broadinstitute.org_GBMLGG.Merge_rnaseqv2__illuminahiseq_rnaseqv2__unc_edu__Level_3__RSEM_genes__data.Level_3.2016012800.0.0", "Glioma Prediction/data/gexpr")

