import pandas as pd
import numpy as np
import os, requests, gzip, tqdm

wd = "Glioma Prediction/data"

# Process CNA
if not os.path.exists(f"{wd}/hg38_reference_genome.gff3"):
    with open(f"{wd}/hg38_reference_genome.gz", "wb") as f:
        data = requests.get("https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_47/gencode.v47.annotation.gff3.gz").content
        f.write(data)
    with gzip.open(f"{wd}/hg38_reference_genome.gz", "rb") as f:
        data = f.read()
        with open(f"{wd}/hg38_reference_genome.gff3", "wb") as o:
            o.write(data)
genome = pd.read_csv(f"{wd}/hg38_reference_genome.gff3", skiprows=7, sep="\t", header=None)
genome.columns = ["seqid", "source", "type", "start", "end", "score", "strand", "phase", "attributes"]
genome = genome.loc[genome["type"] == "gene"]
genome["seqid"] = genome["seqid"].str.replace("chr", "")
genome["seqid"] = genome["seqid"].str.replace("X", "23")
genome = genome.loc[genome["seqid"].str.isnumeric()]
genome["symbol"] = genome["attributes"].apply(lambda x : x[x.find("gene_name") + 10:x.find(";", x.find("gene_name"))])
genome = genome[["seqid", "symbol", "start", "end"]]
genome["seqid"] = genome["seqid"].astype(int)
df = pd.read_csv(f"{wd}/cna/GBMLGG.snp__humanhap550__hudsonalpha_org__Level_3__segmented_cna__seg.seg.txt", sep="\t")
df = df.merge(genome, left_on="Chromosome", right_on="seqid")
df = df.loc[((df["start"] < df["End"]) & (df["start"] > df["Start"])) | ((df["end"] < df["End"]) & (df["end"] > df["Start"]))]
df["Sample"] = df["Sample"].apply(lambda x : "-".join(x.split("-")[:3]))
df = df.loc[df["symbol"].str.find("ENSG") == -1]

genes = np.array(sorted(df["symbol"].unique()))
cna_mat = np.zeros((df["Sample"].nunique(), len(genes)))
df = df.sort_values("Sample")
idxs = []
for i, sample in enumerate(tqdm.tqdm(df["Sample"].unique())):
    cnas = df.loc[df["Sample"] == sample, ["symbol", "Segment_Mean"]]
    cna_mat[i, [np.argwhere(x == genes).ravel()[0] for x in cnas["symbol"]]] = cnas["Segment_Mean"].to_numpy()
    idxs.append(sample)
cna_mat = pd.DataFrame(cna_mat, index=idxs, columns=genes)
cna_mat.to_csv(f"{wd}/cnas.csv", index=True)

# Process mutations
samples = {}
genes = []
for sample in os.listdir(f"{wd}/mut"):
    if sample != "MANIFEST.txt":
        sample_id = "-".join(sample.split("-")[:-1])
        df = pd.read_csv(f"{wd}/mut/{sample}", sep="\t")
        genes += df["Hugo_Symbol"].to_list()
        samples[sample_id] = df.groupby("Hugo_Symbol")["Variant_Classification"].count()

genes = np.array(sorted(list(set(genes))))
mut_mat = np.zeros((len(samples), len(genes)))
idxs = []
for i, (k, v) in enumerate(samples.items()):
    idxs.append(k)
    muts = np.argwhere(genes.reshape(-1,1) == v.index.to_numpy().reshape(1, -1))[:, 0]
    mut_mat[i, muts] = v.to_numpy()
mut_mat = pd.DataFrame(mut_mat, index=idxs, columns=genes)
mut_mat.to_csv(f"{wd}/mutations.csv")

# Process gene expression
df = pd.read_csv(f"{wd}/gexpr/GBMLGG.rnaseqv2__illuminahiseq_rnaseqv2__unc_edu__Level_3__RSEM_genes_normalized__data.data.txt", sep="\t")
df = df.iloc[1:, :]
df.columns = ["gene_id"] + ["-".join(x.split("-")[:3]) for x in df.columns[1:]]
df["gene_id"] = df["gene_id"].apply(lambda x : x.split("|")[0])
df = df.loc[df["gene_id"] != "?"].reset_index().drop("index", axis=1)
df = df.T
df.columns = df.iloc[0, :]
df = df.iloc[1:, :]
df = df[~df.index.duplicated(keep="first")]
df.to_csv(f"{wd}/gexpr.csv", index=True)

# Process labels
df = pd.read_csv(f"{wd}/clin/GBMLGG.merged_only_clinical_clin_format.txt", sep="\t", index_col=0)
df = df.loc[["admin.disease_code", "patient.bcr_patient_barcode"]].T.set_index("patient.bcr_patient_barcode")
df.columns = ["response"]
df.index = [x.upper() for x in df.index]
df["response"] = [0 if x == "lgg" else 1 for x in df["response"]]
df.to_csv(f"{wd}/response.csv")

