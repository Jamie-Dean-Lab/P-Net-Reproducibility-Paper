"""Print mutation and CNV burden as the model sees them, across training and external validation cohorts.

Uses the exact same files and encoders as the experiments:
  - Training:          P1000_final_analysis_set_cross_important_only.csv  (mut_binary)
                       P1000_data_CNA_paper.csv                           (cnv_amp / cnv_del)
  - External val:      combined/mut_combined.csv                          (mut_binary)
                       combined/cnv_combined.csv                          (cnv_amp_from_signed / cnv_del_from_signed)
  - Labels:            combined/labels_combined.csv                       (metastatic: 1=Met500, 0=PRAD)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from prostate_cancer_prediction.feature_encoders import (
    cnv_amp, cnv_amp_from_signed, cnv_del, cnv_del_from_signed, mut_binary,
)

DATA_DIR = os.path.join("prostate_cancer_prediction", "data", "_database")
PROSTATE_DIR = os.path.join(DATA_DIR, "prostate")
COMBINED_DIR = os.path.join(PROSTATE_DIR, "external_validation", "combined")

# Selected-gene universe
_sel = set(pd.read_csv(os.path.join(DATA_DIR, "genes",
    "tcga_prostate_expressed_genes_and_cancer_genes.csv"))["genes"])
_hugo = set(pd.read_csv(
    os.path.join(DATA_DIR, "genes", "HUGO_genes",
                 "protein-coding_gene_with_coordinate_minimal.txt"),
    sep="\t", header=None).iloc[:, 3].unique())
SELECTED_GENES = _sel & _hugo

def sg(df):
    return [g for g in SELECTED_GENES if g in df.columns]

# ---------------------------------------------------------------------------
# Training (same files/encoders as base_config.py)
# ---------------------------------------------------------------------------
train_mut_raw = pd.read_csv(os.path.join(PROSTATE_DIR, "processed",
    "P1000_final_analysis_set_cross_important_only.csv"), index_col=0)
train_cnv_raw = pd.read_csv(os.path.join(PROSTATE_DIR, "processed",
    "P1000_data_CNA_paper.csv"), index_col=0)
train_labels  = pd.read_csv(os.path.join(PROSTATE_DIR, "processed",
    "response_paper.csv"), index_col=0)

tidx  = train_mut_raw.index.intersection(train_labels.index)
y     = train_labels.loc[tidx, "response"]
t_mut = mut_binary(train_mut_raw.loc[tidx, sg(train_mut_raw)])
t_amp = cnv_amp(train_cnv_raw.loc[tidx, sg(train_cnv_raw)])
t_del = cnv_del(train_cnv_raw.loc[tidx, sg(train_cnv_raw)])

# ---------------------------------------------------------------------------
# External validation (same files/encoders as nested_CV_base_config.py)
# ---------------------------------------------------------------------------
ext_labels = pd.read_csv(os.path.join(COMBINED_DIR, "labels_combined.csv"), index_col=0)
ext_mut_raw = pd.read_csv(os.path.join(COMBINED_DIR, "mut_combined.csv"), index_col=0)
ext_cnv_raw = pd.read_csv(os.path.join(COMBINED_DIR, "cnv_combined.csv"), index_col=0)

eidx = ext_mut_raw.index.intersection(ext_labels.index)
y_ext = ext_labels.loc[eidx, "metastatic"]  # 1 = Met500, 0 = PRAD

e_mut = mut_binary(ext_mut_raw.loc[eidx, sg(ext_mut_raw)])
e_amp = cnv_amp_from_signed(ext_cnv_raw.loc[eidx, sg(ext_cnv_raw)])
e_del = cnv_del_from_signed(ext_cnv_raw.loc[eidx, sg(ext_cnv_raw)])

# ---------------------------------------------------------------------------
# Shared gene sets for fair comparison
# ---------------------------------------------------------------------------
sg_mut = sorted(set(t_mut.columns) & set(e_mut.columns))
sg_cnv = sorted(set(t_amp.columns) & set(e_amp.columns))

def burden(df, genes, mask=None):
    sub = df[[g for g in genes if g in df.columns]]
    if mask is not None:
        sub = sub.loc[mask]
    return (sub != 0).sum(axis=1).mean()

def rate(df, gene, mask=None):
    if gene not in df.columns:
        return float("nan")
    s = df[gene] if mask is None else df.loc[mask, gene]
    return (s != 0).mean() * 100

n_tp = int((y == 0).sum())
n_tm = int((y == 1).sum())
n_m5 = int((y_ext == 1).sum())
n_pr = int((y_ext == 0).sum())

print(f"Shared selected genes — mutations: {len(sg_mut)}, CNV: {len(sg_cnv)}")
print()
print("Mutation burden (mean mutated genes per sample, important-only in training):")
print(f"  Training primary    (n={n_tp:>3}): {burden(t_mut, sg_mut, y==0):.1f}")
print(f"  Training metastatic (n={n_tm:>3}): {burden(t_mut, sg_mut, y==1):.1f}")
print(f"  Ext Met500          (n={n_m5:>3}): {burden(e_mut, sg_mut, y_ext==1):.1f}")
print(f"  Ext PRAD            (n={n_pr:>3}): {burden(e_mut, sg_mut, y_ext==0):.1f}")
print()
print("  Gene-level mutation rates (% samples mutated):")
print(f"  {'Gene':<10} {'T-prim':>8} {'T-met':>8} {'Met500':>8} {'PRAD':>8}")
for gene in ["TP53", "SPOP", "PTPRD"]:
    print(f"  {gene:<10} {rate(t_mut, gene, y==0):>7.1f}%  {rate(t_mut, gene, y==1):>7.1f}%"
          f"  {rate(e_mut, gene, y_ext==1):>7.1f}%  {rate(e_mut, gene, y_ext==0):>7.1f}%")
print()
print("CNV amplification burden (mean amplified genes per sample):")
print(f"  Training primary    (n={n_tp:>3}): {burden(t_amp, sg_cnv, y==0):.1f}")
print(f"  Training metastatic (n={n_tm:>3}): {burden(t_amp, sg_cnv, y==1):.1f}")
print(f"  Ext Met500          (n={n_m5:>3}): {burden(e_amp, sg_cnv, y_ext==1):.1f}")
print(f"  Ext PRAD            (n={n_pr:>3}): {burden(e_amp, sg_cnv, y_ext==0):.1f}")
print()
print("  Gene-level amplification rates (% samples amplified):")
print(f"  {'Gene':<10} {'T-prim':>8} {'T-met':>8} {'Met500':>8} {'PRAD':>8}")
for gene in ["AR", "MYC", "CCND1"]:
    print(f"  {gene:<10} {rate(t_amp, gene, y==0):>7.1f}%  {rate(t_amp, gene, y==1):>7.1f}%"
          f"  {rate(e_amp, gene, y_ext==1):>7.1f}%  {rate(e_amp, gene, y_ext==0):>7.1f}%")
print()
print("CNV deletion burden (mean deleted genes per sample):")
print(f"  Training primary    (n={n_tp:>3}): {burden(t_del, sg_cnv, y==0):.1f}")
print(f"  Training metastatic (n={n_tm:>3}): {burden(t_del, sg_cnv, y==1):.1f}")
print(f"  Ext Met500          (n={n_m5:>3}): {burden(e_del, sg_cnv, y_ext==1):.1f}")
print(f"  Ext PRAD            (n={n_pr:>3}): {burden(e_del, sg_cnv, y_ext==0):.1f}")
print()
print("  Gene-level deletion rates (% samples deleted):")
print(f"  {'Gene':<10} {'T-prim':>8} {'T-met':>8} {'Met500':>8} {'PRAD':>8}")
for gene in ["PTEN", "RB1", "TP53", "MAP3K7"]:
    print(f"  {gene:<10} {rate(t_del, gene, y==0):>7.1f}%  {rate(t_del, gene, y==1):>7.1f}%"
          f"  {rate(e_del, gene, y_ext==1):>7.1f}%  {rate(e_del, gene, y_ext==0):>7.1f}%")
