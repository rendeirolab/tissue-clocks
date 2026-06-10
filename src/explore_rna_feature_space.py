"""
Explore gene expression data from GTEx
"""

from pathlib import Path

import requests
import pandas as pd
import matplotlib.pyplot as plt
from anndata import AnnData
import scanpy as sc
import matplotlib

from src.ops import process, plot_latent


metadata_dir = Path("metadata")
data_dir = Path("data") / "gtex"
results_dir = Path("results") / "gtex"
f = "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_reads.gct.gz"
expr_file = data_dir / "gene_expression" / f

figkws = dict(bbox_inches="tight", dpi=300)


def get_gene_expression_file():
    url = f"https://storage.googleapis.com/gtex_analysis_v8/rna_seq_data/{f}"
    r = requests.get(url)
    if r.ok:
        with expr_file.open("wb") as out:
            out.write(r.content)


def get_gene_expression(**kwargs):
    expr_pq_file = expr_file.replace_(".gz", ".pq")
    if not expr_pq_file.exists():
        if not expr_file.exists():
            get_gene_expression_file()
        df = pd.read_csv(
            expr_file,
            index_col=[0, 1],
            sep="\t",
            skiprows=2,
            **kwargs,
            # engine="pyarrow",
        )
        df.to_parquet(expr_pq_file)
    df = pd.read_parquet(expr_pq_file, **kwargs)
    return df


def analyze_expression():
    output_dir = results_dir / "expression"
    output_dir.mkdir(exist_ok=True)

    hist = pd.read_csv(data_dir / "GTEx Portal.csv", index_col=0)
    df = get_gene_expression()
    df = df.groupby(level="Description").sum()
    df.index.name = "Gene"
    df.columns = df.columns.str.extract(
        r"(GTEX-.*-\d+)-SM-\w\w\w\w\w", expand=False
    ).rename("Tissue Sample ID")
    df = df.T

    info = pd.read_table(
        metadata_dir / "GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt",
        index_col=0,
    )
    info[["SMTS", "SMTSD", "SMPTHNTS"]]

    sel = df.index.isin(hist.index)
    x = df.loc[sel]
    obs = hist.loc[x.index]
    obs["Age decade"] = obs["Age Bracket"].str.slice(0, 1).astype(int)
    a = AnnData(x, obs=obs)
    a.raw = a
    a.write(output_dir / "expression.h5ad")

    a = sc.read(output_dir / "expression.h5ad")

    sc.pp.log1p(a)
    sc.pp.normalize_total(a)
    process(a)
    a.write(output_dir / "expression.processed.h5ad")

    a = sc.read(output_dir / "expression.processed.h5ad")
    n_tissues = a.obs["Tissue"].nunique()
    n_ages = a.obs["Age Bracket"].nunique()
    a.uns["Tissue_colors"] = [
        matplotlib.colors.rgb2hex(c)
        for c in plt.get_cmap("tab20", n_tissues)(range(n_tissues))
    ]
    a.uns["Age Bracket_colors"] = [
        matplotlib.colors.rgb2hex(c)
        for c in plt.get_cmap("inferno", n_ages + 1)(range(n_ages))
    ]
    prefix = output_dir / "all_samples"
    voi = ["Tissue_simple", "Tissue", "Age Bracket", "Age decade", "Sex"]
    a.obs["Tissue_simple"] = a.obs["Tissue"].str.replace(r" - .*", "", regex=True)
    a.obs["Age decade"] = a.obs["Age Bracket"].str.slice(0, 1).astype(int)
    plot_latent(a, prefix=prefix, voi=voi)
