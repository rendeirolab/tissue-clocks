# SPDX-FileCopyrightText: Copyright (c) 2026 Rendeiro Lab, CeMM - Research Center for Molecular Medicine, Austrian Academy of Sciences
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#
# Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE).

"""
"""

import requests
from urlpath import URL
import pandas as pd
import matplotlib.pyplot as plt
from anndata import AnnData
import scanpy as sc
import matplotlib

from src import config

data_dir = config.data_dir / "dna_methylation"
output_dir = config.results_dir / "dna_methylation"
output_dir.mkdir(exist_ok=True)
base_url = URL("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE213nnn/GSE213478/suppl/")
f = "GSE213478_methylation_DNAm_noob_final_BMIQ_all_tissues_987.txt.gz"
dname_file = data_dir / f
var_f = "GSE213478_RAW.tar"
dname_var_file = data_dir / var_f
dname_pq_file = data_dir / "dname.pq"

figkws = dict(bbox_inches="tight", dpi=300)


def get_dname_file():
    import tarfile

    dname_var_file.parent.mkdir()
    for url, file in zip([base_url / f, base_url / var_f], [dname_var_file, var_f]):
        if (data_dir / file).exists():
            continue
        r = requests.get(url)
        if r.ok:
            with file.open("wb") as out:
                out.write(r.content)

        if url.name.endswith(".tar"):
            with tarfile.open(url.name) as tar:
                tar.extractall(file.parent)


def get_dname_data(top_n_var: int = 10_000) -> AnnData:
    f = dname_pq_file.with_suffix(f".subset_top_{top_n_var}_std.pq")
    if not f.exists():
        x = pd.read_csv((data_dir / f).with_suffix(""), index_col=0, engine="pyarrow")
        # mean = x.mean(1)
        std = x.std(1)

        # fig, ax = plt.subplots()
        # ax.scatter(mean, std, alpha=0.1, s=1)

        x_sel = x.loc[std.sort_values().tail(top_n_var).index, :].T
        x_sel.to_parquet(f)
    x_sel = pd.read_parquet(f)

    v = f.parent / "var.pq"
    if not v.exists():
        var = pd.read_csv(
            data_dir / "GPL21145_MethylationEPIC_15073387_v-1-0.csv.gz",
            index_col=0,
            skiprows=7,
            engine="python",
        )
        var.to_parquet(v)
    var = pd.read_parquet(v)
    var = var.loc[:, ~var.isnull().all()]
    var_sel = var.loc[x_sel.columns]
    # for col in var_sel.columns:
    #     var_sel.loc[var_sel[col].isnull(), col] = np.nan

    var_sel["UCSC_RefGene_Name-simplified"] = var["UCSC_RefGene_Name"].apply(
        lambda x: ";".join(list(set(str(x).split(";"))))
    )

    obs = pd.read_csv(data_dir.parent / "GTEx Portal.csv", index_col=0)
    obs["Age Decade"] = obs["Age Bracket"].str.slice(0, 1).astype(int)
    x_sel.index = x_sel.index.str.extract(r"(GTEX-.*-.*)-\w\w-\w\w\w\w\w")[0].rename("")

    a = AnnData(x_sel, obs=obs.reindex(x_sel.index), var=var_sel)
    return a


def analyze_dname():
    from src.ops import process, plot_latent

    n = 10_000

    a = get_dname_data(n)

    a.obs.to_csv("obs.csv")
    a.obs = pd.read_csv("obs.csv", index_col=0)
    a.uns["var"] = a.var[["UCSC_RefGene_Name-simplified"]].to_dict()
    a.var = pd.DataFrame(index=a.var.index.tolist())
    a.write((output_dir / f"dna_methylation.top_{n}_std_vars.h5ad"))
    a = sc.read((output_dir / f"dna_methylation.top_{n}_std_vars.h5ad"))

    a.X = 10**a.X
    process(a, confounder=None)

    voi = ["Tissue_simple", "Tissue", "Age Bracket", "Age Decade", "Sex"]
    n_tissues = a.obs["Tissue"].nunique()
    n_ages = a.obs["Age Bracket"].nunique()
    a.obs["Tissue_simple"] = a.obs["Tissue"].str.replace(r" - .*", "", regex=True)
    a.obs["Age Decade"] = a.obs["Age Bracket"].str.slice(0, 1).astype(float)
    a.uns["Tissue_colors"] = [
        matplotlib.colors.rgb2hex(c)
        for c in plt.get_cmap("tab20", n_tissues)(range(n_tissues))
    ]
    a.uns["Age Bracket_colors"] = [
        matplotlib.colors.rgb2hex(c)
        for c in plt.get_cmap("inferno", n_ages + 1)(range(n_ages))
    ]
    a.uns["var"]["mean"] = a.var["mean"].to_dict()
    a.uns["var"]["std"] = a.var["std"].to_dict()
    a.var = pd.DataFrame(index=a.var.index)
    a.write(output_dir / f"dna_methylation.top_{n}_std_vars.processed.h5ad")
    a = sc.read(output_dir / f"dna_methylation.top_{n}_std_vars.processed.h5ad")

    prefix = output_dir / f"all_samples.top_{n}_std_vars"
    plot_latent(a, prefix=prefix, voi=voi)
