# SPDX-FileCopyrightText: Copyright (c) 2026 Rendeiro Lab, CeMM - Research Center for Molecular Medicine, Austrian Academy of Sciences
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#
# Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE).

"""
Visualize the distribution of samples in histopathology databases particularly across the axis of age.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn_extensions import clustermap
import networkx as nx
import statsmodels.api as sm

from src.utils import get_restricted_info

figkws = dict(bbox_inches="tight", dpi=300)


def main():
    sample_wise()
    individual_wise()


def sample_wise():
    input_dir = Path("data") / "gtex"
    input_dir.mkdir(exist_ok=True)
    output_dir = Path("results") / "gtex" / "cohort_visualization2"
    output_dir.mkdir(exist_ok=True)

    meta = pd.read_csv(input_dir / "GTEx Portal.csv", index_col=0)
    meta["Age Bracket"] = pd.Categorical(meta["Age Bracket"], ordered=True)
    meta["Tissue Simple"] = meta["Tissue"].str.replace(r" - .*", "", regex=True)

    # # Across all tissues
    # fig, ax = plt.subplots(figsize=(6, 4))
    # sns.barplot(
    #     x=meta["Age Bracket"].cat.categories,
    #     y=meta["Age Bracket"].value_counts().sort_index(),
    #     ax=ax,
    #     palette="magma",
    # )
    # ax.set(xlabel="Age bracket", ylabel="Sample count")
    # fig.savefig(output_dir / f"GTEx_slides_by_age.2.svg", **figkws)
    # fig.savefig(output_dir / f"GTEx_slides_by_age.2.png", **figkws)

    # # Across all ages
    # for var in ["Tissue", "Tissue Simple"]:
    #     fig, ax = plt.subplots(figsize=(4, 6))
    #     c = meta[var].value_counts()
    #     sns.barplot(x=c, y=c.index, ax=ax, palette="magma_r", orient="horiz")
    #     ax.set(xlabel="Tissue", ylabel="Sample count")
    #     fig.savefig(output_dir / f"GTEx_slides_by_{var}.2.svg", **figkws)
    #     fig.savefig(output_dir / f"GTEx_slides_by_{var}.2.png", **figkws)

    # # Number of organs per individual
    # for var in ["Tissue", "Tissue Simple"]:
    #     fig, ax = plt.subplots(figsize=(4, 6))
    #     c = meta[var].value_counts()
    #     sns.barplot(x=c, y=c.index, ax=ax, palette="magma_r", orient="horiz")
    #     ax.set(xlabel="Tissue", ylabel="Sample count")
    #     fig.savefig(output_dir / f"GTEx_slides_by_{var}.2.svg", **figkws)
    #     fig.savefig(output_dir / f"GTEx_slides_by_{var}.2.png", **figkws)

    # #
    # fig, ax = plt.subplots(figsize=(6, 3))
    # c = meta.groupby("Subject ID")["Tissue"].nunique()
    # sns.histplot(
    #     x=c,
    #     ax=ax,
    #     # palette="magma_r",
    #     # orient='horiz'
    # )
    # ax.set(xlabel="Number of tissues sampled per individual", ylabel="Individuals")
    # fig.savefig(output_dir / f"GTEx_tissues_per_individual.2.svg", **figkws)
    # fig.savefig(output_dir / f"GTEx_tissues_per_individual.2.png", **figkws)

    # Individually
    for t in meta["Tissue"].unique():
        m = meta.query(f"Tissue == '{t}'")

        fig, ax = plt.subplots()
        sns.barplot(
            x=m["Age Bracket"].cat.categories,
            y=m["Age Bracket"].value_counts().sort_index(),
            ax=ax,
            palette="magma",
        )
        ax.set(title=t, ylabel="Number of whole slide images", xlabel="Age Bracket")
        fig.savefig(output_dir / f"GTEx_slides_by_age_{t}.svg", **figkws)
        fig.savefig(output_dir / f"GTEx_slides_by_age_{t}.pdf", **figkws)

    # In same plot
    fig, axes = plt.subplots(
        10,
        4,
        figsize=(3.8 * 4, 3 * 10),
        sharex=True,
        sharey=True,
        gridspec_kw=dict(wspace=0),
    )
    for tissue, ax in zip(meta["Tissue"].unique(), axes.flatten()):
        m = meta.query(f"Tissue == '{tissue}'")

        sns.barplot(
            x=m["Age Bracket"].cat.categories,
            y=m["Age Bracket"].value_counts().sort_index(),
            ax=ax,
            palette="magma",
        )
        ax.set(title=f"{tissue}, n = {m.shape[0]}", xlabel="", ylabel="")

    for ax in axes[:, 0]:
        ax.set(ylabel="Number of whole slide images")
    for ax in axes[-1, :]:
        ax.set(xlabel="Age Bracket")

    fig.savefig(output_dir / "GTEx_slides_by_age_and_tissue.2.svg", **figkws)
    fig.savefig(output_dir / "GTEx_slides_by_age_and_tissue.2.pdf", **figkws)

    # Observe pathology and interaction with demographic variables
    path_counts = (
        meta["Pathology Categories"]
        .dropna()
        .str.split(", ")
        .apply(pd.Series)
        .stack()
        .value_counts()
    )

    fig, ax = plt.subplots(figsize=(6, 8))
    sns.barplot(x=path_counts, y=path_counts.index, orient="horiz", ax=ax)
    ax.set(xlabel="Term counts")
    fig.savefig(output_dir / "GTEx_pathology_classes.barplot.svg", **figkws)
    fig.savefig(output_dir / "GTEx_pathology_classes.barplot.pdf", **figkws)
    ax.set(xscale="log")
    fig.savefig(output_dir / "GTEx_pathology_classes.barplot.log.svg", **figkws)
    fig.savefig(output_dir / "GTEx_pathology_classes.barplot.log.pdf", **figkws)

    exclude = ["tma", "no_abnormalities", "clean_specimens"]
    path_counts = path_counts.drop(exclude)

    tissue_cats = dict()
    age_cats = dict()
    sex_cats = dict()
    har_cats = dict()
    for cat in path_counts.index:
        meta["c"] = meta["Pathology Categories"].str.contains(cat)
        tissue_cats[cat] = meta.groupby("Tissue")["c"].mean()
        age_cats[cat] = meta.groupby("Age Bracket")["c"].mean()
        sex_cats[cat] = meta.groupby("Sex")["c"].mean()
        har_cats[cat] = meta.groupby("Hardy Scale")["c"].mean()
    t = pd.DataFrame(tissue_cats).astype(float)
    a = pd.DataFrame(age_cats).astype(float)
    s = pd.DataFrame(sex_cats).astype(float)
    h = pd.DataFrame(har_cats).astype(float)

    del meta["c"]

    kws = dict(dendrogram_ratio=0.05, figsize=(7, 10), cbar_kws=dict(label="Frequency"))
    grid = sns.clustermap(
        t.T,
        metric="correlation",
        col_cluster=True,
        xticklabels=True,
        yticklabels=True,
        **kws,
    )
    grid.ax_heatmap.get_children()[0].set(rasterized=True)
    grid.fig.savefig(
        output_dir / "GTEx_pathology_classes.tissue_dependent.clustermap.svg", **figkws
    )
    grid.fig.savefig(
        output_dir / "GTEx_pathology_classes.tissue_dependent.clustermap.pdf", **figkws
    )

    grid = sns.clustermap(a.T, col_cluster=False, yticklabels=True, **kws)
    grid.ax_heatmap.get_children()[0].set(rasterized=True)
    grid.fig.savefig(
        output_dir / "GTEx_pathology_classes.age_dependent.clustermap.svg", **figkws
    )
    grid.fig.savefig(
        output_dir / "GTEx_pathology_classes.age_dependent.clustermap.pdf", **figkws
    )

    grid = sns.clustermap(s.T, col_cluster=False, yticklabels=True, **kws)
    grid.ax_heatmap.get_children()[0].set(rasterized=True)
    grid.fig.savefig(
        output_dir / "GTEx_pathology_classes.sex_dependent.clustermap.svg", **figkws
    )
    grid.fig.savefig(
        output_dir / "GTEx_pathology_classes.sex_dependent.clustermap.pdf", **figkws
    )

    grid = sns.clustermap(h.T, col_cluster=False, yticklabels=True, **kws)
    grid.ax_heatmap.get_children()[0].set(rasterized=True)
    grid.fig.savefig(
        output_dir / "GTEx_pathology_classes.hardy_dependent.clustermap.svg", **figkws
    )
    grid.fig.savefig(
        output_dir / "GTEx_pathology_classes.hardy_dependent.clustermap.pdf", **figkws
    )

    # Export as GEXF for plotting graph in Gephi
    v = (
        t.reset_index()
        .melt(id_vars="Tissue")
        .rename(columns={"Tissue": "source", "variable": "target"})
        .assign(attr="tissue")
    )
    v = pd.concat(
        [
            v,
            v.rename(columns={"target": "source", "source": "target"}).assign(
                attr="path"
            ),
        ]
    )

    G = nx.from_pandas_edgelist(v.loc[v["value"] > 0.1], edge_attr="value")
    n = v.set_index("source")["attr"].to_dict()
    n = {k: {"name": k, "class": v} for k, v in n.items()}
    nx.set_node_attributes(G, n)
    nx.write_gexf(G, output_dir / "GTEx_pathology_classes.gexf")
    output_dir / "GTEx_pathology_classes.gephy_visualization.gephi"
    output_dir / "GTEx_pathology_classes.gephy_visualization.svg"

    c_tissue_cats = dict()
    c_age_cats = dict()
    c_sex_cats = dict()
    c_har_cats = dict()
    for cat in path_counts.index:
        meta["c"] = meta["Pathology Categories"].str.contains(cat)
        c_tissue_cats[cat] = meta.groupby("Tissue")["c"].sum()
        c_age_cats[cat] = meta.groupby("Age Bracket")["c"].sum()
        c_sex_cats[cat] = meta.groupby("Sex")["c"].sum()
        c_har_cats[cat] = meta.groupby("Hardy Scale")["c"].sum()
    c_t = pd.DataFrame(c_tissue_cats).astype(int)
    c_a = pd.DataFrame(c_age_cats).astype(int)
    c_s = pd.DataFrame(c_sex_cats).astype(int)
    c_h = pd.DataFrame(c_har_cats).astype(int)

    del meta["c"]
    grid = sns.clustermap(
        c_t.T,
        metric="correlation",
        col_cluster=True,
        xticklabels=True,
        yticklabels=True,
        **kws,
    )
    grid.ax_heatmap.get_children()[0].set(rasterized=True)
    grid.fig.savefig(
        output_dir / "GTEx_pathology_classes.tissue_dependent.count.clustermap.svg",
        **figkws,
    )
    grid.fig.savefig(
        output_dir / "GTEx_pathology_classes.tissue_dependent.count.clustermap.pdf",
        **figkws,
    )
    grid = sns.clustermap(c_a.T, col_cluster=False, yticklabels=True, **kws)
    grid.ax_heatmap.get_children()[0].set(rasterized=True)
    grid.fig.savefig(
        output_dir / "GTEx_pathology_classes.age_dependent.count.clustermap.svg",
        **figkws,
    )
    grid.fig.savefig(
        output_dir / "GTEx_pathology_classes.age_dependent.count.clustermap.pdf",
        **figkws,
    )
    grid = sns.clustermap(c_s.T, col_cluster=False, yticklabels=True, **kws)
    grid.ax_heatmap.get_children()[0].set(rasterized=True)
    grid.fig.savefig(
        output_dir / "GTEx_pathology_classes.sex_dependent.count.clustermap.svg",
        **figkws,
    )
    grid.fig.savefig(
        output_dir / "GTEx_pathology_classes.sex_dependent.count.clustermap.pdf",
        **figkws,
    )
    grid = sns.clustermap(c_h.T, col_cluster=False, yticklabels=True, **kws)
    grid.ax_heatmap.get_children()[0].set(rasterized=True)
    grid.fig.savefig(
        output_dir / "GTEx_pathology_classes.hardy_dependent.count.clustermap.svg",
        **figkws,
    )
    grid.fig.savefig(
        output_dir / "GTEx_pathology_classes.hardy_dependent.count.clustermap.pdf",
        **figkws,
    )

    # Plot frequency across age as line plot
    t = a.iloc[:-1, :]
    y = t.index.str.slice(0, 1).astype(float)
    x = t.astype("float32").assign(intercept=1)
    lfc = sm.OLS(y, x).fit().params.sort_values()

    # a += 0.0001
    # lfc = (a.iloc[-2] / a.iloc[0]).sort_values()

    fig, axes = plt.subplots(
        1, 2, figsize=(2 * 5, 1 * 5), gridspec_kw=dict(width_ratios=[0.3, 0.6])
    )
    v = lfc.abs().max()
    v /= 2
    axes[0].scatter(lfc, lfc.index, c=lfc, cmap="coolwarm", vmin=-v, vmax=v)
    axes[0].axvline(0, linestyle="--", color="grey")
    axes[0].set(xlabel=r"$\beta$  (change with age)", ylabel="Pathological feature")
    axes[0].set_yticks(axes[0].get_yticks()[::2])
    # axes[0].set_yticklabels(axes[0].get_yticklabels()[::2])

    sel = lfc.tail(5).index.tolist() + lfc.head(1).index.tolist()
    ax2 = plt.twinx(axes[1])
    colors = sns.color_palette("tab10")
    for s, color in zip(sel, colors):
        axes[1].plot(a[s] * 100, c=color, label=s)
        axes[1].text(a.index[0], a.iloc[0][s] * 100, s=s, c=color)
        axes[1].text(a.index[-1], a.iloc[-1][s] * 100, s=s, c=color)
        ax2.plot(c_a[s], linestyle="--", linewidth=0.85)
    axes[1].set(xlabel="Age Bracket", ylabel="Frequency (%)")
    axes[1].legend()
    ax2.set(ylabel="Frequency (absolute)", yscale="log")
    fig.savefig(
        output_dir / "GTEx_pathology_classes.change_with_age.lineplot.svg", **figkws
    )
    fig.savefig(
        output_dir / "GTEx_pathology_classes.change_with_age.lineplot.pdf", **figkws
    )

    # Observe changes across age of same pathological category in different tissue
    tc_age_cats = dict()
    for cat in path_counts.index:
        meta["c"] = meta["Pathology Categories"].str.contains(cat)
        tc_age_cats[cat] = meta.groupby(["Tissue", "Age Bracket"])["c"].mean()
    tc_age_cats = pd.DataFrame(tc_age_cats)

    lfcs = dict()
    for tissue in tc_age_cats.index.levels[0]:
        t = tc_age_cats.loc[tissue].iloc[:-1, :]
        y = t.index.str.slice(0, 1).astype(float)
        x = t.astype("float32").assign(intercept=1).fillna(0)
        lfcs[tissue] = sm.OLS(y, x).fit().params.sort_values()
    lfcs = pd.DataFrame(lfcs).drop("intercept")

    sign = (lfcs > 0).astype(int).replace(0, -1)
    lfcs_cr = (lfcs.abs() ** (1 / 3)) * sign

    kws = dict(
        dendrogram_ratio=0.05,
        figsize=(7, 10),
        cbar_kws=dict(label=r"$\beta$ (change with age)"),
    )
    grid = sns.clustermap(
        lfcs, cmap="RdBu_r", center=0, xticklabels=True, yticklabels=True, **kws
    )
    grid.ax_heatmap.get_children()[0].set(rasterized=True)
    grid.fig.savefig(
        output_dir / "GTEx_pathology_classes.change_with_age.per_tissue.clustermap.svg",
        **figkws,
    )
    grid.fig.savefig(
        output_dir / "GTEx_pathology_classes.change_with_age.per_tissue.clustermap.pdf",
        **figkws,
    )

    kws = dict(
        dendrogram_ratio=0.05,
        figsize=(7, 10),
        cbar_kws=dict(label=r"$\beta$ (change with age)"),
        metric="correlation",
    )
    grid = sns.clustermap(
        lfcs_cr, cmap="RdBu_r", center=0, xticklabels=True, yticklabels=True, **kws
    )
    grid.ax_heatmap.get_children()[0].set(rasterized=True)
    grid.fig.savefig(
        output_dir
        / "GTEx_pathology_classes.change_with_age.per_tissue.clustermap.cube_root.svg",
        **figkws,
    )
    grid.fig.savefig(
        output_dir
        / "GTEx_pathology_classes.change_with_age.per_tissue.clustermap.cube_root.pdf",
        **figkws,
    )


def individual_wise():
    output_dir = Path("results") / "gtex" / "cohort_visualization2"
    output_prefix = output_dir / "restricted_info.SUFFIX"

    df, var_annot = get_restricted_info()
    # General
    general = ["Cohort", "Sex", "Age", "Race", "Ethnicity", "Height", "Weight", "BMI"]
    df["Cohort"] = pd.Categorical(df["Cohort"])

    race = {
        1: "Asian",
        2: "Black or African American",
        3: "White",
        4: "American Indian or Alaskan native",
        98: "Unreported",
        99: "Unknown",
    }
    df["Race"] = pd.Categorical(df["Race"].replace(race))
    ethnicity = {
        0: "Not Hispanic or Latino",
        1: "Hispanic or Latino",
        98: "Unreported",
        99: "Unknown",
    }
    df["Ethnicity"] = pd.Categorical(df["Ethnicity"].replace(ethnicity))

    # Death related variables
    df["Time of Death"] = pd.to_datetime(df["Time of Death"])
    df["Time of Death (hour)"] = df["Time of Death"].dt.hour + (
        df["Time of Death"].dt.minute / 60
    )
    df["Ischemic Time (Minutes)"] = df["Ischemic Time (Minutes)"].astype(float)
    df["Number Of Hours In Refrigeration"] = df[
        "Number Of Hours In Refrigeration"
    ].astype(float)

    # Serology
    serology = var_annot.loc[var_annot.index.str.startswith("LB"), "description"]

    q = df[serology]
    # q[q > 2] = np.nan
    grid = clustermap(
        q.fillna(False).astype(bool),
        mask=q.isnull(),
        figsize=(12, 10),
        row_colors=df[general],
        cmap="gnuplot",
        dendrogram_ratio=0.1,
    )
    grid.ax_heatmap.get_children()[0].set(rasterized=True)
    grid.savefig(output_prefix.with_suffix(".serology.clustermap.svg"), **figkws)
    grid.savefig(output_prefix.with_suffix(".serology.clustermap.pdf"), **figkws)

    # Morbidity
    morbidity = var_annot.loc[
        var_annot.index.str.startswith("MH"), "description"
    ].drop_duplicates()
    morbidity = morbidity[morbidity.isin(df.columns)]
    morbidity_exclude = ["Blood Donation Denial Reason"]
    morbidity = list(filter(lambda x: x not in morbidity_exclude, morbidity))

    q = df[morbidity].replace(99.0, np.nan).convert_dtypes()
    q = q.loc[:, q.dtypes == "Int64"].astype(float)

    grid = clustermap(
        q.fillna(0) > 0,
        mask=q.isnull(),
        figsize=(22, 10),
        row_colors=df[general],
        cmap="gnuplot",
        dendrogram_ratio=0.1,
    )
    grid.ax_heatmap.get_children()[0].set(rasterized=True)
    grid.savefig(output_prefix.with_suffix(".morbidity.clustermap.svg"), **figkws)
    grid.savefig(output_prefix.with_suffix(".morbidity.clustermap.pdf"), **figkws)


if __name__ == "__main__" and "get_ipython" not in locals():
    main()


# Examples: liver fibrosis vs normal
# slides = ['GTEX-111VG-0826', 'GTEX-12WSD-1426']
# for slide_name in slides:
#     slide_file = Path(f"{slide_name}.svs")
#     if not slide_file.exists():
#         url = f"https://brd.nci.nih.gov/brd/imagedownload/{slide_name}"
#         with open(slide_file, "wb") as handle:
#             req = requests.get(url)
#             handle.write(req.content)
