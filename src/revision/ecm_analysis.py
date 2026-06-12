# SPDX-FileCopyrightText: Copyright (c) 2026 Rendeiro Lab, CeMM - Research Center for Molecular Medicine, Austrian Academy of Sciences
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#
# Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE).

"""
Plan:
- [x] show the expression of these genes in light of tissue specificity;
- [x] show what fraction are significantly changing with age, and by what magnitude;
- [x] perform simple enrichment analysis (e.g. Fisher or hyperg tests) on these sets from all differentially expressed genes to assess whether they are enriched in the changes;
- [x] check the correlation of age gaps with these modules to identify whether ECM remodeling is a major factor driving aging changes, and in which tissues that may be the case.
"""

from pathlib import Path
import json

from tqdm import tqdm
import numpy as np
import pandas as pd
import scanpy as sc
import sklearn
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from seaborn_extensions import clustermap

from pandarallel import pandarallel

from src.utils import get_restricted_info

tqdm().pandas()
pandarallel.initialize(progress_bar=False)

metadata_dir = Path("metadata")
data_dir = Path("data")
expr_dir = Path("data") / "gtex" / "gene_expression"
gaps_dir = Path("results") / "gtex" / "fine_tuned" / "_pre_2024-01-19_age_X_frac1.0"
results_dir = Path("results")
output_dir = Path("results") / "tissue_clocks_revision" / "ecm_analysis"
output_dir.mkdir(parents=True, exist_ok=True)

ecm_genes = json.load((metadata_dir / "ecm_genes.json").open())
all_genes = list(set([g for m in ecm_genes for g in ecm_genes[m]]))
ecm_mapping = (
    pd.Series({g: c for c, gs in ecm_genes.items() for g in gs}, name="group")
    .reindex(all_genes)
    .rename_axis(index="gene")
    .astype("category")
)

color_mapping = dict(zip(ecm_mapping.cat.categories, sns.color_palette()))

figkws = dict(bbox_inches="tight", dpi=300)

meta = pd.read_csv(Path("data") / "gtex" / "GTEx Portal.csv", index_col=0)
meta["Organ"] = meta["Tissue"].str.extract(r"(\w+).*", expand=False)
restricted, _ = get_restricted_info()
meta = meta.merge(
    restricted[["Age"]], left_on="Subject ID", right_index=True, how="left"
)
exclude_entities = [
    "Bladder",
    "Cervix - Ectocervix",
    "Cervix - Endocervix",
    "Fallopian Tube",
    "Kidney - Medulla",
]


age_gaps = pd.read_parquet(
    gaps_dir / "tissue-specific_clocks.Ridge.GroupKFold.predictions_residuals.pq"
)
age_gaps = age_gaps.query("Tissue not in @exclude_entities")


def main():
    for voi in ["Tissue", "Organ"]:
        tissue_specificity(voi)


def plot_ecm_genes_table():
    # Make nice plot with a table-like structure
    # describing which genes are part of which group
    from matplotlib.table import Table

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_axis_off()
    tab = Table(ax, bbox=[0, 0, 1, 1])
    ncols = len(ecm_genes)
    nrows = max(len(g) for g in ecm_genes.values())
    width = 1 / ncols
    height = 1.0 / nrows
    for i, group in enumerate(ecm_genes):
        # add color matched from color_mapping
        tab.add_cell(
            0,
            i,
            width,
            height,
            text="\n".join(group.split(" ")),
            loc="center",
            facecolor=color_mapping[group],
        )
        for j, gene in enumerate(sorted(ecm_genes[group])):
            tab.add_cell(
                j + 1, i, width, height, text=gene, loc="center", facecolor="white"
            )
    ax.add_table(tab)
    # make fontsize 10 for all cells
    for cell in tab.get_celld().values():
        cell.set_fontsize(100)
    fig.savefig(output_dir / "ecm_genes_table.svg", **figkws)


def correlation_with_age_gaps(target_var: str = "residuals_adj"):
    g = age_gaps
    # Load up expression data
    e = pd.read_parquet(expr_dir / "log_cpm.grouped_by_sample.pq")
    es = e[all_genes]

    # Calculate correlation of each gene with age gaps in each tissue
    _corrs = dict()
    for tissue in tqdm(sorted(g["Tissue"].unique())):
        gt = g.query("Tissue == @tissue")[target_var]
        gt = gt.reindex(es.index).dropna()
        et = es.reindex(gt.index).dropna(axis=1, how="all")
        gt = gt.reindex(et.index).dropna()
        corrs = et.corrwith(gt, method="spearman")
        _corrs[tissue] = corrs
    corrs = pd.DataFrame(_corrs)

    g = clustermap(
        corrs.T,
        config="abs",
        center=0,
        cmap="RdBu_r",
        figsize=(14, 9),
        square=False,
        metric="euclidean",
    )
    g.fig.savefig(
        output_dir / f"ecm_genes.{target_var}_correlation.clustermap.svg", **figkws
    )

    # Aggregate correlations within each gene group, plot as violin plots
    pg = corrs.groupby(ecm_mapping).mean().T

    g = clustermap(
        pg, config="abs", center=0, cmap="RdBu_r", figsize=(5, 9), square=False
    )
    g.fig.savefig(
        output_dir / f"ecm_gene_groups.{target_var}_correlation.clustermap.svg",
        **figkws,
    )

    pppv = pg.melt()
    order = pppv.groupby("group")["value"].mean().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(4, 2))
    sns.boxplot(
        data=pppv,
        x="value",
        y="group",
        hue="group",
        orient="horiz",
        order=order,
        ax=ax,
        saturation=0.5,
    )
    sns.swarmplot(
        data=pppv,
        x="value",
        y="group",
        hue="group",
        orient="horiz",
        order=order,
        ax=ax,
        alpha=0.5,
    )
    ax.axvline(0, linestyle="--", color="black", linewidth=2)
    ax.set(xlabel="Coefficient with age", ylabel="ECM gene group")
    fig.savefig(
        output_dir / f"ecm_gene_groups.{target_var}_correlation.violin.svg", **figkws
    )


def overlap_with_differnetially_expressed():
    from scipy.stats import fisher_exact

    # Load up differential expression results
    f = (
        results_dir
        / "gtex"
        / "expression"
        / "age_X_frac1.0"
        / "tissue-specific_clocks.Ridge.GroupKFold.coefficients.pq"
    )
    coefs = pd.read_parquet(f)
    coefs["Organ"] = coefs["Tissue"].str.replace(r" - .*", "", regex=True)
    coefs = coefs.query("~Tissue.isin(@exclude_entities)")

    _res = list()
    fig, axes = plt.subplots(4, 10, figsize=(10 * 3, 4 * 3), sharey=True)
    for tissue, ax in zip(sorted(coefs["Tissue"].unique()), axes.flatten()):
        t = coefs.query("Tissue == @tissue")
        t = t.drop(t.index[t.index.str.contains(r"_|\)")])
        ecm = t.reindex(all_genes)
        rest = t.drop(all_genes)

        # Make Fisher exact test table (separately for up- and down-regulated genes)
        for direction, sign in zip(["up", "down"], [">", "<"]):
            both = ecm.query(f"original {sign} 1e-5").size
            p = t.query(f"original {sign} 1e-5").size
            n = rest.query(f"original {sign} 1e-5").size
            s = "<=" if sign == ">" else "<="
            neither = (
                ecm.loc[
                    ~(ecm.index.isin(t.query(f"original.abs() {s} 1e-5").index))
                ].size
                + rest.loc[
                    ~(rest.index.isin(t.query(f"original.abs() {s} 1e-5").index))
                ].size
            )
            table = np.array([[both, p], [n, neither]])
            oddsratio, pvalue = fisher_exact(table, alternative="greater")
            _res.append(
                [
                    tissue,
                    direction,
                    both,
                    p,
                    n,
                    neither,
                    oddsratio,
                    pvalue,
                ]
            )
        # Make Fischer test regardless of direction
        both = ecm.loc[ecm.index.isin(t.query("original.abs() > 1e-5").index)].size
        p = t.query("original.abs() > 1e-5").size
        n = rest.query("original.abs() > 1e-5").size
        neither = (
            ecm.loc[~(ecm.index.isin(t.query("original.abs() > 1e-5").index))].size
            + rest.loc[~(rest.index.isin(t.query("original.abs() > 1e-5").index))].size
        )
        table = np.array([[both, p], [n, neither]])
        oddsratio, pvalue = fisher_exact(table, alternative="greater")
        _res.append(
            [
                tissue,
                "both",
                both,
                p,
                n,
                neither,
                oddsratio,
                pvalue,
            ]
        )
    res = pd.DataFrame(
        _res,
        columns=[
            "Tissue",
            "Direction",
            "Both",
            "ECM only",
            "Other only",
            "Neither",
            "Odds ratio",
            "P-value",
        ],
    )

    resp = res.pivot_table(
        index="Tissue", columns="Direction", values=["Odds ratio", "P-value"]
    ).sort_values(("Odds ratio", "both"))
    resp.to_csv(output_dir / "ecm_genes_fisher_exact_test.p-values.csv")

    # Report what count of the ECM genes are significantly changing with age, per ECM gene group
    gene_set_count = pd.Series({s: len(ecm_genes[s]) for s in ecm_genes})
    sig = coefs.query("original > 1e-5")
    up = {
        group: sig.groupby("Tissue").apply(
            lambda x: x.index.isin(ecm_genes[group]).sum()
        )
        for group in ecm_genes
    }
    p = pd.DataFrame(up) / gene_set_count * 100
    g = clustermap(
        p,
        square=False,
        config="abs",
        figsize=(3, 6),
        vmin=0,
        vmax=75,
        cbar_kws=dict(label="% genes upregulated with age"),
        dendrogram_ratio=0.05,
        cmap="Reds",
    )
    g.fig.savefig(
        output_dir / "ecm_genes_upregulated_with_age.clustermap.svg", **figkws
    )
    sig = coefs.query("original < -1e-5")
    dn = {
        group: sig.groupby("Tissue").apply(
            lambda x: x.index.isin(ecm_genes[group]).sum()
        )
        for group in ecm_genes
    }
    p = pd.DataFrame(dn) / gene_set_count * 100
    g = clustermap(
        p,
        square=False,
        config="abs",
        figsize=(3, 6),
        vmin=0,
        vmax=75,
        cbar_kws=dict(label="% genes downregulated with age"),
        dendrogram_ratio=0.05,
        cmap="Blues",
        row_linkage=g.dendrogram_row.linkage,
        col_linkage=g.dendrogram_col.linkage,
    )
    g.fig.savefig(
        output_dir / "ecm_genes_downregulated_with_age.clustermap.svg", **figkws
    )

    # Test with Fisher exact test whether the ECM genes are enriched in the differentially expressed genes

    # Show gene's coefficients with age
    colors = [color_mapping[c] for c in ecm_mapping]
    for group in ["Tissue", "Organ"]:
        if group == "Tissue":
            p = coefs.groupby(group).apply(lambda x: x.loc[all_genes, "original"])
        else:
            p = (
                coefs.drop("Tissue", axis=1)
                .reset_index()
                .groupby([group, "index"])
                .mean()
                .reset_index(level=group)
                .groupby(group)
                .apply(lambda x: x.loc[all_genes, "original"])
            )
        g = clustermap(
            p,
            config="abs",
            center=0,
            cmap="RdBu_r",
            figsize=(16, 8),
            dendrogram_ratio=0.05,
            square=False,
            metric="correlation",
            # metric="euclidean",
            col_colors=colors,
        )
        g.fig.savefig(
            output_dir
            / f"ecm_genes.age_coefficients.{group}.clustermap.with_colors.svg",
            **figkws,
        )

        pg = p.T.groupby(ecm_mapping).mean().T
        g = clustermap(
            pg,
            config="abs",
            center=0,
            cmap="RdBu_r",
            figsize=(5, 9),
            dendrogram_ratio=0.05,
            square=False,
            metric="correlation",
        )
        g.fig.savefig(
            output_dir / f"ecm_gene_groups.age_coefficients.{group}.clustermap.svg",
            **figkws,
        )

        pppv = pg.melt()
        order = pppv.groupby("group")["value"].mean().sort_values().index.tolist()
        fig, ax = plt.subplots(figsize=(4, 3))
        # sns.violinplot(
        #     data=pppv,
        #     x="value",
        #     y="group",
        #     hue="group",
        #     orient="horiz",
        #     order=order,
        #     ax=ax,
        # )

        sns.boxplot(
            data=pppv,
            x="value",
            y="group",
            hue="group",
            orient="horiz",
            order=order,
            ax=ax,
            saturation=0.5,
        )
        sns.swarmplot(
            data=pppv,
            x="value",
            y="group",
            hue="group",
            orient="horiz",
            order=order,
            ax=ax,
            alpha=0.5,
        )
        ax.axvline(0, linestyle="--", color="grey", linewidth=1)
        ax.set(xlabel="Coefficient with age", ylabel="ECM gene group")
        fig.savefig(
            output_dir / f"ecm_gene_groups.age_coefficients.{group}.violin.svg",
            **figkws,
        )


def tissue_specificity(voi: str = "Organ"):
    e = get_mean_expression_data(voi)

    colors = [color_mapping[c] for c in ecm_mapping]

    # plot legend separately
    fig, ax = plt.subplots(figsize=(2, 2))
    handles = [
        matplotlib.patches.Patch(color=color_mapping[c], label=c)
        for c in ecm_mapping.cat.categories
    ]
    ax.legend(
        handles=handles,
        loc="center",
        frameon=False,
        fontsize=8,
        handletextpad=0.5,
        handlelength=1,
        borderpad=0.5,
        labelspacing=0.5,
        columnspacing=1,
        ncol=1,
    )
    ax.axis("off")
    fig.savefig(output_dir / f"{voi}_ecm_gene_groups.legend.svg", **figkws)

    kwargs = dict(square=False, figsize=(16, 8), dendrogram_ratio=0.05)
    p = e[all_genes].rename_axis(columns="gene")
    g = clustermap(p, config="abs", metric="correlation", **kwargs)
    g.fig.savefig(output_dir / f"{voi}_ecm_genes.clustermap.a.svg", **figkws)
    g = clustermap(p, config="z", **kwargs)
    g.fig.savefig(output_dir / f"{voi}_ecm_genes.clustermap.z.svg", **figkws)
    # g = clustermap(p, config="z", metric="euclidean", col_colors=ecm_mapping)
    g = clustermap(p, config="z", metric="euclidean", col_colors=colors, **kwargs)
    g.fig.savefig(
        output_dir / f"{voi}_ecm_genes.clustermap.z.with_colors.svg", **figkws
    )

    # Gene groups
    # # Simple average
    kwargs = dict(figsize=(5, 9), square=False, dendrogram_ratio=0.05)
    pg = e.T.groupby(ecm_mapping).mean().T
    g = clustermap(pg, config="abs", metric="correlation", **kwargs)
    g.fig.savefig(output_dir / f"{voi}_ecm_groups.clustermap.a.svg", **figkws)
    g = clustermap(pg, config="z", metric="euclidean", **kwargs)
    g.fig.savefig(output_dir / f"{voi}_ecm_groups.clustermap.z.svg", **figkws)

    # # Scoring
    ea = sc.AnnData(e)
    for group in ecm_genes:
        sc.tl.score_genes(ea, ecm_genes[group], score_name=group)

    kwargs = dict(figsize=(3, 8), square=False, dendrogram_ratio=0.05)
    p = ea.obs
    g = clustermap(
        p, config="abs", metric="correlation", **kwargs, cmap="RdBu_r", center=0
    )
    g.fig.savefig(output_dir / f"{voi}_ecm_group_scores.clustermap.a.svg", **figkws)
    g = clustermap(p, config="z", metric="euclidean", **kwargs)
    g.fig.savefig(output_dir / f"{voi}_ecm_group_scores.clustermap.z.svg", **figkws)


def get_mean_expression_data(var: str = "Organ"):
    if (expr_dir / f"log_cpm.grouped_by_{var}.pq").exists():
        return pd.read_parquet(expr_dir / f"log_cpm.grouped_by_{var}.pq")
    if not (expr_dir / "log_cpm.grouped_by_sample.pq").exists():
        e = get_gene_expression_data()
        s = e.obs.index.str.extract(r"^(GTEX-.*-\d+)-SM-.*", expand=False)
        m = e.to_df().groupby(s).mean()
        m.to_parquet(expr_dir / "log_cpm.grouped_by_sample.pq")
    else:
        m = pd.read_parquet(expr_dir / "log_cpm.grouped_by_sample.pq")
    o = meta.reindex(m.index)
    mo = m.groupby(o[var]).mean()
    mo.to_parquet(expr_dir / f"log_cpm.grouped_by_{var}.pq")
    return mo


def get_gene_expression_data():
    expr_log_cpm = pd.read_parquet(expr_dir / "log_cpm.pq")
    obs = pd.read_parquet(expr_dir / "log_cpm.obs.pq")
    e = sc.AnnData(expr_log_cpm, obs)
    return e
