from pathlib import Path

from tqdm import tqdm
import numpy as np
import pandas as pd
import sklearn
import matplotlib
import matplotlib.pyplot as plt
from seaborn_extensions import clustermap
from src.utils import clustermap_marsilea
from pandarallel import pandarallel

from tenacity import retry, stop_after_attempt

from src.utils import get_restricted_info

tqdm().pandas()
pandarallel.initialize(progress_bar=False)


metadata_dir = Path("metadata")
data_dir = Path("data")
expr_dir = Path("data") / "gtex" / "gene_expression"
results_dir = Path("results")
input_dir = (
    results_dir
    / "gtex"
    / "fine_tuned"
    / "_pre_2024-01-19_age_X_frac1.0"
    / "gene_expression"
)
output_dir = input_dir.parent / "gene_expression_resubmission"
output_dir.mkdir(parents=True, exist_ok=True)

figkws = dict(bbox_inches="tight", dpi=300)

meta = pd.read_csv(Path("data") / "gtex" / "GTEx Portal.csv", index_col=0)
meta["Tissue Simple"] = meta["Tissue"].str.extract(r"(\w+).*", expand=False)
restricted, _ = get_restricted_info()
meta = meta.merge(
    restricted[["Age"]], left_on="Subject ID", right_index=True, how="left"
)


# Enrichment


@retry(stop=stop_after_attempt(3))
def enrichr(genes: list[str], gene_sets: list[str]) -> pd.DataFrame:
    import gseapy

    r = gseapy.enrichr(genes, gene_sets=gene_sets)
    return r.results


gene_sets = [
    "NCI-Nature_2016",
    "KEGG_2021_Human",
    "WikiPathway_2023_Human",
    "GO_Biological_Process_2023",
    "GO_Cellular_Component_2023",
]
threshold = 0.035
tissues = sorted(meta["Tissue"].unique())
tracker = tqdm(total=4 * 1 * len(tissues) * 5, leave=True, position=0)
for target_var in ["Age", "residuals_adj"]:
    # target_var = "residuals_adj"
    for fit_type in ["Ridge"]:
        # fit_type = 'Ridge'
        fits = pd.read_csv(
            input_dir
            / f"{target_var}.express_regression.z.{fit_type}_fit.with_covariates.csv",
            index_col=0,
        ).drop(
            [
                "Intercept",
                "Age",
                "Cohort_Surgical",
                "Cohort_Organ Donor (OPO)",
                "Cohort_Postmortem",
                "Sex_Female",
                "Sex_Male",
            ],
            errors="ignore",
        )
        tissues = sorted(fits["tissue"].unique())

        # # Based on percentile threshold per tissue, ignore direction
        # _enr = list()
        # for tissue in tissues:
        #     # tissue = tissues[0]
        #     v = max(1e-25, fits["coef"].abs().quantile(0.985))
        #     # v = 1e-5
        #     # v = 0.035

        #     genes = fits.query(f"tissue == @tissue & mean > 1 & abs(coef) > {v}")
        #     # Cap at 200
        #     genes = genes["coef"].abs().sort_values().tail(200).index.tolist()
        #     tracker.update(1)
        #     if len(genes) < 5:
        #         continue
        #     r = enrichr(genes, gene_sets=gene_sets)
        #     _enr.append(r.assign(tissue=tissue))
        # enr = pd.concat(_enr)
        # enr.to_csv(
        #     output_dir
        #     / f"{target_var}.express_regression.{fit_type}_fit.abs_top_genes.enrichr.csv"
        # )

        # # Based on percentile threshold per tissue
        # _enr = list()
        # for tissue in tissues:
        #     # tissue = tissues[0]
        #     for sign, direction in [("", ">"), ("-", "<")]:
        #         # sign, direction = "", ">"
        #         v = max(1e-25, fits["coef"].abs().quantile(0.985))
        #         genes = fits.query(
        #             f"tissue == @tissue & mean > 1 & coef {direction} {sign}{v}"
        #         ).index.tolist()
        #         tracker.update(1)
        #         if len(genes) < 5:
        #             continue
        #         r = enrichr(genes, gene_sets=gene_sets)
        #         _enr.append(r.assign(tissue=tissue, direction=direction))
        # enr = pd.concat(_enr)
        # enr.to_csv(
        #     output_dir
        #     / f"{target_var}.express_regression.{fit_type}_fit.top_genes.enrichr.csv"
        # )

        # # Simply top 200 up and down
        # for n_genes in [200, 50, 10]:
        #     _enr = list()
        #     # for tissue in tissues[tissues.index(tissue) + 1 :]:
        #     for tissue in tqdm(tissues):
        #         for sign, direction in [("", ">"), ("-", "<")]:
        #             rs = fits.query(
        #                 f"tissue == @tissue & mean > 1 & coef {direction}0"
        #             ).sort_values("coef")
        #             genes = (
        #                 rs.tail(n_genes) if sign else rs.head(n_genes)
        #             ).index.tolist()
        #             r = enrichr(genes, gene_sets=gene_sets)
        #             _enr.append(r.assign(tissue=tissue, direction=direction))
        #             tracker.update(1)
        #     enr = pd.concat(_enr)
        #     enr.to_csv(
        #         output_dir
        #         / f"{target_var}.express_regression.{fit_type}_fit.top_{n_genes}_genes.enrichr.csv"
        #     )

        # # Based on fixed threshold across whole dataset, no direction
        # if (
        #     output_dir
        #     / f"{target_var}.express_regression.{fit_type}_fit.sig_genes.enrichr.csv"
        # ).exists():
        #     continue
        # _enr = list()
        # for tissue in tissues:
        #     rs = fits.query(
        #         "tissue == @tissue & mean > 1 & abs(coef) > @threshold"
        #     ).sort_values("coef")
        #     genes = rs.index.tolist()
        #     if not genes:
        #         continue
        #     r = enrichr(genes, gene_sets=gene_sets)
        #     _enr.append(r.assign(tissue=tissue))
        #     tracker.update(1)
        # enr = pd.concat(_enr)
        # enr.to_csv(
        #     output_dir
        #     / f"{target_var}.express_regression.{fit_type}_fit.sig_genes.enrichr.csv"
        # )

        # Based on fixed threshold across whole dataset, with direction
        if (
            output_dir
            / f"{target_var}.express_regression.{fit_type}_fit.sig_genes_direction.enrichr.csv"
        ).exists():
            continue
        _enr = list()
        for tissue in tissues:
            for sign, direction in [("", ">"), ("-", "<")]:
                rs = fits.query(
                    f"tissue == @tissue & mean > 1 & abs(coef) > @threshold & coef {direction}0"
                ).sort_values("coef")
                genes = rs.index.tolist()
                if not genes:
                    continue
                r = enrichr(genes, gene_sets=gene_sets)
                _enr.append(r.assign(tissue=tissue, direction=direction))
                tracker.update(1)
        enr = pd.concat(_enr)
        enr.to_csv(
            output_dir
            / f"{target_var}.express_regression.{fit_type}_fit.sig_genes_direction.enrichr.csv"
        )


# Plot enrichments
fit_type = "Ridge"
exclude = [
    "Kidney - Medulla",
    "Fallopian Tube",
    "Bladder",
    "Cervix - Ectocervix",
    "Cervix - Endocervix",
]
for label in [
    # "abs_top_genes",
    # "top_genes",
    # "top_200_genes",
    # "top_50_genes",
    # "top_10_genes",
    "sig_genes",
    "sig_genes_direction",
]:
    for target_var in ["Age", "prediction", "residuals_adj"]:
        file = (
            output_dir
            / f"{target_var}.express_regression.{fit_type}_fit.{label}.enrichr.csv"
        )
        if not file.exists():
            continue
        enr = pd.read_csv(file, index_col=0).query("~tissue.isin(@exclude)")
        gb = ["tissue", "direction"] if "direction" in enr.columns else ["tissue"]

        # # enr["n_genes"] = enr["Genes"].str.count(";") + 1
        # # enr["n_gene_set"] = (
        # #     enr["Overlap"].str.split("/").parallel_apply(lambda x: int(x[1]))
        # # )
        # # enr["ratio"] = enr["n_genes"] / enr["n_gene_set"]
        # # enr = enr.query("ratio >= 0.1")

        # # # across gene sets
        # enrp = enr.pivot_table(
        #     index=["Gene_set", "Term"], columns=gb, values="Odds Ratio"
        # )
        # sel = (
        #     enr.set_index(["Gene_set", "Term"])
        #     .groupby(gb)["Odds Ratio"]
        #     .nlargest(2)
        #     .unstack(gb)
        #     .index
        # )
        # p = enrp.reindex(sel).fillna(0)
        # p = p.loc[p.var(1) > 0, p.var(0) > 0]
        # of = (
        #     output_dir
        #     / f"{target_var}.express_regression.{fit_type}_fit.{label}.enrichr.all_gene_sets.clustermap.svg"
        # )
        # # if not of.exists():
        # g = clustermap_marsilea(
        #     p,
        #     config="abs",
        #     square=True,
        #     robust=True,
        #     metric="cosine",
        # )
        # g.fig.savefig(of, **figkws)

        # # per gene set
        for gs in gene_sets:
            enrs = enr.query("Gene_set == @gs")
            n_genes = (
                enrs.set_index("tissue")["Genes"]
                .str.split(",")
                .explode()
                .groupby(level=0)
                .size()
            )
            enrs = (
                enrs.groupby(gb + ["Gene_set", "Term"])["Combined Score"]
                .mean()
                .reset_index()
            )
            enrp = enrs.pivot_table(
                index=["Term"],
                columns=gb,
                values="Combined Score",
            )
            if enrp.empty:
                continue

            t1 = enrp.var(axis=1).sort_values().dropna().tail(24).index.tolist()
            t1 += enrp.mean(axis=1).sort_values().dropna().tail(24).index.tolist()
            t1 = list(set(t1))
            # t2 = enrp.var(axis=0).sort_values().dropna().tail(24).index.tolist()
            # t2 += enrp.mean(axis=0).sort_values().dropna().tail(24).index.tolist()
            # t2 = list(set(t2))
            t2 = enrp.columns

            p = enrp.loc[t1, t2].fillna(0)
            p = p.loc[p.var(1) > 0, p.var(0) > 0]
            of = (
                output_dir
                / f"{target_var}.express_regression.{fit_type}_fit.{label}.enrichr.{gs}.clustermap.var_mean_selection.svg"
            )
            if p.columns.nlevels > 1:
                p1 = p.reorder_levels([1, 0], axis=1)[">"]
                p2 = p.reorder_levels([1, 0], axis=1)["<"] * -1
                p3 = p1.fillna(0) + p2.fillna(0)
                p3 = p3.loc[p3.var(1) > 0, p3.var(0) > 0]
                if p3.shape[1] > 2 and p3.shape[0] > 2:
                    g = clustermap_marsilea(
                        p3,
                        metric="correlation",
                        square=True,
                        cmap="coolwarm",
                        center=0,
                        vmin=-10,
                        vmax=10,
                    )
                    g.fig.savefig(of.with_suffix(".mean_direction.svg"), **figkws)

                    p3 = (
                        p3.T.groupby(p3.columns.str.replace(r" - .*", "", regex=True))
                        .mean()
                        .T
                    )
                    g = clustermap_marsilea(
                        p3,
                        metric="correlation",
                        square=True,
                        cmap="coolwarm",
                        center=0,
                        vmin=-10,
                        vmax=10,
                    )
                    g.fig.savefig(of.with_suffix(".mean_direction.organ.svg"), **figkws)

                p.columns = (
                    p.columns.get_level_values(1) + " " + p.columns.get_level_values(0)
                )
            vmax = p.quantile(0.95).quantile(0.75)
            g = clustermap_marsilea(
                p,
                metric="correlation",
                square=True,
                vmax=vmax,
                # col_colors=n_genes.to_frame("n_genes"),
            )
            g.fig.savefig(of, **figkws)

            t = p.quantile(0.85).quantile(0.75)
            ps = p.loc[(p > t).sum(1) >= 2]
            ps = ps.loc[:, ps.var() > 0]
            g = clustermap_marsilea(
                ps,
                metric="correlation",
                square=True,
                vmax=vmax,
                # col_colors=n_genes.to_frame("n_genes"),
            )
            g.fig.savefig(of.with_suffix(".sel.svg"), **figkws)

            o = ps.columns.str.replace(r" - .*", "", regex=True)
            ps = ps.T.groupby(o).mean().T
            g = clustermap_marsilea(
                ps,
                metric="correlation",
                square=True,
                vmax=vmax,
                # col_colors=n_genes.to_frame("n_genes"),
            )
            g.fig.savefig(of.with_suffix(".sel.organ.svg"), **figkws)


# Replot the signature based analysis
# output_dir = Path("results") / "gtex" / "expression.signature_level"
# target_var = 'Age'
# z = 'o'
# out = (
#     output_dir
#     / f"{target_var}.express_regression.{z}.{fit_type}_fit.with_covariates.csv"
# )
# df = pd.read_csv(out, index_col=0)
# df = df.loc[df.index.str.startswith("Signature:")]

# c = df.pivot_table(index='tissue', columns='gene', values='coef')
# p = df.pivot_table(index='tissue', columns='gene', values='p_value')

# # c = c.loc[:, c.isnull().sum(0) < 3]
# # p = p.loc[:, c.columns]

# g = clustermap(c.fillna(0), mask=c.isnull(), annot=(p < 1e-5).astype(int), cmap='RdBu_r', center=0)


label = "sig_genes_direction"
fit_type = "Ridge"
exclude = [
    "Kidney - Medulla",
    "Fallopian Tube",
    "Bladder",
    "Cervix - Ectocervix",
    "Cervix - Endocervix",
]
exc_paths = [
    "Spermatogenesis",
    "Allograft Rejection",
    "KRAS Signaling Dn",
    "Pancreas Beta Cells",
    "UV Response Dn",
]
excel_file_to_write = "enrichments_significant.xlsx"
gs = "MSigDB_Hallmark_2020"
target_var = "Age"
file = input_dir / f"{target_var}.express_regression.{fit_type}_fit.{label}.enrichr.csv"
enr = (
    pd.read_csv(file, index_col=0)
    .query("~tissue.isin(@exclude)")
    .query("Gene_set == @gs")
    .drop(["Old P-value", "Old Adjusted P-value"], axis=1)
)
enr = enr[
    ["tissue", "direction", "Gene_set", "Term"]
    + list(enr.columns.difference(["tissue", "direction", "Gene_set", "Term"]))
]
enr["direction"] = enr["direction"].replace({">": "Upregulated", "<": "Downregulated"})
with pd.ExcelWriter(
    excel_file_to_write,
    engine="openpyxl",
    mode="a" if Path(excel_file_to_write).exists() else "w",
) as writer:
    enr.query("`Adjusted P-value` < 0.05 & `Odds Ratio` > 1").to_excel(
        writer, sheet_name=target_var, index=False
    )


for gs in ["MSigDB_Hallmark_2020"]:
    _comp = list()
    for target_var in ["Age", "residuals_adj"]:
        file = (
            input_dir
            / f"{target_var}.express_regression.{fit_type}_fit.{label}.enrichr.csv"
        )
        if not file.exists():
            continue
        enr = (
            pd.read_csv(file, index_col=0)
            .query("~tissue.isin(@exclude)")
            .query("Gene_set == @gs")
        )
        enr.query("`Adjusted P-value` < 0.05 & `Odds Ratio` > 1").to_excel(
            "enrichments_significant.xlsx", sheet_name=target_var, index=False
        )
        gb = ["tissue", "direction"] if "direction" in enr.columns else ["tissue"]
        enrs = (
            enr.groupby(gb + ["Gene_set", "Term"])["Odds Ratio"]
            .mean()
            .reset_index()
            .pivot_table(index=["Term"], columns=gb, values="Odds Ratio")
        )
        pvals = (
            enr.groupby(gb + ["Gene_set", "Term"])["Adjusted P-value"]
            .mean()
            .reset_index()
            .pivot_table(index=["Term"], columns=gb, values="Adjusted P-value")
        )
        s = (pvals < 0.01).T.groupby(level=0).any()
        p1 = enrs.reorder_levels([1, 0], axis=1)[">"]
        p2 = enrs.reorder_levels([1, 0], axis=1)["<"] * -1
        p3 = p1.fillna(0) + p2.fillna(0)
        p3 = p3.drop(exc_paths)
        s3 = (
            s.groupby(s.index.str.replace(r" - .*", "", regex=True))
            .any()
            .astype(int)
            .drop(exc_paths, axis=1)
        )
        p3 = p3.T.groupby(p3.columns.str.replace(r" - .*", "", regex=True)).mean()

        g = clustermap(
            p3.fillna(0),
            mask=p3.isnull(),
            annot=s3,
            cmap="RdBu_r",
            center=0,
            vmin=-3,
            vmax=3,
            dendrogram_ratio=0.05,
            figsize=(8, 6),
        )
        g.fig.savefig(
            output_dir / "enrichments.MSigDB_Hallmark_2020.organ.heatmap.all.svg",
            **figkws,
        )

        s4 = s3.loc[(s3.abs().gt(0).sum(1) > 5), (s3.abs().gt(0).sum() > 4)]

        g = clustermap(
            p3.loc[s4.index, s4.columns].fillna(0),
            mask=p3.loc[s4.index, s4.columns].isnull(),
            annot=s4,
            cmap="RdBu_r",
            center=0,
            vmin=-3,
            vmax=3,
            dendrogram_ratio=0.05,
            figsize=(5, 5),
        )
        g.fig.savefig(
            output_dir / "enrichments.MSigDB_Hallmark_2020.organ.heatmap.sel.svg",
            **figkws,
        )

        organs = s4.index[g.dendrogram_row.reordered_ind]
        pathways = s4.columns[g.dendrogram_col.reordered_ind]
        n = len(pathways)
        fig, axes = plt.subplots(
            1, n, figsize=(1 * n, 3 * 1), sharey=True, gridspec_kw={"wspace": 0.01}
        )
        for ax, pathway in zip(axes, pathways):
            d = p3.loc[organs, pathway]
            sns.barplot(x=d.values, y=d.index, orient="h", ax=ax)
            ax.axvline(0, color="gray", linestyle="dashed")
            ax.set(title=pathway, xlim=(-3, 3))
        fig.savefig(
            output_dir / "enrichments.MSigDB_Hallmark_2020.organ.barplots.svg",
            **figkws,
        )

        organ_groups = {
            "Integumentary and structural": ["Skin", "Adipose", "Muscle"],
            "Neuroendocrine": [
                "Brain",
                "Nerve",
                "Adrenal Gland",
                "Pituitary",
                "Thyroid",
            ],
            "Cardiovascular, respiratory, and metabolic": [
                "Heart",
                "Artery",
                "Liver",
                "Pancreas",
                "Kidney",
                "Lung",
            ],
            "Digestive organs": ["Esophagus", "Stomach", "Small Intestine", "Colon"],
            "Reproductive organs": ["Ovary", "Uterus", "Vagina", "Prostate", "Testis"],
            # "Respiratory": ["Lung"],
            # "Immune and hematopoietic": ["Spleen"],
            # "Secretory and exocrine glands": ["Breast", "Minor Salivary Gland"]
        }
        m = len(organ_groups)
        lens = [len(g) for g in organ_groups.values()]
        fig, axes = plt.subplots(
            m,
            n,
            figsize=(1 * n, 1.5 * m),
            sharex=True,
            sharey="row",
            gridspec_kw={"wspace": 0.01, "hspace": 0.05, "height_ratios": lens},
        )
        for axs, organ_group in zip(axes, list(organ_groups)):
            for ax, pathway in zip(axs, pathways):
                d = p3.loc[organ_groups[organ_group], pathway].rename_axis(organ_group)
                s = s3.loc[d.index, pathway]
                sns.barplot(x=d.values, y=d.index, orient="h", ax=ax)
                for i in d.index:
                    if s.loc[i] == 1:
                        ax.text(
                            d.loc[i] + 0.1 if d.loc[i] > 0 else d.loc[i] - 0.1,
                            i,
                            "*",
                            color="black",
                            fontsize=12,
                            verticalalignment="center",
                        )
                ax.axvline(0, color="gray", linestyle="dashed")
                ax.set(xlim=(-3, 3))
                if organ_group == list(organ_groups)[0]:
                    ax.set(title=pathway)
        fig.tight_layout()
        fig.supxlabel("Enrichment (Odds Ratio)")
        fig.savefig(
            output_dir / "enrichments.MSigDB_Hallmark_2020.organ_groups.barplots.svg",
            **figkws,
        )
        for axs, organ_group in zip(axes, list(organ_groups)):
            for ax, pathway in zip(axs, pathways):
                d = p3.loc[organ_groups[organ_group], pathway].rename_axis(organ_group)
                s = s3.loc[d.index, pathway]
                sns.barplot(
                    x=d.values,
                    y=d.index,
                    orient="h",
                    ax=ax,
                    hue=d.values,
                    palette="RdBu_r",
                    legend=False,
                    hue_norm=(-3, 3),
                )
                for i in d.index:
                    if s.loc[i] == 1:
                        ax.text(
                            d.loc[i] + 0.1 if d.loc[i] > 0 else d.loc[i] - 0.1,
                            i,
                            "*",
                            color="black",
                            fontsize=12,
                            verticalalignment="center",
                        )
                ax.axvline(0, color="gray", linestyle="dashed")
                ax.set(xlim=(-3, 3))
                if organ_group == list(organ_groups)[0]:
                    ax.set(title=pathway)
        fig.tight_layout()
        fig.supxlabel("Enrichment (Odds Ratio)")
        fig.savefig(
            output_dir
            / "enrichments.MSigDB_Hallmark_2020.organ_groups.barplots.with_color.svg",
            **figkws,
        )
        # # Most consistent changes
        # up = p3.mean() > 0
        # sel_up = p3[up[up].index].std().sort_values().head(12).index
        # dn = p3.mean() < 0
        # sel_dn = p3[dn[dn].index].std().sort_values().head(12).index
        # sel = sel_up.tolist() + sel_dn.tolist()
        # pp = p3.loc[:, sel]
        # g = clustermap(
        #     pp.fillna(0),
        #     mask=pp.isnull(),
        #     cmap="RdBu_r",
        #     center=0,
        #     vmin=-2,
        #     vmax=2,
        #     metric="correlation",
        # )

        # Some specific plots
        comp = p3.T
        tissues = [
            "Heart - Atrial Appendage",
            "Heart - Left Ventricle",
            "Artery - Aorta",
            "Artery - Coronary",
            "Artery - Tibial",
            "Esophagus - Mucosa",
            "Esophagus - Muscularis",
            "Stomach",
            "Small Intestine - Terminal Ileum",
            "Colon - Sigmoid",
            "Colon - Transverse",
            "Pancreas",
            "Liver",
            "Adipose - Subcutaneous",
            "Adipose - Visceral (Omentum)",
            "Skin - Not Sun Exposed (Suprapubic)",
            "Skin - Sun Exposed (Lower leg)",
        ]
        organs = [
            "Heart",
            "Artery",
            "Esophagus",
            "Stomach",
            "Small Intestine",
            "Colon",
            "Pancreas",
            "Liver",
            "Adipose",
            "Skin",
        ]
        pathways = p3.std().sort_values().tail(6).index.tolist()
        pathways += p3.std().sort_values().head(6).index.tolist()
        pathways = comp.loc[pathways].mean(1).sort_values().index
        n = len(pathways)

        fig, axes = plt.subplots(
            1, n, figsize=(1 * n, 3 * 1), sharey=True, gridspec_kw={"wspace": 0.01}
        )
        for ax, pathway in zip(axes, pathways):
            p = comp.index[comp.index.str.contains(pathway)].tolist()
            d = comp.loc[p, organs]
            d = d.reset_index().melt(id_vars=["Term"])
            sns.barplot(data=d, x="value", y="tissue", orient="h", ax=ax)
            ax.axvline(0, color="gray", linestyle="dashed")
            ax.set(title=pathway, xlim=(-3, 3))
        fig.savefig(
            output_dir
            / f"comparison.{label}.enrichr.{gs}.organs.examples.barplots.svg",
            **figkws,
        )


# Plot specific genes
threshold = 0.035
tissues = sorted(meta["Tissue"].unique())
tracker = tqdm(total=4 * 1 * len(tissues) * 5, leave=True, position=0)
for target_var in ["Age", "prediction_adj", "residuals_adj"]:
    # target_var = "residuals_adj"
    for fit_type in ["Ridge"]:
        # fit_type = 'Ridge'
        fits = (
            pd.read_csv(
                input_dir
                / f"{target_var}.express_regression.z.{fit_type}_fit.with_covariates.csv",
                index_col=0,
            )
            .drop(
                [
                    "Intercept",
                    "Age",
                    "Cohort_Surgical",
                    "Cohort_Organ Donor (OPO)",
                    "Cohort_Postmortem",
                    "Sex_Female",
                    "Sex_Male",
                    "Ischemic Time (minutes)",
                ],
                errors="ignore",
            )
            .rename_axis(index="gene")
        )
        tissues = sorted(fits["tissue"].unique())

        sigs = fits.query("mean > 1 & abs(coef) > @threshold").index.unique()

        # s = fits.reset_index().query("gene.isin(@sigs)").pivot_table(
        # index="tissue", columns="gene", values="coef"
        # )
        s = (
            fits.reset_index()
            .query("gene.isin(@sigs)")
            .set_index("gene")
            .groupby("tissue")["coef"]
            .nlargest(2)
            .index.levels[1]
            .tolist()
        )
        s += (
            fits.reset_index()
            .query("gene.isin(@sigs)")
            .set_index("gene")
            .groupby("tissue")["coef"]
            .nsmallest(2)
            .index.levels[1]
            .tolist()
        )
        s = list(set(s))

        p = (
            fits.reset_index()
            .query("gene.isin(@s)")
            .pivot_table(index="tissue", columns="gene", values="coef")
            .fillna(0)
        )
        p = p.drop(exclude)
        p = p.groupby(p.index.str.replace(r" - .*", "", regex=True)).mean()

        g = clustermap(
            p,
            config="abs",
            square=False,
            robust=True,
            metric="cosine",
            cmap="RdBu_r",
            center=0,
            figsize=(16, 6),
            yticklabels=True,
            xticklabels=True,
        )
        g.fig.savefig(
            output_dir
            / f"{target_var}.express_regression.{fit_type}_fit.sig_genes.heatmap.svg",
            **figkws,
        )
