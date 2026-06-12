# SPDX-FileCopyrightText: Copyright (c) 2026 Rendeiro Lab, CeMM - Research Center for Molecular Medicine, Austrian Academy of Sciences
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#
# Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE).

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

from src.utils import get_restricted_info

tqdm().pandas()
pandarallel.initialize(progress_bar=False)


metadata_dir = Path("metadata")
data_dir = Path("data")
expr_dir = Path("data") / "gtex" / "gene_expression"
results_dir = Path("results")
output_dir = (
    results_dir
    / "gtex"
    / "fine_tuned"
    / "_pre_2024-01-19_age_X_frac1.0"
    / "gene_expression"
)
output_dir.mkdir(parents=True, exist_ok=True)

figkws = dict(bbox_inches="tight", dpi=300)

meta = pd.read_csv(Path("data") / "gtex" / "GTEx Portal.csv", index_col=0)
meta["Tissue Simple"] = meta["Tissue"].str.extract(r"(\w+).*", expand=False)
restricted, _ = get_restricted_info()
meta = meta.merge(
    restricted[["Age"]], left_on="Subject ID", right_index=True, how="left"
)


def get_expression():
    annot_f = expr_dir / "GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt"
    if not annot_f.exists():
        import requests

        req = requests.get(
            f"https://storage.googleapis.com/adult-gtex/annotations/v8/metadata-files/{annot_f.name}"
        )
        with annot_f.open("wb") as h:
            h.write(req.content)
    annot = pd.read_table(annot_f, index_col=0)
    annot["Tissue"] = annot["SMTSD"]
    annot["Tissue Simple"] = annot["Tissue"].str.extract(r"(\w+).*", expand=False)
    annot["Subject ID"] = annot.index.str.extract(r"(GTEX-.*?)-.*")[0].values
    annot = annot.merge(
        restricted[["Age", "Sex", "Cohort", "Ischemic Time (Minutes)"]],
        left_on="Subject ID",
        right_index=True,
        how="left",
    )

    if not (expr_dir / "log_cpm.pq").exists():
        e = pd.read_csv(
            expr_dir / "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_reads.gct",
            skiprows=2,
            sep="\t",
            engine="c",
        ).drop("Name", axis=1)
        e = e.groupby("Description").max()
        expr_cpm = ((e / e.sum()) * 1e6).T
        expr_log_cpm: pd.DataFrame = np.log1p(expr_cpm)
        expr_log_cpm.to_parquet(expr_dir / "log_cpm.pq")
    expr_log_cpm = pd.read_parquet(expr_dir / "log_cpm.pq")
    obs = annot.reindex(expr_log_cpm.index)
    obs.to_parquet(expr_dir / "log_cpm.obs.pq")
    return expr_log_cpm, obs


# Get gene expression
exp, annot = get_expression()


# Regress age out (for later use)
if not (expr_dir / "log_cpm.age_regressed.clipped.pq").exists():
    from sklearn.linear_model import Ridge

    tissues = annot["Tissue"].unique()
    tissues = sorted(tissues[annot["Tissue"].value_counts() > 100].tolist())
    _fixed = list()
    # for tissue in tqdm(tissues[tissues.index(tissue):]):
    for tissue in tqdm(tissues):
        samples = annot.query("Tissue == @tissue").index.tolist()
        x_to_fix = exp.loc[samples]
        mean = x_to_fix.mean()
        y_to_fix = annot.loc[samples, "Age"].to_frame()
        model = Ridge(2.5, max_iter=100_000)
        fix = dict()
        for gene in tqdm(x_to_fix.columns):
            model.fit(y_to_fix, x_to_fix[gene])
            fix[gene] = x_to_fix[gene] - model.predict(y_to_fix)
        exp_fixed = pd.DataFrame(fix)
        exp_fixed += mean
        _fixed.append(exp_fixed)
    fixed = pd.concat(_fixed)
    fixed.to_parquet(expr_dir / "log_cpm.age_regressed.pq")
    annot.reindex(fixed.index).to_parquet(expr_dir / "log_cpm.age_regressed.obs.pq")
    fixed = fixed.clip(lower=0, upper=None)
    fixed.to_parquet(expr_dir / "log_cpm.age_regressed.clipped.pq")
    annot.reindex(fixed.index).to_parquet(
        expr_dir / "log_cpm.age_regressed.clipped.obs.pq"
    )

    p = exp.mean(), exp.std()
    f = fixed.mean(), fixed.std()
    fig, axes = plt.subplots(2, 2, figsize=(4 * 2, 4 * 1), sharex=True, sharey=True)
    for ax in axes[0]:
        ax.scatter(p[0], p[1], alpha=0.1, s=5, rasterized=True)
    for ax in axes[1]:
        ax.scatter(f[0], f[1], alpha=0.1, s=5, rasterized=True)
    for ax in axes.flatten():
        ax.set(xlabel="Mean", ylabel="Standard deviation")
    for ax in axes[:, 1]:
        ax.set(xscale="log", yscale="log")
    fig.savefig(
        output_dir / "log_cpm.age_regressed.clipped.mean_vs_std.scatter.svg", **figkws
    )

# Get genes associated with chronological age (Age), biological age (prediction), and histology residuals (residuals_adj)
model_name = "Ridge"
cv_name = "GroupKFold"

s = f"tissue-specific_clocks.{model_name}.{cv_name}."
df = pd.read_parquet(output_dir.parent / (s + "predictions_residuals.pq"))
df["Subject ID"] = df.index.str.extract(r"(GTEX-.*)-\d+")[0].values

covariates = ["Age", "Sex", "Cohort", "Ischemic Time (Minutes)"]
tracker = tqdm(total=12, leave=True, position=0)
for fit_type in ["Ridge"]:  # , "Lasso"
    # fit_type = 'Ridge'
    for target_var in ["Age", "prediction_adj", "residuals_adj"][::-1]:
        # target_var = 'Age'
        # target_var = 'residuals_adj'
        res = df.pivot_table(index="Subject ID", columns="Tissue", values=target_var)
        # Regression of histology residuals based on gene expression
        for z in ["z"]:  # "o"
            _fits = list()
            for tissue in tqdm(res.columns, leave=False, position=1):
                sel = annot.query("Tissue == @tissue")
                x = exp.loc[sel.index]
                x.index = sel["Subject ID"]
                x = x.groupby(level=0).mean()
                y = res.reindex(x.index)[tissue].dropna()
                if "residuals" in target_var:
                    y = (
                        y.where(y > y.quantile(0.05))
                        .where(y < y.quantile(0.95))
                        .dropna()
                    )
                x = x.reindex(y.index)
                x = x.loc[:, (x.var().fillna(0) > 1e-5) & (x.mean() > 0.1)]
                if covariates:
                    covs = pd.get_dummies(restricted[covariates])
                    x = x.join(covs, how="left")
                mean = x.mean()
                if z == "z":
                    x = (x - x.mean()) / x.std()
                x = x.loc[:, (x.var().fillna(0) > 0)]
                model = getattr(sklearn.linear_model, fit_type)(2.5, max_iter=100_000)
                model.fit(x, y)
                fit = pd.DataFrame(
                    dict(
                        mean=mean,
                        coef=pd.Series(model.coef_, x.columns),
                        tissue=tissue,
                    )
                )
                fit.loc["Intercept", "coef"] = model.intercept_
                fit.loc["Intercept", "tissue"] = tissue
                _fits.append(fit)
            fits = pd.concat(_fits).rename_axis(index="gene")
            fits.to_csv(
                output_dir
                / f"{target_var}.express_regression.{z}.{fit_type}_fit.with_covariates.csv"
            )
            tracker.update(1)

for fit_type in ["Ridge", "Lasso"]:
    # fit_type = 'Ridge'
    for target_var in ["Age", "prediction", "residuals", "residuals_adj"]:
        # target_var = "residuals_adj"
        o_fits = pd.read_csv(
            output_dir / f"{target_var}.express_regression.o.{fit_type}_fit.csv",
            index_col=0,
        )
        z_fits = pd.read_csv(
            output_dir / f"{target_var}.express_regression.z.{fit_type}_fit.csv",
            index_col=0,
        )
        tissues = sorted(o_fits["tissue"].unique())

        # Compare fits with and without z-score
        fig, axes = plt.subplots(
            4, 10, figsize=(10 * 3, 4 * 3), sharex=True, sharey=True
        )
        for ax, tissue in tqdm(zip(axes.flatten(), tissues)):
            n = z_fits.query("tissue == @tissue")["coef"]
            o = o_fits.query("tissue == @tissue")["coef"].reindex(n.index)
            m = o_fits.query("tissue == @tissue")["mean"].reindex(n.index)
            ax.scatter(o, n, c=m, s=1, alpha=0.25, rasterized=True, cmap="magma")
            ax.set(title=tissue)
        fig.savefig(
            output_dir
            / f"{target_var}.express_regression.{fit_type}_fit.o_z_comparison.scatter.svg",
            **figkws,
        )

        # MA plots
        fig, axes = plt.subplots(
            4, 10, figsize=(10 * 3, 4 * 3), sharex=True, sharey=True
        )
        for ax, tissue in tqdm(zip(axes.flatten(), tissues)):
            f = z_fits.query("tissue == @tissue")
            ax.axvline(1, linestyle="--", color="grey", linewidth=0.5)
            ax.axhline(0.0075, linestyle="--", color="grey", linewidth=0.5)
            ax.axhline(-0.0075, linestyle="--", color="grey", linewidth=0.5)
            ax.scatter(f["mean"], f["coef"], s=1, alpha=0.25, rasterized=True)
            ax.set(title=tissue)
        for ax in axes[:, 0]:
            ax.set(ylabel="Coef")
        for ax in axes[-1, :]:
            ax.set(xlabel="Mean")
        fig.savefig(
            output_dir
            / f"{target_var}.express_regression.z.{fit_type}.ma_plots_per_tissue.svg",
            **figkws,
        )


# # Overview across tissues
# # # 'Most aged' tissues (this was replaced by the cross-prediction strategy implemented in another script)
# # # # Option 1. Intercepts of pan-tissue clock (somehow not run for Ridge)
# s = f"pan-tissue_clock.LinearRegression.{cv_name}."
# clock_coefs = pd.read_parquet(output_dir.parent / (s + "with_tissue.coefficients.pq"))
# q1 = clock_coefs.loc[clock_coefs.index.str.startswith("Tissue_"), "original"]
# q1 = (q1 - q1.mean()).sort_values()
# q1.index = q1.index.str.replace("Tissue_", "")

# # # # Option 1b. Residuals of pan-tissue clock
# s = f"pan-tissue_clock.Ridge.{cv_name}."
# clock_metrics = pd.read_parquet(output_dir.parent / (s + "metrics.pq"))
# clock_coefs = pd.read_parquet(output_dir.parent / (s + "coefficients.pq"))
# clock_preds = pd.read_parquet(output_dir.parent / (s + "predictions_residuals.pq"))
# q2 = (
#     clock_preds.join(meta["Tissue"])
#     .groupby("Tissue")["residuals_adj"]
#     .mean()
#     .sort_values(ascending=False)
# )
# q2 += abs(q2.min())

# # # Option 2. Post hoc ratio prediction / actual
# q3 = np.log(
#     df.groupby("Tissue")["prediction"].mean() / df.groupby("Tissue")["Age"].mean()
# ).sort_values()

# # # Option 3. Aggregated residuals
# q4 = df.groupby("Tissue")["residuals_adj"].mean().sort_values()

# # q = pd.DataFrame([q1, q2, q3, q4]).T
# q = q2


# diffs = (
#     fits.query("mean > 1 & abs(coef) > 0.005")["tissue"].value_counts().reindex(q.index)
# )

# fig, axes = plt.subplots(1, 2, figsize=(3 * 2, 6))
# sns.barplot(x=q, y=q.index, orient="horiz", ax=axes[0])
# sns.barplot(x=diffs, y=diffs.index, orient="horiz", ax=axes[1])
# fig.savefig(
#     output_dir / "express_regression.ridge_fit.significant_per_tissue.barplot.svg",
#     **figkws,
# )
# axes[1].set_xscale("log")
# fig.savefig(
#     output_dir / "express_regression.ridge_fit.significant_per_tissue.barplot.log.svg",
#     **figkws,
# )

# Check expression
for target_var in ["Age", "prediction_adj", "residuals_adj"]:
    # target_var = "residuals_adj"
    for fit_type in ["Ridge"]:
        # fit_type = 'Ridge'
        fits = pd.read_csv(
            output_dir
            / f"{target_var}.express_regression.z.{fit_type}_fit.with_covariates.csv",
            index_col=0,
        )
        tissues = sorted(fits["tissue"].dropna().unique())

        # Top genes
        v = max(1e-25, fits["coef"].abs().quantile(0.995))
        genes = (
            fits.drop(covariates + ["Intercept"], errors="ignore")
            .query("mean > 1 & abs(coef) > @v")
            .index.unique()
            .tolist()
        )
        samples = (
            annot.groupby("Tissue Simple")
            .sample(n=20, replace=True)
            .sort_values(["Tissue Simple", "Age"])
            .index.unique()
        )
        for config in ["abs", "z"]:
            g = clustermap(
                exp.loc[samples, genes],
                config=config,
                metric="cosine",
                row_colors=annot.loc[samples, ["Tissue", "Tissue Simple", "Age"]],
                col_colors=exp.loc[:, genes].mean().rename("Mean expression"),
                square=False,
                figsize=(28, 20),
                # row_cluster=False
            )
            g.ax_heatmap.set_rasterized(True)
            g.fig.savefig(
                output_dir
                / f"{target_var}.express_regression.z.{fit_type}_fit.top_genes.expression.clustermap.{config}.svg",
                **figkws,
            )

        s = f"tissue-specific_clocks.{model_name}.{cv_name}."
        df = pd.read_parquet(output_dir.parent / (s + "predictions_residuals.pq"))
        for tissue in tqdm(tissues):
            fits2 = fits.query("tissue == @tissue").drop(
                covariates + ["Intercept"], errors="ignore"
            )
            # Top genes
            v = max(1e-25, fits2["coef"].abs().quantile(0.985))
            genes = (
                fits2.query("mean > 0.1 & abs(coef) > @v")
                .sort_values("coef")
                .index.tolist()
            )
            # sort by age (no pattern)
            samples = annot.query("Tissue == @tissue").sort_values(["Age"]).index
            # sort by residual (linear pattern)
            samples = df.query("Tissue == @tissue").sort_values([target_var]).index
            samples_sel = annot.query("Tissue == @tissue").index
            genes_sel = fits2.query("mean > 1").index

            # regress age out from gene expression
            if target_var != "Age":
                x_to_fix = exp.loc[samples_sel, genes_sel]
                mean = x_to_fix.mean()
                y_to_fix = annot.loc[samples_sel, "Age"].to_frame()

                from sklearn.linear_model import Ridge

                model = Ridge(2.5, max_iter=100_000)
                fixed = dict()
                for gene in tqdm(genes_sel):
                    model.fit(y_to_fix, x_to_fix[gene])
                    fixed[gene] = x_to_fix[gene] - model.predict(y_to_fix)
                exp_fixed = pd.DataFrame(fixed)
                exp_fixed += mean
            else:
                exp_fixed = exp.loc[samples_sel, genes_sel].copy()

            genes = [g for g in genes if g in exp_fixed.columns]
            exp_fixed = exp_fixed.loc[:, genes].copy()
            exp_fixed.index = exp_fixed.index.str.extract(r"(GTEX-.*)-SM-\d+")[0].values
            exp_fixed = exp_fixed.groupby(level=0).mean()
            exp_fixed = exp_fixed.reindex(samples).dropna()
            if exp_fixed.empty:
                continue

            # Plot residuals vs gene expression for a few genes (top 3 up, top 3 down)
            df2 = df.query("Tissue == @tissue").copy()
            if "residuals" in target_var:
                y = df2[target_var]
                y = y.where(y > y.quantile(0.02)).where(y < y.quantile(0.98))
                df2[target_var] = y

            n_genes = 10
            fig, axes = plt.subplots(2, n_genes, figsize=(n_genes * 4, 2 * 4))
            for ax, gene in zip(
                axes[0].tolist() + axes[1].tolist(), genes[:n_genes] + genes[-n_genes:]
            ):
                ax.scatter(
                    df2.loc[samples, target_var].reindex(exp_fixed.index),
                    exp_fixed[gene],
                    alpha=0.25,
                    s=5,
                )
                ax.text(
                    0.01,
                    0.99,
                    f"{fits2.loc[gene, 'coef']:.3f}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                )
                sns.regplot(
                    x=df2.loc[samples, target_var].reindex(exp_fixed.index),
                    y=exp_fixed[gene],
                    ax=ax,
                    scatter=False,
                    color="black",
                )
                ax.set(xlabel=target_var, ylabel=gene)
            fig.suptitle(f"{tissue} - {target_var} - {fit_type}")
            fig.savefig(
                output_dir
                / f"{target_var}.express_regression.z.{fit_type}_fit.top_genes.expression.residuals.{tissue}.svg",
                **figkws,
            )

        # Check coefficient specificity
        cs = fits.pivot_table(
            index="Description", columns="tissue", values="coef", fill_value=0
        )
        g = clustermap(
            cs.loc[genes, :].T,
            cmap="coolwarm",
            center=0,
            col_colors=exp.loc[:, genes].mean().rename("Mean expression"),
            square=False,
            figsize=(28, 20),
        )
        g.ax_heatmap.set_rasterized(True)
        g.fig.savefig(
            output_dir
            / f"{target_var}.express_regression.z.{fit_type}_fit.top_genes.coefficients.clustermap.abs.svg",
            **figkws,
        )


# Plot jointly summaries of clock performance, number of genes differentially expressed
model_name = fit_type = "Ridge"
cv_name = "GroupKFold"
s = f"tissue-specific_clocks.{model_name}.{cv_name}."
df = pd.read_parquet(output_dir.parent / (s + "predictions_residuals.pq"))
df["Subject ID"] = df.index.str.extract(r"(GTEX-.*)-\d+")[0].values
res_adj = df.pivot_table(index="Subject ID", columns="Tissue", values="residuals_adj")
s = f"pan-tissue_clock.{model_name}.{cv_name}."
dfg = pd.read_parquet(output_dir.parent / (s + "predictions_residuals.pq"))
dfg["Subject ID"] = dfg.index.str.extract(r"(GTEX-.*)-\d+")[0].values
pt = dfg.join(meta["Tissue"]).pivot_table(
    index="Subject ID", columns="Tissue", values="residuals_adj"
)

mae = res_adj.abs().mean()
n = res_adj.notnull().sum().reindex(mae.index)

mae["pan_tissue"] = np.nanmean(pt.abs().values)
mae["pan_tissue_mean_indiv"] = pt.mean(1).abs().mean()
n["pan_tissue"] = pt.shape[0]
n["pan_tissue_mean_indiv"] = pt.shape[0]

mae = mae.sort_values(ascending=False)

threshold = 0.005
age_fits = pd.read_csv(
    output_dir / f"Age.express_regression.z.{fit_type}_fit.csv", index_col=0
)
age = (
    age_fits.query("mean > 1 & abs(coef) > @threshold")["tissue"]
    .value_counts()
    .reindex(mae.index)
    .dropna()
)
res_f = pd.read_csv(
    output_dir / f"residuals_adj.express_regression.z.{fit_type}_fit.csv", index_col=0
)
res = (
    res_f.query("mean > 1 & abs(coef) > @threshold")["tissue"]
    .value_counts()
    .reindex(mae.index)
    .dropna()
)


order = mae.reindex(age.index).dropna().index

fig, axes = plt.subplots(
    1,
    3 + 1,
    figsize=(3 * 2, 6),
    sharey=True,
    sharex=False,
    gridspec_kw=dict(width_ratios=[1, 1, 1, 0.2], wspace=0.01),
)
for ax, (y, label) in zip(
    axes,
    [
        (mae, "Mean absolute error (years)"),
        (age, "Genes changing with age"),
        (res, "Genes changing with histology age gap"),
    ],
):
    ax.barh(order, y.loc[order])
    ax.set(xlabel=label)
for ax in axes[1:-1]:
    ax.set(xscale="log")
    ax.set_xlim(left=1)

for ax in axes[1:-1]:
    y_major = matplotlib.ticker.LogLocator(base=10.0, numticks=5)
    ax.xaxis.set_major_locator(y_major)
    y_minor = matplotlib.ticker.LogLocator(
        base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10
    )
    ax.xaxis.set_minor_locator(y_minor)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

axes[2].set_xlim(axes[1].get_xlim())
axes[-1].barh(order, n.loc[order])
axes[-1].set(xlabel="Number of samples")

fig.savefig(
    output_dir
    / "express_regression.ridge_fit.performance_genes_per_tissue.barplot.svg",
    **figkws,
)


# Enrichment

from tenacity import retry, stop_after_attempt


@retry(stop=stop_after_attempt(3))
def enrichr(genes: list[str], gene_sets: list[str]) -> pd.DataFrame:
    import gseapy

    r = gseapy.enrichr(genes, gene_sets=gene_sets)
    return r.results


gene_sets = [
    "MSigDB_Hallmark_2020",
    "NCI-Nature_2016",
    "KEGG_2021_Human",
    "WikiPathway_2023_Human",
    "GO_Biological_Process_2023",
    "GO_Molecular_Function_2023",
    "GO_Cellular_Component_2023",
    "MSigDB_Oncogenic_Signatures",
    "Human_Phenotype_Ontology",
    "OMIM_Disease",
    "MGI_Mammalian_Phenotype_Level_4_2021",
    "UK_Biobank_GWAS_v1",
    "Aging_Perturbations_from_GEO_down",
    "Aging_Perturbations_from_GEO_up",
    "ARCHS4_Tissues",
    "BioPlanet_2019",
    "GTEx_Tissues_V8_2023",
    "Tabula_Sapiens",
]
tissues = sorted(meta["Tissue"].unique())
tracker = tqdm(total=4 * 1 * len(tissues) * 5, leave=True, position=0)
for target_var in ["Age", "prediction", "residuals", "residuals_adj"]:
    # target_var = "residuals_adj"
    for fit_type in ["Ridge"]:
        # fit_type = 'Ridge'
        fits = pd.read_csv(
            output_dir
            / f"{target_var}.express_regression.z.{fit_type}_fit.with_covariates.csv",
            index_col=0,
        )
        tissues = sorted(fits["tissue"].unique())

        # Based on percentile threshold per tissue, ignore direction
        _enr = list()
        for tissue in tissues:
            # tissue = tissues[0]
            v = max(1e-25, fits["coef"].abs().quantile(0.985))
            genes = fits.query(
                f"tissue == @tissue & mean > 1 & abs(coef) > {v}"
            ).index.tolist()
            tracker.update(1)
            if len(genes) < 5:
                continue
            r = enrichr(genes, gene_sets=gene_sets)
            _enr.append(r.assign(tissue=tissue))
        enr = pd.concat(_enr)
        enr.to_csv(
            output_dir
            / f"{target_var}.express_regression.{fit_type}_fit.abs_top_genes.enrichr.csv"
        )

        # Based on percentile threshold per tissue
        _enr = list()
        for tissue in tissues:
            # tissue = tissues[0]
            for sign, direction in [("", ">"), ("-", "<")]:
                # sign, direction = "", ">"
                v = max(1e-25, fits["coef"].abs().quantile(0.985))
                genes = fits.query(
                    f"tissue == @tissue & mean > 1 & coef {direction} {sign}{v}"
                ).index.tolist()
                tracker.update(1)
                if len(genes) < 5:
                    continue
                r = enrichr(genes, gene_sets=gene_sets)
                _enr.append(r.assign(tissue=tissue, direction=direction))
        enr = pd.concat(_enr)
        enr.to_csv(
            output_dir
            / f"{target_var}.express_regression.{fit_type}_fit.top_genes.enrichr.csv"
        )

        # Simply top 200 up and down
        for n_genes in [200, 50, 10]:
            _enr = list()
            # for tissue in tissues[tissues.index(tissue) + 1 :]:
            for tissue in tqdm(tissues):
                for sign, direction in [("", ">"), ("-", "<")]:
                    rs = fits.query(
                        f"tissue == @tissue & mean > 1 & coef {direction}0"
                    ).sort_values("coef")
                    genes = (
                        rs.tail(n_genes) if sign else rs.head(n_genes)
                    ).index.tolist()
                    r = enrichr(genes, gene_sets=gene_sets)
                    _enr.append(r.assign(tissue=tissue, direction=direction))
                    tracker.update(1)
            enr = pd.concat(_enr)
            enr.to_csv(
                output_dir
                / f"{target_var}.express_regression.{fit_type}_fit.top_{n_genes}_genes.enrichr.csv"
            )

        # Based on fixed threshold across whole dataset, no direction
        if (
            output_dir
            / f"{target_var}.express_regression.{fit_type}_fit.sig_genes.enrichr.csv"
        ).exists():
            continue
        _enr = list()
        for tissue in tissues:
            rs = fits.query(
                "tissue == @tissue & mean > 1 & abs(coef) > @threshold"
            ).sort_values("coef")
            genes = rs.index.tolist()
            if not genes:
                continue
            r = enrichr(genes, gene_sets=gene_sets)
            _enr.append(r.assign(tissue=tissue))
            tracker.update(1)
        enr = pd.concat(_enr)
        enr.to_csv(
            output_dir
            / f"{target_var}.express_regression.{fit_type}_fit.sig_genes.enrichr.csv"
        )

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


# q = enr.pivot_table(index=['Gene_set', 'Term'], columns=['tissue', 'direction'], values='Combined Score').fillna(0).loc["MSigDB_Hallmark_2020"]


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
    "abs_top_genes",
    "top_genes",
    "top_200_genes",
    "top_50_genes",
    "top_10_genes",
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
        for gs in [gene_sets]:
            enrs = enr.query("Gene_set == @gs")
            enrs = (
                enrs.groupby(gb + ["Gene_set", "Term"])["Odds Ratio"]
                .mean()
                .reset_index()
            )
            enrp = enrs.pivot_table(
                index=["Term"],
                columns=gb,
                values="Odds Ratio",
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

                p.columns = (
                    p.columns.get_level_values(1) + " " + p.columns.get_level_values(0)
                )
            g = clustermap_marsilea(p, metric="correlation", square=True)
            g.fig.savefig(of, **figkws)

            # # Examples
            # for top_n in [2, 5, 10]:
            #     of = (
            #         output_dir
            #         / f"{target_var}.express_regression.{fit_type}_fit.{label}.enrichr.{gs}.clustermap.top_{top_n}.svg"
            #     )
            #     # if of.exists():
            #     #     continue
            #     sel = (
            #         enrs.set_index(["Term"])
            #         .groupby(gb)["Combined Score"]
            #         .nlargest(top_n)
            #         .unstack(gb)
            #         .index
            #     )
            #     tp = enrp.loc[sel].fillna(0)
            #     if tp.shape[1] < 2:
            #         continue
            #     try:
            #         g = clustermap_marsilea(
            #             tp,
            #             config="abs",
            #             square=True,
            #             robust=True,
            #             metric="cosine",
            #         )
            #     except ValueError:
            #         g = clustermap(
            #             tp,
            #             config="abs",
            #             square=True,
            #             robust=True,
            #             metric="cosine",
            #         )
            #     g.fig.savefig(of, **figkws)

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
from seaborn_extensions import clustermap

for gs in ["MSigDB_Hallmark_2020"]:
    _comp = list()
    for target_var in ["Age", "residuals_adj"]:
        file = (
            output_dir
            / f"{target_var}.express_regression.{fit_type}_fit.{label}.enrichr.csv"
        )
        if not file.exists():
            continue
        enr = (
            pd.read_csv(file, index_col=0)
            .query("~tissue.isin(@exclude)")
            .query("Gene_set == @gs")
        )
        gb = ["tissue", "direction"] if "direction" in enr.columns else ["tissue"]
        enrs = (
            enr.groupby(gb + ["Gene_set", "Term"])["Odds Ratio"]
            .mean()
            .reset_index()
            .pivot_table(index=["Term"], columns=gb, values="Odds Ratio")
        )
        p1 = enrs.reorder_levels([1, 0], axis=1)[">"]
        p2 = enrs.reorder_levels([1, 0], axis=1)["<"] * -1
        p3 = p1.fillna(0) + p2.fillna(0)
        _comp.append(p3.assign(target_var=target_var))
    comp = pd.concat(_comp)
    comp = comp.loc[~comp.index.isin(exc_paths)]

    # Some specific plots
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

    fig, axes = plt.subplots(
        1, 3, figsize=(3 * 1, 3 * 1), sharey=True, gridspec_kw={"wspace": 0.01}
    )
    for ax, pathways in zip(axes, ["p53", "Interferon", "Cholesterol"]):
        p = comp.index[comp.index.str.contains(pathways)].tolist()
        d = comp.loc[p, tissues + ["target_var"]]
        d = d.reset_index().melt(id_vars=["target_var", "Term"])
        sns.barplot(data=d, x="value", y="tissue", hue="target_var", orient="h", ax=ax)
        ax.axvline(0, color="gray", linestyle="dashed")
        ax.set(title=pathways)
    fig.savefig(
        output_dir / f"comparison.{label}.enrichr.{gs}.tissues.examples.barplots.svg",
        **figkws,
    )

    for var1, var2, name in [
        ("Age", "prediction", "Age_vs_prediction"),
        ("Age", "residuals_adj", "Age_vs_residuals_adj"),
    ]:
        of = output_dir / f"comparison.{label}.enrichr.{gs}.{name}.SUFFIX"
        v1 = comp.loc[comp["target_var"] == var1].drop("target_var", axis=1)
        v2 = comp.loc[comp["target_var"] == var2].drop("target_var", axis=1)
        v2 = v2.loc[:, v2.var() > 0]
        v1 = v1[v2.columns]

        kwargs = dict(
            metric="correlation",
            cmap="coolwarm",
            center=0,
            vmin=-7.5,
            vmax=7.5,
            dendrogram_ratio=0.05,
            figsize=(6.5, 9),
        )
        g = clustermap(v1.fillna(0), mask=v1.isnull(), **kwargs)
        g.fig.savefig(of.with_suffix(f".clustermap.{var1}.svg"), **figkws)

        g = clustermap(
            v2.reindex(index=v1.index, columns=v2.columns).fillna(0),
            mask=v2.reindex(index=v1.index, columns=v2.columns).isnull(),
            **kwargs,
            row_linkage=g.dendrogram_row.linkage,
            col_linkage=g.dendrogram_col.linkage,
        )
        g.fig.savefig(
            of.with_suffix(f".clustermap.{var2}.clustered_as_{var1}.svg"), **figkws
        )

        g = clustermap(
            v2.reindex(index=v1.index, columns=v2.columns).fillna(0),
            mask=v2.reindex(index=v1.index, columns=v2.columns).isnull(),
            **kwargs,
        )
        g.fig.savefig(
            of.with_suffix(f".clustermap.{var2}.clustered_independently.svg"), **figkws
        )

        # Across tissues, what is more driven by biological vs chronological age?
        v1m = v1.mean(1).rename(var1)
        v2m = v2.mean(1).rename(var2)
        j = v1m.to_frame().join(v2m)
        j["Biological age"] = j[var1] + j[var2]
        j[f"{var1}_std"] = v1.std(1)
        j[f"{var2}_std"] = v2.std(1)
        j["mean"] = j[[var1, "Biological age"]].mean(1)
        j["diff"] = j[var2] - j["mean"]

        fig, axes = plt.subplots(1, 2, figsize=(2 * 3, 3))
        vmin = j[[var1, "Biological age"]].min(1).min()
        vmax = j[[var1, "Biological age"]].max(1).max()
        for ax in axes:
            ax.plot([vmin, vmax], [vmin, vmax], color="grey", linestyle="--")

        axes[0].scatter(
            j[var1],
            j["Biological age"],
            s=10 + 10 ** (1 / j[f"{var1}_std"]),
            edgecolor="black",
            linewidth=0.5,
        )
        for i in j.index:
            axes[0].annotate(i, (j.loc[i, var1], j.loc[i, "Biological age"]))
        axes[0].set(xlabel=var1, ylabel="Biological age")

        axes[1].scatter(
            j[var1],
            j[var2],
            s=10 + 10 ** (1 / j[f"{var1}_std"]),
            edgecolor="black",
            linewidths=0.5,
        )
        for i in j.index:
            axes[1].annotate(i, (j.loc[i, var1], j.loc[i, var2]))
        axes[1].set(xlabel=var1, ylabel=var2)
        for ax in axes:
            ax.axvline(0, color="grey", linestyle="--")
            ax.axhline(0, color="grey", linestyle="--")
            ax.set_aspect("equal")
        fig.savefig(of.with_suffix(".comparison.scatter.svg"), **figkws)

        for ax in axes:
            # from src.utils import MinorSymLogLocator
            ax.set_xscale("symlog", linthresh=0.25)
            ax.set_yscale("symlog", linthresh=0.25)
            ax.yaxis.set_minor_locator(MinorSymLogLocator(1e-1))
            ax.xaxis.set_minor_locator(MinorSymLogLocator(1e-1))
        fig.savefig(of.with_suffix(".comparison.scatter.symlog.svg"), **figkws)

        # fig, ax = plt.subplots(figsize=(2, 6))
        # ax.axvline(0, color="grey", linestyle="--")
        # s = j["diff"].sort_values()
        # ax.scatter(
        #     x=s,
        #     y=s.index,
        #     s=10
        #     + 10 ** (1 / j[[f"{var1}_std", f"{var2}_std"]].max(1).reindex(s.index)),
        # )
        # fig.savefig(of.with_suffix(".rank_vs_value.sorted_pathways.svg"), **figkws)


# label = "top_50_genes"
# for gs in gene_sets:
#     _comp = list()
#     for target_var in ["Age", "prediction", "residuals", "residuals_adj"]:
#         file = (
#             output_dir
#             / f"{target_var}.express_regression.{fit_type}_fit.{label}.enrichr.csv"
#         )
#         if not file.exists():
#             continue
#         enr = (
#             pd.read_csv(file, index_col=0)
#             .query("~tissue.isin(@exclude)")
#             .query("Gene_set == @gs")
#         )
#         enrs = (
#             enr.groupby(gb + ["Gene_set", "Term"])["Odds Ratio"]
#             .mean()
#             .reset_index()
#             .pivot_table(index=["Term"], columns=gb, values="Odds Ratio")
#             .assign(target_var=target_var)
#         )
#         _comp.append(enrs)
#     comp = pd.concat(_comp)

#     for var1, var2, name in [
#         ("Age", "prediction", "Age_vs_prediction"),
#         ("Age", "residuals_adj", "Age_vs_residuals_adj"),
#     ]:
#         of = output_dir / f"comparison.{label}.enrichr.{gs}.clustermap.{name}.svg"
#         v1 = comp.loc[comp['target_var'] == var1].drop("target_var", axis=1)
#         v2 = comp.loc[comp['target_var'] == var2].drop("target_var", axis=1)
#         fc = np.log(v2 / v1).stack((0, 1)).sort_values()

#         p = fc.unstack('Term')
#         p = p.loc[p.var(1) > 0, p.var(0) > 0]
#         g = clustermap_marsilea(p.fillna(0), metric="correlation", square=True, mask=p.isnull(), cmap='coolwarm')


# Plot some particular cases
fit_type = "Ridge"
label = "top_200_genes"
cases = [
    {
        "name": "Soft organs-MSigDB",
        "target_var": "Age",
        "tissues": [
            "Pancreas",
            "Lung",
            "Liver",
            "Spleen",
        ],
        "gene_sets_libraries": ["MSigDB_Hallmark_2020"],
        "value": "Odds Ratio",
        "n": 1,
    },
    {
        "name": "Soft organs-MSigDB",
        "target_var": "residuals_adj",
        "tissues": [
            "Pancreas",
            "Lung",
            "Liver",
            "Spleen",
        ],
        "gene_sets_libraries": ["MSigDB_Hallmark_2020"],
        "value": "Odds Ratio",
        "n": 1,
    },
    {
        "name": "Vasculature-MSigDB",
        "target_var": "Age",
        "tissues": [
            "Artery - Aorta",
            "Artery - Coronary",
            "Artery - Tibial",
            "Heart - Left Ventricle",
        ],
        "gene_sets_libraries": ["MSigDB_Hallmark_2020"],
        "value": "Odds Ratio",
        "n": 1,
    },
    {
        "name": "Vasculature-MSigDB",
        "target_var": "residuals_adj",
        "tissues": [
            "Artery - Aorta",
            "Artery - Coronary",
            "Artery - Tibial",
            "Heart - Left Ventricle",
        ],
        "gene_sets_libraries": ["MSigDB_Hallmark_2020"],
        "value": "Odds Ratio",
        "n": 1,
    },
    {
        "name": "Skin-MSigDB",
        "target_var": "Age",
        "tissues": [
            "Skin - Sun Exposed (Lower leg)",
            "Skin - Not Sun Exposed (Suprapubic)",
        ],
        "gene_sets_libraries": ["MSigDB_Hallmark_2020"],
        "value": "Odds Ratio",
        "n": 1,
    },
    {
        "name": "Skin-MSigDB",
        "target_var": "residuals_adj",
        "tissues": [
            "Skin - Sun Exposed (Lower leg)",
            "Skin - Not Sun Exposed (Suprapubic)",
        ],
        "gene_sets_libraries": ["MSigDB_Hallmark_2020"],
        "value": "Odds Ratio",
        "n": 1,
    },
    {
        "name": "GI_tract-MSigDB",
        "target_var": "Age",
        "tissues": [
            "Esophagus - Mucosa",
            "Stomach",
            "Colon - Transverse",
            "Colon - Sigmoid",
        ],
        "gene_sets_libraries": ["MSigDB_Hallmark_2020"],
        "value": "Odds Ratio",
        "n": 1,
    },
    {
        "name": "GI_tract-MSigDB",
        "target_var": "residuals_adj",
        "tissues": [
            "Esophagus - Mucosa",
            "Stomach",
            "Colon - Transverse",
            "Colon - Sigmoid",
        ],
        "gene_sets_libraries": ["MSigDB_Hallmark_2020"],
        "value": "Odds Ratio",
        "n": 1,
    },
    {
        "name": "Soft organs",
        "target_var": "Age",
        "tissues": [
            "Pancreas",
            "Lung",
            "Liver",
            "Spleen",
        ],
        "gene_sets_libraries": [
            "MSigDB_Hallmark_2020",
            "GO_Biological_Process_2023",
            "WikiPathway_2023_Human",
        ],
        "value": "Odds Ratio",
        "n": 1,
    },
    {
        "name": "Soft organs",
        "target_var": "residuals_adj",
        "tissues": [
            "Pancreas",
            "Lung",
            "Liver",
            "Spleen",
        ],
        "gene_sets_libraries": [
            "MSigDB_Hallmark_2020",
            "GO_Biological_Process_2023",
            "WikiPathway_2023_Human",
        ],
        "value": "Odds Ratio",
        "n": 1,
    },
    {
        "name": "Vasculature",
        "target_var": "Age",
        "tissues": [
            "Artery - Aorta",
            "Artery - Coronary",
            "Artery - Tibial",
            "Heart - Left Ventricle",
        ],
        "gene_sets_libraries": [
            "MSigDB_Hallmark_2020",
            "GO_Biological_Process_2023",
            "WikiPathway_2023_Human",
        ],
        "value": "Odds Ratio",
        "n": 1,
    },
    {
        "name": "Vasculature",
        "target_var": "residuals_adj",
        "tissues": [
            "Artery - Aorta",
            "Artery - Coronary",
            "Artery - Tibial",
            "Heart - Left Ventricle",
        ],
        "gene_sets_libraries": [
            "MSigDB_Hallmark_2020",
            "GO_Biological_Process_2023",
            "WikiPathway_2023_Human",
        ],
        "value": "Odds Ratio",
        "n": 1,
    },
    {
        "name": "Skin",
        "target_var": "Age",
        "tissues": [
            "Skin - Sun Exposed (Lower leg)",
            "Skin - Not Sun Exposed (Suprapubic)",
        ],
        "gene_sets_libraries": [
            "MSigDB_Hallmark_2020",
            "GO_Biological_Process_2023",
            "WikiPathway_2023_Human",
        ],
        "value": "Odds Ratio",
        "n": 1,
    },
    {
        "name": "Skin",
        "target_var": "residuals_adj",
        "tissues": [
            "Skin - Sun Exposed (Lower leg)",
            "Skin - Not Sun Exposed (Suprapubic)",
        ],
        "gene_sets_libraries": [
            "MSigDB_Hallmark_2020",
            "GO_Biological_Process_2023",
            "WikiPathway_2023_Human",
        ],
        "value": "Odds Ratio",
        "n": 1,
    },
    {
        "name": "GI_tract",
        "target_var": "Age",
        "tissues": [
            "Esophagus - Mucosa",
            "Stomach",
            "Colon - Transverse",
            "Colon - Sigmoid",
        ],
        "gene_sets_libraries": [
            "MSigDB_Hallmark_2020",
            "GO_Biological_Process_2023",
            "WikiPathway_2023_Human",
        ],
        "value": "Odds Ratio",
        "n": 1,
    },
    {
        "name": "GI_tract",
        "target_var": "residuals_adj",
        "tissues": [
            "Esophagus - Mucosa",
            "Stomach",
            "Colon - Transverse",
            "Colon - Sigmoid",
        ],
        "gene_sets_libraries": [
            "MSigDB_Hallmark_2020",
            "GO_Biological_Process_2023",
            "WikiPathway_2023_Human",
        ],
        "value": "Odds Ratio",
        "n": 1,
    },
]
for case in cases:
    # case = cases[6]
    name, target_var, tissues, gene_sets_libraries, value, n = case.values()
    enr = pd.read_csv(
        output_dir
        / f"{target_var}.express_regression.{fit_type}_fit.{label}.enrichr.csv",
        index_col=0,
    )
    enrs = enr.query(
        "Gene_set.isin(@gene_sets_libraries) & tissue.isin(@tissues)"
    ).copy()
    enrs["tissue"] = pd.Categorical(enrs["tissue"], categories=tissues, ordered=True)
    # gb1 = ["tissue", "Gene_set", "Term"]
    # if "direction" in enrs.columns:
    #     gb1.append("direction")
    # enrs = enrs.groupby(gb1, observed=True)[value].mean().reset_index()
    # enrp = enrs.pivot_table(index=gb, columns=["tissue"], values=value)
    gb = ["Gene_set", "Term"] if len(gene_sets_libraries) > 1 else ["Term"]
    sel = (
        enrs.set_index(gb)
        .groupby(["tissue"], observed=True)[value]
        .nlargest(n)
        .unstack(["tissue"])
        .index.unique()
    )
    p = (
        enrs.set_index(gb)
        .loc[sel]
        .reset_index()
        .melt(id_vars=["tissue", "Term"], value_vars=value, value_name="value")
        .sort_values("tissue")
    )

    # # sort
    # ts = p.pivot_table(
    #     index="Term", columns="tissue", values="value"
    # )  # .reset_index().melt(id_vars=["Term"]).sort_values("Term", ascending=True).dropna()
    # ordered = (
    #     ts.sort_values(tissues, ascending=False)
    #     .T.sort_values(ts.index.tolist(), ascending=False)
    #     .T.reset_index()
    #     .melt(id_vars=["Term"])
    #     .dropna()
    # )

    # plot selected top terms as barplots
    fs = 0.1 + (enrs["Term"].str.len().max() ** 0.1), 1 + 0.1 ** enrs["Term"].nunique()
    fig, ax = plt.subplots(1, 1, figsize=fs)
    sns.barplot(
        data=p,
        y="Term",
        x="value",
        hue="tissue",
        ax=ax,
        orient="h",
        hue_order=tissues,
        dodge=False,
        errorbar=None,
    )
    ax.axvline(0, linestyle="--", color="grey", linewidth=0.5)
    ax.set(xlabel=value, aspect="auto", title=target_var)
    fig.savefig(
        output_dir
        / f"specific_cases:{name}.{target_var}.express_regression.{fit_type}_fit.{label}.enrichr.barplot.top_{n}.svg",
        **figkws,
    )


# sb = mp.StackBar(p.T)
# lab = mp.Labels(p.index)
# cb = ma.ClusterBoard(p.T)
# cb.add_layer(sb)
# cb.add_bottom(lab)
# cb.render()
