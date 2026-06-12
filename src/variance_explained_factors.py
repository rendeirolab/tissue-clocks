# SPDX-FileCopyrightText: Copyright (c) 2026 Rendeiro Lab, CeMM - Research Center for Molecular Medicine, Austrian Academy of Sciences
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#
# Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE).

"""
Analysis of factors contributing to variance in histological space
"""

from tqdm import tqdm
import pandas as pd
import scanpy as sc
import statsmodels.api as sm
import parmap
import matplotlib.pyplot as plt
import seaborn as sns


from src import config
from src.utils import get_individual_factors, clustermap_marsilea


output_dir = config.results_dir / "variance_explained_factors"
output_dir.mkdir(exist_ok=True)


def fit_one_out(factor: str | None):
    labels = ["Tissue", "PC", "R2", "R2_adj", "F_pvalue", "AIC", "BIC", "Factor"]
    y3 = y2.drop(factor, axis=1) if factor is not None else y2
    y3 = y3.loc[:, y3.var(axis=0) > 0]
    fit2 = sm.OLS(x[pc], sm.add_constant(y3)).fit()
    return pd.Series(
        [
            tissue,
            pc,
            fit2.rsquared,
            fit2.rsquared_adj,
            fit2.f_pvalue,
            fit2.aic,
            fit2.bic,
            factor if factor is not None else "All",
        ],
        index=labels,
    )


if __name__ != "__main__" or "get_ipython" in locals():
    raise RuntimeError("In intereactive IPython session. Stopping.")

a = sc.read_h5ad(config.results_dir / "fine_tuned" / "anndata.h5ad")
y = get_individual_factors()

corr = y.corr()
corr.index = corr.index.str.split("(").str[0].str.strip()
corr.columns = corr.columns.str.split("(").str[0].str.strip()
g = clustermap_marsilea(corr, xticklabels=False, center=0, cmap="coolwarm", square=True)
g.fig.savefig(output_dir / "correlation_matrix.pdf", **config.figkws)

# Use PCA space to calculate variance explained by different factors
# Fit linear models to each PC
_res = list()
tissues = sorted(a.obs["Tissue"].unique())
for tissue in tqdm(tissues, position=0):
    b = a[a.obs.query("Tissue == @tissue").index].copy()
    sc.pp.pca(b)
    var = pd.Series(b.uns["pca"]["variance"])
    var.index = ["PC" + str(i + 1).zfill(2) for i in range(len(var))]
    var_ratio = var / var.sum()
    x = pd.DataFrame(b.obsm["X_pca"], index=b.obs.index)
    x.columns = var.index
    x = x.loc[:, var_ratio > 0.01]
    x.index = x.index.str.extract(r"(GTEX-\w+)-.*", expand=False)
    x = x.groupby(level=0).mean()
    y2 = y.reindex(x.index).dropna()
    x = x.reindex(y2.index)
    for pc in tqdm(x.columns, leave=False, desc=tissue, position=1):
        # Fit full model with all variables
        full = fit_one_out(None)
        full["PC_variance_ratio"] = var_ratio.loc[pc]
        # And fit a model missing each of the other variables
        loo = pd.concat(parmap.map(fit_one_out, y2.columns, pm_processes=4), axis=1).T
        # Now calculate the difference to the full model for each variable
        loo["R2-diff"] = loo["R2"] - full["R2"]
        loo["R2_adj-diff"] = loo["R2_adj"] - full["R2_adj"]
        # Scale the difference in each PC by the variance ratio associated with it
        loo["R2-diff-scaled"] = loo["R2-diff"] * full["PC_variance_ratio"]
        loo["R2_adj-diff-scaled"] = loo["R2_adj-diff"] * full["PC_variance_ratio"]
        _res.append(pd.concat([full.to_frame().T, loo], axis=0))
    res = pd.concat(_res).drop_duplicates(keep="last")
    res.to_csv(output_dir / "variance_explained_factors.csv", index=False)

res = pd.read_csv(output_dir / "variance_explained_factors.csv").query(
    "Factor != 'All'"
)

fits = pd.read_csv(output_dir / "variance_explained_factors.csv").query(
    "Factor == 'All'"
)
fits["R2-scaled"] = fits["R2"] * fits["PC_variance_ratio"]
fits["R2_adj-scaled"] = fits["R2_adj"] * fits["PC_variance_ratio"]
r2_scaled = fits.groupby(["Tissue"])["R2-scaled"].sum().sort_values(ascending=False)
r2_adj_scaled = (
    fits.groupby(["Tissue"])["R2_adj-scaled"].sum().sort_values(ascending=False)
)

tissues = r2_scaled.where((r2_scaled > 0) & (r2_scaled <= 0.5)).dropna().index

fig, ax = plt.subplots(figsize=(2, 5))
sns.barplot(x=r2_scaled.loc[tissues] * 100, y=tissues, orient="h", ax=ax)
ax.set(xlabel="% histological variance explained by metadata", ylabel="Tissues")
ax.axvline(0, linestyle="--", color="grey")
fig.savefig(
    output_dir / "variance_explained_factors.full_model.svg",
    **config.figkws,
)

for metric in ["R2-diff-scaled", "R2_adj-diff-scaled"]:
    output_prefix = (
        output_dir
        / f"variance_explained_factors.difference_to_full_model.{metric}.SUFFIX"
    )

    # The final answer is the sum (across PCs) of the scaled difference to the full model
    res2 = res.groupby(["Tissue", "Factor"])[metric].sum().unstack().T * 100
    res2 = res2.loc[res2.var(1) > 1e-20, tissues]

    # # simplify factor labels for plotting
    res2.index = res2.index.str.split(r"\(").str[0].str.strip()
    o = res2.mean(1).sort_values(ascending=True).index

    # # Across all tissues
    fig, ax = plt.subplots(figsize=(2, 16))
    sns.barplot(
        data=res2.stack().rename(metric).reset_index(),
        x=metric,
        y="Factor",
        orient="h",
        ax=ax,
        order=o,
    )
    ax.set(xlabel="% difference to full model", ylabel=f"Factors (n = {len(o)})")
    ax.axvline(0, linestyle="--", color="grey")
    fig.savefig(
        output_prefix.with_suffix(".all_error_bars.svg"),
        **config.figkws,
    )
    fig.savefig(
        output_prefix.with_suffix(".all_error_bars.pdf"),
        **config.figkws,
    )

    of = a.obs["Tissue"].value_counts().rename("Samples").to_frame()
    vf = (y >= 0).mean().sort_values().rename("Fraction not NA").to_frame()
    vf.index = vf.index.str.split(r"\(").str[0].str.strip()
    kwargs = dict(square=True, col_colors=of, row_colors=vf, vmax=0.1, robust=True)

    # # # Per tissue
    g = clustermap_marsilea(res2, **kwargs)
    g.fig.savefig(
        output_prefix.with_suffix(".per_tissue.value.pdf"),
        **config.figkws,
    )

    r = res2.rank(axis=0)
    g = clustermap_marsilea(
        r.loc[o], square=True, col_colors=of, row_colors=vf, row_cluster=False
    )
    g.fig.savefig(
        output_prefix.with_suffix(".per_tissue.rank.sorted.pdf"),
        **config.figkws,
    )

    # # Collapse similar factors (levels of same factor)
    res3 = res2.groupby(res2.index.str.split("_").str[0]).mean()
    res3 = res3.loc[res3.var(1) > 1e-20, :]
    o = res3.mean(1).sort_values(ascending=True).index
    kwargs = dict(square=True, col_colors=of, vmax=0.1, robust=True)

    # # # Across all tissues
    fig, ax = plt.subplots(figsize=(2, 14))
    sns.barplot(
        data=res3.stack().rename(metric).reset_index(),
        x=metric,
        y="Factor",
        orient="h",
        ax=ax,
        order=o,
    )
    ax.set(xlabel="% difference to full model", ylabel=f"Factors (n = {len(o)})")
    ax.axvline(0, linestyle="--", color="grey")
    fig.savefig(
        output_prefix.with_suffix(".all_error_bars.collapsed.svg"),
        **config.figkws,
    )
    fig.savefig(
        output_prefix.with_suffix(".all_error_bars.collapsed.pdf"),
        **config.figkws,
    )

    # # # Per tissue
    g = clustermap_marsilea(res3.loc[o], **kwargs, row_cluster=False)
    g.fig.savefig(
        output_prefix.with_suffix(".per_tissue.collapsed.sorted.pdf"),
        **config.figkws,
    )

    # More detail
    fig, axes = plt.subplots(1, 2, figsize=(4 * 2, 4))
    axes[0].scatter(res3.loc["demographics:Age"], res3.loc["demographics:Sex"])
    for col in res3.columns:
        axes[0].text(
            res3.loc["demographics:Age", col], res3.loc["demographics:Sex", col], col
        )
    axes[0].set(xlabel="Age", ylabel="Sex")
    axes[1].scatter(res3.loc["demographics:Age"], res3.loc["death:Ischemic Time"])
    for col in res3.columns:
        axes[1].text(
            res3.loc["demographics:Age", col], res3.loc["death:Ischemic Time", col], col
        )
    axes[1].set(xlabel="Age", ylabel="Ischemic Time")
    for ax in axes:
        ax.axvline(0, linestyle="--", color="grey")
        ax.axhline(0, linestyle="--", color="grey")
    fig.savefig(
        output_prefix.with_suffix(".scatter_detail.pdf"),
    )
