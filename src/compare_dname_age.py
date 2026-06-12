# SPDX-FileCopyrightText: Copyright (c) 2026 Rendeiro Lab, CeMM - Research Center for Molecular Medicine, Austrian Academy of Sciences
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#
# Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE).

"""
Compare predictions of age of GTEx samples using DNA methylation data with histology.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn_extensions import clustermap
import pingouin as pg

from src.utils import get_restricted_info

# GTEx
data_dir = Path("data") / "gtex"
results_dir = (
    Path("results") / "gtex" / "dna_methylation" / "comparison_histology_dname"
)
results_dir.mkdir(exist_ok=True, parents=True)
figkws = dict(dpi=300, bbox_inches="tight")
meta = pd.read_csv(data_dir / "GTEx Portal.csv", index_col=0)
rest, _ = get_restricted_info()

# clock_table = pd.read_html('metadata/pyaging_clock_table.2026-01-27.html', index_col=0)[0]

for var in ["Age", "prediction", "prediction_adj", "residuals", "residuals_adj"]:
    # var = 'residuals_adj'
    # Load DNAme clocks
    dname = pd.read_csv(
        results_dir.parent / "pyage" / "gtex_age_prediction.csv", index_col=0
    )
    dname.index = dname.index.str.extract(r"(GTEX-\w+-\d{4}).*")[0].values
    tissue = dname[["Tissue"]].copy()
    dname = (
        dname.drop(["Age", "Sex", "Cohort", "Tissue"], axis=1).groupby(level=0).mean()
    )
    clock_names = dname.columns.tolist()

    uns = pd.read_csv(
        results_dir.parent / "pyage" / "gtex_age_prediction.uns.csv", index_col=0
    )
    uns.index = uns.index.str.replace("_metadata", "")

    if var.startswith("residual"):
        d = dname.assign(
            **{"Subject ID": dname.index.str.extract(r"(GTEX-\w+)-\d{4}.*")[0].values}
        )
        d = (
            d.reset_index()
            .merge(rest["Age"], left_on="Subject ID", right_index=True, how="left")
            .set_index("index")
            .drop(["Subject ID"], axis=1)
        )
        dname = (d.T - d["Age"]).T.drop(["Age"], axis=1)

        fig, ax = plt.subplots(figsize=(3, 5))
        mae = dname.abs().mean().rename("MAE").sort_values(ascending=True)
        sns.barplot(x=mae, y=mae.index, orient="h", ax=ax)
        fig.savefig(results_dir / "dname_clocks_mae.barplot.svg", **figkws)

    # Load histology
    gaps_dir = Path("results") / "gtex" / "fine_tuned" / "_pre_2024-01-19_age_X_frac1.0"
    gaps = pd.read_parquet(
        gaps_dir / "tissue-specific_clocks.Ridge.GroupKFold.predictions_residuals.pq"
    )
    if var.startswith("residual"):
        gaps.loc[gaps[var].abs() > 50, var] = np.nan
    gaps["Individual"] = gaps.index.str.extract(r"(GTEX-\w+)-\d{4}.*")[0].values
    for op in ["mean", "max", "std"]:
        m = (
            gaps.groupby("Individual")[var]
            .apply(op)
            .to_frame(var)
            .assign(Tissue=op)
            .reset_index()
        )
        m.index = m["Individual"].rename("Tissue Sample ID") + "-0000"
        gaps = pd.concat([gaps, m], axis=0)
    gaps = gaps.drop(["Individual"], axis=1)

    # Join
    j = gaps[[var]].join(dname).dropna()
    c = j.corr()
    g = clustermap(c, center=0, cmap="RdBu_r", figsize=(5, 5), dendrogram_ratio=0.1)
    g.fig.savefig(results_dir / f"histology_dname_{var}_correlation.svg", **figkws)

    c2 = (
        c.loc[var]
        .drop(var)
        .sort_values(ascending=False)
        .rename("Correlation to histology")
    )
    fig, ax = plt.subplots(figsize=(3, 5))
    sns.barplot(x=c2, y=c2.index, orient="h", ax=ax)
    fig.savefig(
        results_dir / f"histology_dname_{var}_correlation.all_tissues.barplot.svg",
        **figkws,
    )

    tissues = tissue.value_counts().index
    fig, axes = plt.subplots(
        2, len(clock_names) // 2, figsize=(3 * len(clock_names) // 2, 3 * 2)
    )
    for ax, clock in zip(axes.flatten(), clock_names):
        ax.scatter(j[var], j[clock], s=5, alpha=0.5, rasterized=True)
        r = pg.corr(j[var], j[clock]).squeeze()
        ax.axhline(0, color="grey", linestyle="--", linewidth=1)
        ax.axvline(0, color="grey", linestyle="--", linewidth=1)
        vmin = j[[var, clock]].min().min()
        vmax = j[[var, clock]].max().max()
        ax.plot([vmin, vmax], [vmin, vmax], color="grey", linestyle="--")
        ax.set(
            title=f"{clock} - r = {r['r']:.2f}",
            xlabel=f"Histology {var}",
            ylabel=f"DNA methylation {'residual' if var.startswith('residual') else 'age'}",
        )
    fig.tight_layout()
    fig.savefig(
        results_dir / f"histology_dname_{var}_correlation_scatter.all_tissues.svg",
        **figkws,
    )

    h = (
        j.join(tissue)
        .groupby("Tissue")
        .corr()[var]
        .sort_values()
        .where(lambda x: x < 1)
        .dropna()
    )
    fig, ax = plt.subplots(figsize=(3, 3))
    sns.histplot(h)
    ax.axvline(0, linestyle="--", color="grey")
    ax.set(xlabel="Pearson correlation", ylabel="Number of tissue-clock relationships")
    fig.tight_layout()
    fig.savefig(
        results_dir / f"histology_dname_{var}_correlation_histplot.svg", **figkws
    )

    tissues = sorted(tissue["Tissue"].value_counts().index)
    n = len(clock_names)
    fig, axes = plt.subplots(n, len(tissues), figsize=(3 * len(tissues), 3 * n))
    _res = list()
    for axs, clock in zip(axes, clock_names):
        axs[0].set(ylabel=clock)
        if clock == clock_names[0]:
            for ax, t in zip(axs, tissues):
                ax.set(title=t)
        for ax, t in zip(axs, tissues):
            j2 = j.join(tissue).query("Tissue == @t")
            if j2.empty:
                continue
            r = pg.corr(j2[var], j2[clock]).assign(clock=clock, tissue=t).squeeze()
            _res.append(r)
            ax.scatter(j2[var], j2[clock], s=5, alpha=0.5)
            ax.axhline(0, color="grey", linestyle="--", linewidth=0.25)
            ax.axvline(0, color="grey", linestyle="--", linewidth=0.25)
            # vmin = j2[[var, clock]].min().min()
            # vmax = j2[[var, clock]].max().max()
            # ax.plot([vmin, vmax], [vmin, vmax], color="grey", linestyle="--")
            ax.set(
                # title=f"{clock}, {t} - r = {r['r']:.2f}",
                xlabel=f"Histology {var}",
                ylabel=f"DNA methylation {'residual' if var.startswith('residual') else 'age'}",
            )
    fig.savefig(
        results_dir / f"histology_dname_{var}_correlation_scatter.per_tissue.svg",
        **figkws,
    )

    pd.DataFrame(_res).groupby("tissue")["r"].mean().sort_values()
