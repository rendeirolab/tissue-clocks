# SPDX-FileCopyrightText: Copyright (c) 2026 Rendeiro Lab, CeMM - Research Center for Molecular Medicine, Austrian Academy of Sciences
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#
# Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE).

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from seaborn_extensions import clustermap
import statsmodels.api as sm

from src import config
from src.utils import get_pathology_data


def visualize_gtex_pathology_data():
    output_dir = config.results_dir / "cohort"
    output_dir.mkdir(exist_ok=True)
    # # prepare a matrix of pathology categories
    meta = pd.read_csv(config.gtex_csv, index_col=0)
    path_df = get_pathology_data()

    # Frequency of pathologies and change with age

    fig, axes = plt.subplots(4, 10, figsize=(10 * 3, 4 * 3), sharex=False, sharey=False)
    fig.suptitle("Frequency and change of pathology with age", y=0.92)
    _r = list()
    for ax, tissue in zip(axes.flatten(), meta["Tissue"].unique()):
        t = (
            path_df.join(a.obs[["Tissue", "Age"]])
            .query(f'Tissue == "{tissue}"')
            .drop("Tissue", axis=1)
        )
        x = t.astype("float32").drop("Age", axis=1)
        x = x.loc[:, x.var() > 0]
        x = x.loc[:, x.sum() > 2]
        y = t["Age"]
        fit = sm.OLS(y, x.assign(intercept=1)).fit()
        r = (
            fit.params.to_frame("coefficient")
            .join(fit.pvalues.rename("pvalues"))
            .join(x.mean().rename("frequency"))
            .drop("intercept")
        )
        _r.append(r.assign(tissue=tissue))

        ax.scatter(data=r, x="frequency", y="coefficient")
        ax.axhline(0, linestyle="--", color="grey")
        ax.set(title=tissue)
        for i in r.index:
            ax.annotate(i, (r.loc[i, "frequency"], r.loc[i, "coefficient"]))
        # ax.label_outer()
    # fig.tight_layout()
    for ax in axes[:, 0]:
        ax.set(ylabel=r"Change of pathology with age ($\beta$)")
    for ax in axes[-1, :]:
        ax.set(xlabel="Relative incidence")
    fig.savefig(
        output_dir / "pathologies.age_relationship.per_tissue.scatter.svg",
        **config.figkws,
    )

    res = pd.concat(_r).rename_axis("index")

    fig, ax = plt.subplots()
    ax.scatter(res["frequency"], res["coefficient"], alpha=0.5, s=10)
    ax.axhline(0, linestyle="--", color="grey")
    ax.axvline(0.01, linestyle="--", color="grey", linewidth=0.5)
    ax.axvline(0.1, linestyle="--", color="grey", linewidth=0.5)
    ax.axvline(1, linestyle="--", color="grey", linewidth=0.5)
    ax.set(
        xscale="log",
        xlabel="Relative incidence",
        ylabel=r"Change of pathology with age ($\beta$)",
    )
    fig.savefig(
        output_dir / "pathologies.age_relationship.all_tissues.scatter.svg",
        **config.figkws,
    )

    fig, ax = plt.subplots()
    ax.scatter(res["coefficient"], -np.log10(res["pvalues"]), alpha=0.5, s=10)
    ax.axvline(0, linestyle="--", color="grey")
    ax.set(
        ylabel="Significance",
        xlabel=r"Change of pathology with age ($\beta$)",
    )
    fig.savefig(
        output_dir / "pathologies.age_relationship.all_tissues.volcano.svg",
        **config.figkws,
    )

    freq = res.pivot_table(
        index="tissue", columns="index", values="frequency", fill_value=0
    )
    coef = res.pivot_table(
        index="tissue", columns="index", values="coefficient", fill_value=0
    )

    kwargs = dict(metric="correlation", dendrogram_ratio=0.1, figsize=(10, 7))
    grid = clustermap(
        freq, cbar_kws=dict(label="Relative incidence"), cmap="magma", **kwargs
    )
    grid.savefig(
        output_dir
        / "pathologies.age_relationship.frequency.all_tissues.clustermap.svg",
        **config.figkws,
    )
    grid = clustermap(
        coef,
        cbar_kws=dict(label="Change with Age\n(coefficient)"),
        cmap="coolwarm",
        center=0,
        **kwargs,
    )
    grid.savefig(
        output_dir
        / "pathologies.age_relationship.coefficient.all_tissues.clustermap.svg",
        **config.figkws,
    )
