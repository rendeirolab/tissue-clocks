# SPDX-FileCopyrightText: Copyright (c) 2026 Rendeiro Lab, CeMM - Research Center for Molecular Medicine, Austrian Academy of Sciences
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#
# Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE).

"""
High-level operations that can be reused across datasets.
"""

import typing as tp
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn_extensions import clustermap
from anndata import AnnData
from pymde import preserve_neighbors as PYMDE
from pymde import penalties
from sklearn.manifold import Isomap, MDS
from sklearn.decomposition import FastICA

from src import config
from src.utils import rasterize_scanpy

figkws = config.figkws


def process(
    a: AnnData,
    short: bool = False,
    confounder: tp.Optional[str] = "Hardy Scale",
    covariates: list[str] = ["Sex", "Age Bracket", "Tissue"],
) -> AnnData:
    sc.pp.scale(a)

    if confounder is not None:
        covs = [c for c in covariates if c in a.obs.columns and a.obs[c].nunique() > 1]
        sc.pp.combat(a, confounder, covariates=covs)
        sc.pp.scale(a)
    sc.pp.pca(a)
    sc.pp.neighbors(a)
    sc.tl.umap(a)
    if short:
        return a
    sc.tl.diffmap(a)
    # remove first dimension of diffmap
    a.obsm["X_diffmap"] = a.obsm["X_diffmap"][:, 1:]
    sc.tl.draw_graph(a)
    isomap(a, n_components=10)
    ica(a, n_components=min(50, a.shape[0] - 1))
    # mds(a, n_components=10)
    pymde(a, n_components=10)
    pymde(a, n_components=10, config="alternate")
    return a


def plot_latent(
    a,
    output_prefix: Path,
    voi: list[str],
    label: str = ".",
    embeddings: tp.Optional[list[str]] = None,
    plot_embeddings: bool = True,
    plot_embeddings_3d: bool = True,
    plot_correlations: bool = True,
):
    if not label.startswith("."):
        label = "." + label
    if not label.endswith("."):
        label += "."

    if "." not in output_prefix.name:
        output_prefix = output_prefix.parent / (output_prefix.name + ".SUFFIX")

    if embeddings is None:
        embeddings = [q.replace("X_", "") for q in a.obsm.keys()]
    if not plot_embeddings:
        embeddings = []

    a = a[a.obs.sample(frac=1).index].copy()
    for embedding in embeddings:
        kws = dict(ncols=1)
        if a.obsm["X_" + embedding].shape[1] >= 4:
            kws = dict(components=["1,2", "2,3", "3,4", "4,5"], ncols=4)
        fig = sc.pl.embedding(a, basis=embedding, color=voi, **kws)
        rasterize_scanpy(fig)

        # remove redundant legends
        # account for continuous vars that that get a colormap legend with their own axes
        axes = np.asarray([ax for ax in fig.axes if ax.get_title() != ""]).reshape(
            len(voi), kws["ncols"]
        )
        for ax in axes[:, :-1].flat:
            leg = ax.get_legend()
            if leg is not None:
                leg.set_visible(False)
        for ax in [ax for ax in fig.axes if ax.get_title() == ""][:-1]:
            ax.remove()
        fig.savefig(
            output_prefix.with_suffix(f".{label}dimres.{embedding}.svg"), **figkws
        )

        if plot_embeddings_3d and (a.obsm["X_" + embedding].shape[1] >= 3):
            cmpnts = (
                ["1,2,3", "2,3,4", "3,4,5", "4,5,6"]
                if a.obsm["X_" + embedding].shape[1] >= 6
                else ["1,2,3", "2,3,4"]
            )
            kws = dict(
                components=cmpnts,
                projection="3d",
                ncols=len(cmpnts),
                size=120000 / a.shape[0],
            )
            fig = sc.pl.embedding(a, basis=embedding, color=voi, **kws)
            rasterize_scanpy(fig)

            # remove redundant legends
            # account for continuous vars that that get a colormap legend with their own axes
            axes = np.asarray([ax for ax in fig.axes if ax.get_title() != ""]).reshape(
                len(voi), kws["ncols"]
            )
            for ax in axes[:, :-1].flat:
                leg = ax.get_legend()
                if leg is not None:
                    leg.set_visible(False)
            for ax in [ax for ax in fig.axes if ax.get_title() == ""][:-1]:
                ax.remove()
            fig.savefig(
                output_prefix.with_suffix(f".{label}dimres.{embedding}-3d.svg"),
                **figkws,
            )

    _corrs = list()
    for embedding in embeddings:
        if a.obsm["X_" + embedding].shape[1] <= 2:
            continue
        lat = pd.DataFrame(a.obsm["X_" + embedding], index=a.obs.index)
        lat.columns += 1
        for col in voi:
            if a.obs[col].dtype in ["int", "float"]:
                lat[col] = a.obs[col]
            elif a.obs[col].nunique() == 2:
                lat[col] = (a.obs[col] == a.obs[col].unique()[1]).astype(int)
            else:
                d = pd.get_dummies(a.obs[col])
                sel = d.columns[~d.columns.isin(lat.columns)]
                lat = lat.join(d[sel])
        s = lat.columns.str.isdigit().fillna(True).values.astype(bool)
        corr = lat.corr().loc[s, ~s]
        n = corr.isnull()
        corr = corr.loc[~n.all(1), ~n.all()]

        corr = corr.iloc[:50, :]
        _corrs.append(
            corr.assign(embedding=embedding).rename_axis(index="dim").reset_index()
        )

        grid = clustermap(
            corr,
            config="abs",
            dendrogram_ratio=0.05,
            vmin=-1,
            vmax=1,
            cmap="RdBu_r",
            row_cluster=False,
        )
        grid.ax_heatmap.get_children()[0].set(rasterized=True)
        grid.fig.savefig(
            output_prefix.with_suffix(
                f".{label}{embedding}.feature_correlation.clustermap.svg"
            ),
            **figkws,
        )

    corrs = pd.concat(_corrs)
    corrs.to_csv(
        output_prefix.with_suffix(f".{label}embedding_variable_correlation.csv"),
        index=False,
    )


def discover_groups(
    a: AnnData,
    output_prefix: Path,
    label: str = ".",
    resolutions: list[float] | None = None,
) -> list[str]:
    if not label.startswith("."):
        label = "." + label
    if resolutions is None:
        resolutions = [0.2, 0.3, 0.4, 0.5]
    for res in resolutions:
        sc.tl.leiden(a, resolution=res, key_added=f"leiden_{res}")

    # characterize leiden clusters
    kwargs = dict(dendrogram_ratio=0.05, row_cluster=False)
    for res in resolutions:
        m = a.to_df().groupby(a.obs[f"leiden_{res}"]).mean()
        m = m.loc[:, m.var() > 0]
        for config in ["abs", "z"]:
            g = clustermap(m, config=config, **kwargs)
            g.fig.savefig(
                output_prefix + f"{label}leiden_{res}.clustermap.{config}.svg",
                **figkws,
            )
    return [f"leiden_{res}" for res in resolutions]


def pymde(
    anndata: AnnData, config: str = "default", n_components: int = 2, **kwargs
) -> AnnData:
    if config == "default":
        anndata.obsm["X_pymde"] = (
            PYMDE(anndata.X, embedding_dim=n_components, **kwargs).embed().numpy()
        )
    elif config == "alternate":
        anndata.obsm["X_pymde_alt"] = (
            PYMDE(
                anndata.X,
                embedding_dim=n_components,
                attractive_penalty=penalties.Quadratic,
                repulsive_penalty=None,
                **kwargs,
            )
            .embed()
            .numpy()
        )
    return anndata


def isomap(anndata: AnnData, n_components: int = 2, **kwargs) -> AnnData:
    model = Isomap(n_components=n_components, **kwargs)
    anndata.obsm["X_isomap"] = model.fit_transform(anndata.X)
    return anndata


def ica(anndata: AnnData, n_components: int = 2, **kwargs) -> AnnData:
    model = FastICA(n_components=n_components, whiten="unit-variance", **kwargs)
    anndata.obsm["X_ica"] = model.fit_transform(anndata.X)
    return anndata


def mds(anndata: AnnData, n_components: int = 2, **kwargs) -> AnnData:
    model = MDS(n_components=n_components, **kwargs)
    anndata.obsm["X_mds"] = model.fit_transform(anndata.X)
    return anndata
