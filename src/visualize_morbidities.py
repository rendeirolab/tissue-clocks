# SPDX-FileCopyrightText: Copyright (c) 2026 Rendeiro Lab, CeMM - Research Center for Molecular Medicine, Austrian Academy of Sciences
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#
# Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE).

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import seaborn_extensions as snsx

from src.utils import get_restricted_info, get_engineered_info

results_dir = Path("results") / "gtex" / "cohort"
results_dir.mkdir(exist_ok=True, parents=True)
figkws = dict(dpi=300, bbox_inches="tight")

obs, obs_annot = get_restricted_info()
obsx = get_engineered_info(obs, obs_annot)

morbidity = obsx.loc[:, obsx.columns.str.startswith("morbidity:")]
bool_cols = morbidity.dtypes.apply(lambda c: c.categories.dtype == bool)

fig, ax = plt.subplots(figsize=(10, 10))
morbidity.loc[:, bool_cols].corr().pipe(
    sns.heatmap, ax=ax, xticklabels=True, yticklabels=True
)
fig.savefig(results_dir / "morbidity_similarity.heatmap.svg", **figkws)

c = morbidity.loc[:, bool_cols].dropna()
grid = snsx.clustermap(c.corr())
grid.savefig(results_dir / "morbidity_similarity.clustermap.svg", **figkws)

sel = c.to_numpy().sum(1) > 0
grid = snsx.clustermap(
    c.loc[sel].T.dropna().corr(), row_colors=obs["Cohort"], rasterized=True
)
grid.savefig(results_dir / "morbidity_individuals_similarity.clustermap.svg", **figkws)

# import networkx as nx
# import gravis as gv

# adj = c.loc[sel].T.dropna().corr() > 0.65
# g = nx.from_pandas_adjacency(adj)
# fig = gv.vis(g)
# fig.display()

# fig = gv.d3(g, zoom_factor=0.25)
# fig.export_svg(results_dir / 'morbidity_individuals_similarity.graph.svg', webdriver='firefox')


fig, ax = plt.subplots(figsize=(4, 4))
count = (
    morbidity.loc[:, bool_cols]
    .apply(lambda x: x.dropna().cat.codes.sum())
    .sort_values()
)
sns.histplot(count, ax=ax)
fig.savefig(results_dir / "morbidity_count.histplot.svg", **figkws)

fig, ax = plt.subplots(figsize=(6, 10))
ax.scatter(count, count.index)
ax.set(xlabel="Morbidity count")
fig.savefig(results_dir / "morbidity_count.scatter.svg", **figkws)

fig, ax = plt.subplots(figsize=(6, 10))
ax.scatter(count, count.index)
ax.set(xscale="log", xlabel="Morbidity count")
fig.savefig(results_dir / "morbidity_count.scatter.log.svg", **figkws)


morbs = pd.Series(
    np.nansum(morbidity.loc[:, bool_cols].to_numpy(), axis=1),
    index=morbidity.index,
    dtype=int,
)
fig, ax = plt.subplots(figsize=(4, 4))
ax.scatter(obs["Age"], morbs, alpha=0.5)
sns.regplot(x=obs["Age"], y=morbs, scatter=False, color="black", ax=ax)
ax.set(xlabel="Age", ylabel="Morbidities per individual")
fig.savefig(results_dir / "morbidity_count.vs_age.regplot.svg", **figkws)
