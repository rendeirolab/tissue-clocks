# SPDX-FileCopyrightText: Copyright (c) 2026 Rendeiro Lab, CeMM - Research Center for Molecular Medicine, Austrian Academy of Sciences
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#
# Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE).

from pathlib import Path

from tqdm import tqdm
import pandas as pd
import numpy as np
from statsmodels import api as sm
import matplotlib.pyplot as plt
import seaborn as sns

from seaborn_extensions import clustermap

from src.utils import get_engineered_info


metadata_dir = Path("metadata")
data_dir = Path("data")
input_dir = (
    Path("results") / "tissue_clocks_revision" / "gtex_imagenet_clocks" / "clock"
)
results_dir = (
    Path("results")
    / "tissue_clocks_revision"
    / "gtex_imagenet_clocks"
    / "associations"
    / "morbidity"
)
results_dir.mkdir(parents=True, exist_ok=True)
figkws = dict(dpi=300, bbox_inches="tight")


# lazyslide settings
model_names = [
    "alexnet",
    "vgg16",
    "resnet50",
    "convnext_tiny",
    "convnext_base",
    "convnext_large",
    "maxvit_t",
]
tile_width = 224


# prepare comorbidities
feats = get_engineered_info()
morb = feats.loc[:, feats.columns.str.startswith("morbidity")].astype(bool)
n = morb.sum(0).rename("n_individuals").sort_values()
morb = morb.loc[:, n >= 3].astype(float)
morb["n_comorbidities"] = morb.sum(1).rename("n_comorbidities")


# load predictions
suffix = f"{tile_width}px."
files = sorted(input_dir.glob(f"ridgecv_*{suffix}*.csv"))
model_names = sorted(set([p.stem.split(".")[0].replace("ridgecv_", "") for p in files]))
organs = sorted(set([p.stem.split(suffix)[-1].split(".")[-2] for p in files]))


_fits = list()
for model_name in tqdm(model_names):
    mfiles = [f for f in files if model_name + "." in f.stem]
    organs = sorted(set([p.stem.split(".")[-2] for p in mfiles]))

    for metric in ["Age", "pred", "residual"]:
        for organ in organs:
            file = [f for f in mfiles if organ in f.name][0]
            preds = pd.read_csv(file, index_col=0)
            preds["residual"] = preds["Age"] - preds["pred"]
            preds.index = preds.index.str.split("-").str[:2].map("-".join)
            preds = preds.groupby(level=0).mean()
            df = preds.join(morb).dropna()
            df[metric] = (df[metric] - df[metric].mean()) / df[metric].std()

            morbidities = morb.columns[df.loc[:, morb.columns].any(axis=0)]
            morbidities = morbidities[(df[morbidities].sum(axis=0) > 30)]

            for p in morbidities:
                _fits.append(
                    sm.GLM(
                        df[p],
                        sm.add_constant(df[metric]),
                        family=(
                            sm.families.Binomial()
                            if p != "n_comorbidities"
                            else sm.families.Poisson()
                        ),
                    )
                    .fit()
                    .summary2()
                    .tables[1]
                    .assign(
                        model_name=model_name,
                        Organ=organ,
                        metric=metric,
                        pathology=p,
                    )
                    .loc[metric]
                )

fits = pd.concat(_fits, axis=1).T.reset_index(drop=True)
fits["-log10(P>|z|)"] = -np.log10(fits["P>|z|"].astype(float))
fits.to_csv(results_dir / "morbidities_all_models.stats.csv", index=False)


metric = "residual"
order = (
    fits.query("metric == @metric & Organ == @organ")
    .groupby("model_name")["Coef."]
    .mean()
    .sort_values()
    .index
)
organs = [o for o in organs if o not in ["Bladder", "Cervix", "Fallopian Tube"]]
# organs = ['Brain', 'Colon', 'Lung', 'Skin']
fig1, axes1 = plt.subplots(4, 10, figsize=(10 * 2, 4 * 2), sharey=True, squeeze=False)
fig2, axes2 = plt.subplots(4, 10, figsize=(10 * 3, 4 * 5), sharey=True, squeeze=False)
for organ, ax1, ax2 in zip(organs, axes1.flatten(), axes2.flatten()):
    p = (
        fits.query("metric == @metric & Organ == @organ")
        .pivot_table(
            columns=["pathology"],
            index=["model_name"],
            values=["Coef.", "-log10(P>|z|)"],
        )
        .astype(float)
    )

    pp = p["Coef."].drop("n_comorbidities", axis=1).T.dropna().T
    g = clustermap(
        pp,
        figsize=(8, 4),
        cmap="vlag",
        center=0,
        col_colors=p["Coef."].mean(0),
        robust=True,
    )
    g.fig.suptitle(f"{organ} - Coefficient")
    g.savefig(
        results_dir / f"morbidities_all_models.heatmap.coef.{organ}.svg", **figkws
    )

    x = p.loc[:, ("Coef.", "n_comorbidities")].loc[order]
    y = p.loc[:, ("-log10(P>|z|)", "n_comorbidities")].reindex(x.index)

    ax1.scatter(x, y, s=5, alpha=0.85)
    ax1.set(title=organ, xlabel="Coefficient", ylabel="-log10(p)")

    sns.barplot(y=x.index, x=x, ax=ax2, orient="horiz")
    ax2.scatter(y=x.index, x=x, s=3 + y**np.e)
    for t in x.index:
        if y.loc[t] > -np.log10(0.05):
            ax2.text(x.loc[t], t, s="*")

    # y_legend_values = (
    #     pd.qcut(y, 10).value_counts().sort_index().index.map(lambda x: x.left).tolist()
    # )
    y_legend_values = [0, 2.5, 5]

    handles = [
        ax2.scatter(
            [],
            [],
            s=3 + y_val**np.e,
            label=f"y = {y_val}",
            color=sns.color_palette()[1],
        )
        for y_val in y_legend_values
    ]
    ax2.axvline(0, linestyle="--", color="grey")
    ax2.legend(
        handles=handles,
        title="-log10(P>|z|)",
        scatterpoints=1,
        loc=2,
        bbox_to_anchor=(1, 1),
    )
    sns.despine(fig2)
    ax2.set(title=organ, xlabel="Coefficient", xlim=(-0.185, 0.185))

fig1.savefig(
    results_dir / "morbidities_all_models.volcano.coef.all_organs.svg", **figkws
)
fig2.savefig(
    results_dir / "morbidities_all_models.barplot.coef.all_organs.svg", **figkws
)
