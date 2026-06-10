from pathlib import Path

from tqdm import tqdm
import pandas as pd
import numpy as np
from statsmodels import api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn_extensions import clustermap

from src.utils import get_pathology_data


metadata_dir = Path("metadata")
data_dir = Path("data")
input_dir = Path("results") / "tissue_clocks_revision" / "clock"
results_dir = Path("results") / "tissue_clocks_revision" / "associations" / "pathology"
results_dir.mkdir(parents=True, exist_ok=True)
figkws = dict(dpi=300, bbox_inches="tight")


# lazyslide settings
model_names = [
    "ccnbg63",
    "uni",
    "uni2",
    "conch",
    "virchow",
    "virchow2",
    "hibou-b",
    "hibou-l",
    "midnight",
    "gigapath",
    "h0-mini",
    "phikon",
    "phikonv2",
    "ctranspath",
    "chief",
    "h-optimus-0",
    "h-optimus-1",
    "prism",
    "titan",
]
mpp = 0.5
tile_width = 224


# prepare pathology
pathology = get_pathology_data()

# load predictions
suffix = f".{mpp}mpp.{tile_width}px."
files = sorted(input_dir.glob(f"ridgecv{suffix}*.csv"))
model_names = sorted(
    set([p.stem.split(suffix)[-1].split(".")[-2] for p in files if " - " not in p.name])
)

_fits = list()
for model_name in tqdm(model_names):
    mfiles = [f for f in files if model_name + "." in f.stem and " - " not in f.name]
    organs = sorted(set([p.stem.split(suffix)[-1].split(".")[-1] for p in mfiles]))

    for metric in ["Age", "pred", "residual"]:
        for organ in organs:
            file = [f for f in mfiles if organ in f.name][0]
            preds = pd.read_csv(file, index_col=0)
            preds["residual"] = preds["Age"] - preds["pred"]
            preds = preds.groupby(level=0).mean()
            df = preds.join(pathology).dropna()
            df[metric] = (df[metric] - df[metric].mean()) / df[metric].std()

            pathologies = pathology.columns[df.loc[:, pathology.columns].any(axis=0)]

            for p in pathologies:
                _fits.append(
                    sm.GLM(
                        df[p],
                        sm.add_constant(df[metric]),
                        family=sm.families.Binomial(),
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
fits.to_csv(results_dir / "pathologies_all_models.stats.csv", index=False)


for organ in organs:
    p = (
        fits.query("metric == @metric & Organ == @organ")
        .pivot_table(
            columns=["pathology"],
            index=["model_name"],
            values=["Coef.", "-log10(P>|z|)"],
        )
        .astype(float)
        .fillna(0)  # one case
    )

    g = clustermap(
        p["Coef."],
        figsize=(6, 8),
        cmap="vlag",
        center=0,
        col_colors=p["Coef."].mean(0),
        robust=True,
    )
    g.fig.suptitle(f"{organ} - Coefficient")
    g.savefig(
        results_dir / f"pathologies_all_models.heatmap.coef.{organ}.svg", **figkws
    )


fig, axes = plt.subplots(
    2, 1, figsize=(8, 6), sharex=True, gridspec_kw={"hspace": 0.05}
)
sns.barplot(
    data=fits.query("metric == 'residual'"),
    x="model_name",
    y="Coef.",
    hue="Organ",
    ax=axes[0],
)
sns.barplot(
    data=fits.query("metric == 'residual'"),
    x="model_name",
    y="-log10(P>|z|)",
    hue="Organ",
    ax=axes[1],
)
for ax in axes:
    ax.set_xlabel("")
    ax.axhline(0, color="grey", linestyle="--")
axes[-1].set_xticklabels(ax.get_xticklabels(), rotation=90)
fig.tight_layout()
fig.suptitle("Pathology vs Age residuals")
fig.savefig(results_dir / "pathologies_all_models.stats.barplot.svg", **figkws)
