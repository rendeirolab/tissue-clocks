# SPDX-FileCopyrightText: Copyright (c) 2026 Rendeiro Lab, CeMM - Research Center for Molecular Medicine, Austrian Academy of Sciences
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#
# Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE).

from pathlib import Path

from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels import api as sm

from src.utils import get_telomere_lengths


metadata_dir = Path("metadata")
data_dir = Path("data")
input_dir = Path("results") / "tissue_clocks_revision" / "clock"
results_dir = Path("results") / "tissue_clocks_revision" / "associations" / "telomeres"
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


# prepare telomere lengths
# target_var = "Age"
# meta = pd.read_csv(data_dir / "gtex" / "GTEx Portal.csv", index_col=0)
# rest, _ = get_restricted_info()
t = get_telomere_lengths()
t["Organ"] = t["TissueSiteDetail"].str.split(" - ").str[0]
tl = t.groupby(["Organ", "CollaboratorParticipantID"])["TQImean"].mean().sort_index()
tln = (
    tl.groupby(level="Organ")
    .apply(lambda x: (x - x.mean()) / x.std())
    .reset_index(level=1, drop=True)
    .dropna()
)

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

    fig, axes = plt.subplots(
        len(organs), 3, figsize=(2.2 * 3, 2 * len(organs)), sharex=False, sharey=True
    )
    for metric, axs in zip(["Age", "pred", "residual"], axes.T):
        for organ, ax in zip(organs, axs):
            file = [f for f in mfiles if organ in f.name][0]
            preds = pd.read_csv(file, index_col=0)
            preds["residual"] = preds["Age"] - preds["pred"]
            preds.index = preds.index.str.split("-").str[:2].map("-".join)
            preds = preds.groupby(level=0).mean()
            df = preds.join(tln.loc[organ]).dropna()
            x = df[metric].copy()
            x = (x - x.mean()) / x.std()

            _fits.append(
                sm.OLS(df["TQImean"], sm.add_constant(x))
                .fit()
                .summary2()
                .tables[1]
                .assign(model_name=model_name, Organ=organ, metric=metric)
                .loc[metric]
            )

            if True:
                df["q"] = pd.qcut(df[metric], 5)
                v = df["q"].value_counts().sort_index()
                mapping = {v: k for k, v in enumerate(v.index, 1)}
                df["q"] = df["q"].replace(mapping)
            else:
                bins = [-25, -5, -2, 2, 5, 25]
                df["q"] = pd.cut(df[metric], bins)
            sns.barplot(y="q", x="TQImean", data=df, ax=ax, orient="horiz")
            if True:
                for i in range(1, 6):
                    ax.text(ax.get_xticks()[-1], i - 1, df.query("q == @i").shape[0])
            else:
                for i, b in enumerate(df["q"].unique()):
                    ax.text(
                        ax.get_xticks()[-1], i, f"n = {df.query('q == @b').shape[0]}"
                    )
            ax.axvline(0, color="grey", linestyle="--")
            ax.set(
                title=f"{organ} (n = {df.shape[0]}), {metric}",
                xlabel="Telomere length (Z-score)",
                ylabel=metric,
            )
            ax.yaxis.set_inverted(False)
    fig.tight_layout()
    fig.savefig(results_dir / f"{model_name}_telomere_in_bins.barplot.svg", **figkws)

fits = pd.concat(_fits, axis=1).T.reset_index(drop=True)
fits["-log10(P>|t|)"] = -np.log10(fits["P>|t|"].astype(float))
fits.to_csv(results_dir / "telomere_length_all_models.stats.csv", index=False)

fits = pd.read_csv(results_dir / "telomere_length_all_models.stats.csv")

fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True)
p = fits.pivot_table(
    index=["model_name"],
    columns=["Organ", "metric"],
    values=["Coef.", "-log10(P>|t|)"],
).astype(float)
sns.heatmap(
    p["-log10(P>|t|)"],
    cmap="magma",
    cbar_kws={"label": "-log10(P-value)"},
    ax=axes[0],
    vmax=3,
)
sns.heatmap(
    p["Coef."],
    center=0,
    cmap="vlag",
    cbar_kws={"label": "Coefficient"},
    ax=axes[1],
    vmin=-0.2,
    vmax=0.2,
)
fig.tight_layout()
fig.savefig(results_dir / "telomere_length_all_models.stats.heatmap.svg", **figkws)

fig, axes = plt.subplots(
    2, 1, figsize=(5, 6), sharex=True, gridspec_kw={"hspace": 0.05}
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
    y="-log10(P>|t|)",
    hue="Organ",
    ax=axes[1],
)
for ax in axes:
    ax.set_xlabel("")
    ax.axhline(0, color="grey", linestyle="--")
axes[-1].set_xticklabels(ax.get_xticklabels(), rotation=90)
fig.tight_layout()
fig.suptitle("Telomere length vs Age residuals")
fig.savefig(results_dir / "telomere_length_all_models.stats.barplot.svg", **figkws)
