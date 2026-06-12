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
from src.utils import get_restricted_info


metadata_dir = Path("metadata")
data_dir = Path("data") / "gtex"
input_dir = Path("results") / "tissue_clocks_revision" / "clock"
results_dir = (
    Path("results") / "tissue_clocks_revision" / "dname_comparison" / "telomeres"
)
results_dir.mkdir(parents=True, exist_ok=True)
figkws = dict(dpi=300, bbox_inches="tight")

meta = pd.read_csv(data_dir / "GTEx Portal.csv", index_col=0)
rest, _ = get_restricted_info()


def get_histology_data(
    convert_to_residuals: bool = False, regress_out_age: bool = True
) -> pd.DataFrame:
    mpp = 0.5
    tile_width = 224
    input_dir = Path("results") / "tissue_clocks_revision" / "clock"

    # load predictions
    suffix = f".{mpp}mpp.{tile_width}px."
    files = sorted(input_dir.glob(f"ridgecv{suffix}*.csv"))
    _res = list()
    for file in files:
        model_name = file.stem.split(".")[-2]
        preds = pd.read_csv(file, index_col=0)["pred"]
        _res.append(preds.rename(model_name))
    histo = pd.concat(_res, axis=1)
    histo = histo.T.groupby(level=0).mean().T.dropna()
    if regress_out_age:
        d = histo.assign(
            **{"Subject ID": histo.index.str.extract(r"(GTEX-\w+)-\d{4}.*")[0].values}
        )
        d = (
            d.reset_index()
            .merge(rest["Age"], left_on="Subject ID", right_index=True, how="left")
            .set_index("index")
            .drop(["Subject ID"], axis=1)
        )
        histo = pd.DataFrame(
            {
                col: sm.OLS(histo[col], sm.add_constant(d["Age"])).fit().resid
                for col in histo.columns
            },
            index=histo.index,
        )
    # Convert to residuals
    if convert_to_residuals:
        d = histo.assign(
            **{"Subject ID": histo.index.str.extract(r"(GTEX-\w+)-\d{4}.*")[0].values}
        )
        d = (
            d.reset_index()
            .merge(rest["Age"], left_on="Subject ID", right_index=True, how="left")
            .set_index("index")
            .drop(["Subject ID"], axis=1)
        )
        histo = (d.T - d["Age"]).T.drop(["Age"], axis=1)
    return histo


def get_dname_data(
    convert_to_residuals: bool = False, regress_out_age: bool = True
) -> pd.DataFrame:
    dname = pd.read_csv(
        Path("results")
        / "gtex"
        / "dna_methylation"
        / "pyage"
        / "gtex_age_prediction.csv",
        index_col=0,
    )
    dname.index = dname.index.str.extract(r"(GTEX-\w+-\d{4}).*")[0].values
    dname = (
        dname.drop(["Age", "Sex", "Cohort", "Tissue"], axis=1).groupby(level=0).mean()
    )
    # Regress out age
    if regress_out_age:
        d = dname.assign(
            **{"Subject ID": dname.index.str.extract(r"(GTEX-\w+)-\d{4}.*")[0].values}
        )
        d = (
            d.reset_index()
            .merge(rest["Age"], left_on="Subject ID", right_index=True, how="left")
            .set_index("index")
            .drop(["Subject ID"], axis=1)
        )
        dname = pd.DataFrame(
            {
                col: sm.OLS(dname[col], sm.add_constant(d["Age"])).fit().resid
                for col in dname.columns
            },
            index=dname.index,
        )
    # Convert to residuals
    if convert_to_residuals:
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
    return dname


# prepare telomere lengths
t = get_telomere_lengths()
t["Organ"] = t["TissueSiteDetail"].str.split(" - ").str[0]
# tl = t.groupby(["Organ", "CollaboratorParticipantID"])["TQImean"].mean().sort_index()
# tln = (
#     t.groupby(level="Organ")
#     .apply(lambda x: (x - x.mean()) / x.std())
#     .reset_index(level=1, drop=True)
#     .dropna()
# )
t = (
    t.groupby("Organ")["TQImean"]
    .apply(lambda x: (x - x.mean()) / x.std())
    .rename("value")
    .reset_index()
    .set_index("slide_id")
)
organs = t["Organ"].unique().tolist()
# organs = ["Brain", "Colon", "Lung", "Skin"]
organs = ["Colon", "Lung"]


runs = [
    dict(
        modality="histology",
        metric="residuals",
        df=get_histology_data(convert_to_residuals=True, regress_out_age=False),
    ),
    dict(
        modality="histology",
        metric="residuals_adj",
        df=get_histology_data(convert_to_residuals=False, regress_out_age=True),
    ),
    dict(
        modality="dname",
        metric="residuals",
        df=get_dname_data(convert_to_residuals=True, regress_out_age=False),
    ),
    dict(
        modality="dname",
        metric="residuals_adj",
        df=get_dname_data(convert_to_residuals=False, regress_out_age=True),
    ),
]

for run in runs:
    modality = run["modality"]
    metric = run["metric"]
    dfo = run["df"]
    model_names = dfo.columns.tolist()

    fig, axes = None, np.asarray([[None] * len(model_names)] * len(organs))

    # fig, axes = plt.subplots(
    #     len(organs),
    #     len(model_names),
    #     figsize=(2.2 * len(model_names), 2 * len(organs)),
    #     sharex=False,
    #     sharey=True,
    # )
    _fits = list()
    for organ, axs in zip(organs, axes):
        for model_name, ax in tqdm(zip(model_names, axs)):
            df = dfo[[model_name]].join(t.query("Organ == @organ")["value"]).dropna()

            if df.empty:
                if ax is not None:
                    ax.set_visible(False)
                continue

            x = df[model_name].copy()
            x = (x - x.mean()) / x.std()
            y = df["value"].copy()

            _fits.append(
                sm.OLS(y, sm.add_constant(x))
                .fit()
                .summary2()
                .tables[1]
                .assign(
                    model_name=model_name, Organ=organ, metric=metric, n=df.shape[0]
                )
                .loc[model_name]
            )
            if ax is None:
                continue

            if True:
                df["q"] = pd.qcut(df[model_name], 5)
                v = df["q"].value_counts().sort_index()
                mapping = {v: k for k, v in enumerate(v.index, 1)}
                df["q"] = df["q"].replace(mapping)
            else:
                bins = [-25, -5, -2, 2, 5, 25]
                df["q"] = pd.cut(df[model_name], bins)
            sns.barplot(y="q", x="value", data=df, ax=ax, orient="horiz")
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
                title=f"{organ} (n = {df.shape[0]}), {model_name}",
                xlabel="Telomere length (Z-score)",
                ylabel=model_name,
            )
            ax.yaxis.set_inverted(False)
    if fig is not None:
        fig.tight_layout()
        fig.savefig(results_dir / f"{modality}.telomere_in_bins.barplot.svg", **figkws)

    fits = pd.concat(_fits, axis=1).T.reset_index(drop=True)
    fits["-log10(P>|t|)"] = -np.log10(fits["P>|t|"].astype(float))
    fits.to_csv(
        results_dir / f"{modality}.telomere_length_all_models.{metric}.stats.csv",
        index=False,
    )

    fits = pd.read_csv(
        results_dir / f"{modality}.telomere_length_all_models.{metric}.stats.csv"
    )

    fig, axes = plt.subplots(1, 2, figsize=(8, 8), sharex=True)
    p = fits.pivot_table(
        index=["model_name"],
        columns=["Organ", "metric"],
        values=["Coef.", "-log10(P>|t|)"],
    ).astype(float)
    order = p["Coef."].mean(axis=1).sort_values(ascending=False).index.tolist()
    p = p.reindex(order)
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
        vmin=-0.32,
        vmax=0.32,
    )
    fig.tight_layout()
    fig.savefig(
        results_dir
        / f"{modality}.telomere_length_all_models.{metric}.stats.heatmap.svg",
        **figkws,
    )

    ci_lower = fits["[0.025"].values
    ci_upper = fits["0.975]"].values
    coef_vals = fits["Coef."].values
    yerr_lower = coef_vals - ci_lower
    yerr_upper = ci_upper - coef_vals

    fig, axes = plt.subplots(
        2, 1, figsize=(5, 6), sharex=True, gridspec_kw={"hspace": 0.05}
    )

    for i, organ in enumerate(["Colon", "Lung"]):
        data = fits.query("metric == @metric & Organ == @organ").set_index("model_name")
        x_positions = [order.index(m) for m in data.index if m in order]
        coefs = [data.loc[m, "Coef."] for m in data.index if m in order]
        lower = [yerr_lower[data.index.get_loc(m)] for m in data.index if m in order]
        upper = [yerr_upper[data.index.get_loc(m)] for m in data.index if m in order]
        axes[0].bar(
            [x + i * 0.4 for x in x_positions],
            coefs,
            width=0.4,
            label=organ,
            yerr=[lower, upper],
            capsize=2,
        )
    axes[0].set(ylim=(-0.32, 0.32))
    axes[0].set_xticks(range(len(order)))
    axes[0].set_xticklabels(order, rotation=90, ha="right", va="top")
    axes[0].legend()

    sns.barplot(
        data=fits.query("metric == @metric"),
        x="model_name",
        y="-log10(P>|t|)",
        hue="Organ",
        ax=axes[1],
        order=order,
    )
    axes[1].set(ylim=(0, 5.1))
    for ax in axes:
        ax.set_xlabel("")
        ax.axhline(0, color="grey", linestyle="--")
    axes[1].set_xticklabels(
        axes[1].get_xticklabels(), rotation=90, ha="right", va="top"
    )
    fig.tight_layout()
    fig.suptitle("Telomere length vs Age residuals")
    fig.savefig(
        results_dir
        / f"{modality}.telomere_length_all_models.{metric}.stats.barplot.svg",
        **figkws,
    )
