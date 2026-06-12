# SPDX-FileCopyrightText: Copyright (c) 2026 Rendeiro Lab, CeMM - Research Center for Molecular Medicine, Austrian Academy of Sciences
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#
# Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE).

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import get_restricted_info, signed_fold_change, get_telomere_lengths


metadata_dir = Path("metadata")
data_dir = Path("data")
expr_dir = Path("data") / "gtex" / "gene_expression"
gaps_dir = Path("results") / "gtex" / "fine_tuned" / "_pre_2024-01-19_age_X_frac1.0"
output_dir = Path("results") / "gtex" / "predict_gaps_from_tissue"
output_dir.mkdir(parents=True, exist_ok=True)
figkws = dict(dpi=300, bbox_inches="tight")

exclude_entities = [
    "Bladder",
    "Cervix - Ectocervix",
    "Cervix - Endocervix",
    "Fallopian Tube",
    "Kidney - Medulla",
]
exclude_entities += ["max", "sum", "std"]

# target_var = "Age"
target_var = "residuals_adj"
preds = pd.read_parquet(
    gaps_dir / "tissue-specific_clocks.Ridge.GroupKFold.predictions_residuals.pq"
)
preds = preds.query("Tissue not in @exclude_entities")
preds.index = preds.index.to_series().str.extract(r"(GTEX-\w+)-\d{4}", expand=False)
ind = preds.drop("Tissue", axis=1).groupby(level=0).mean().assign(Tissue="mean")
preds = pd.concat([preds, ind])
meta = pd.read_csv(data_dir / "gtex" / "GTEx Portal.csv", index_col=0)
rest, _ = get_restricted_info()
tissues = sorted(
    preds["Tissue"].drop_duplicates().drop(exclude_entities, errors="ignore")
)


# # prepare telomere lengths
tl = (
    get_telomere_lengths()
    .set_index(["TissueSiteDetail", "CollaboratorParticipantID"])["TQImean"]
    .sort_index()
)
tln = (
    tl.groupby(level="TissueSiteDetail")
    .apply(lambda x: (x - x.mean()) / x.std())
    .reset_index(level=1, drop=True)
    .dropna()
)
tlng = tln.groupby("CollaboratorParticipantID").mean()
tln = pd.concat(
    [
        tln,
        tlng.to_frame()
        .assign(TissueSiteDetail="mean")
        .set_index("TissueSiteDetail", append=True)
        .reorder_levels([1, 0])
        .sort_index(),
    ]
)

_res = list()
for tissue in tissues:
    if tissue not in tln.index.get_level_values("TissueSiteDetail"):
        continue
    df = (
        preds.query("Tissue == @tissue")
        .join(tln.loc[tissue])
        .drop(["Tissue"], axis=1)
        .dropna()
    )
    if df.shape[0] < 3:
        continue
    _res.append(df.corr().stack().reset_index().assign(Tissue=tissue, n=df.shape[0]))
res = pd.concat(_res, axis=0)
res.columns = ["var_1", "var_2", "value", "Tissue", "n"]


for metric in ["Age", "prediction_adj", "residuals_adj"]:
    fig, axes = plt.subplots(
        4, 10, figsize=(2 * 12.25, 2 * 4), sharex=False, sharey=True
    )
    for tissue, ax in zip(sorted(res["Tissue"].unique()), axes.flatten()):
        df = (
            preds.query("Tissue == @tissue")
            .join(tln.loc[tissue])
            .drop(["Tissue"], axis=1)
            .dropna()
        )
        df = df.groupby(level=0).mean()
        if df.shape[0] < 100:
            ax.set(title=tissue, xlabel="", ylabel="")
            ax.set_visible(False)
            continue
        if metric != "residuals_adj":
            df["q"] = pd.qcut(df[metric], 5)
            v = df["q"].value_counts().sort_index()
            mapping = {v: k for k, v in enumerate(v.index, 1)}
            df["q"] = df["q"].replace(mapping)
        else:
            bins = [-25, -5, -2, 2, 5, 25]
            df["q"] = pd.cut(df[metric], bins)
        sns.barplot(y="q", x="TQImean", data=df, ax=ax, orient="horiz")
        if metric != "residuals_adj":
            for i in range(1, 6):
                ax.text(ax.get_xticks()[-1], i - 1, df.query("q == @i").shape[0])
        else:
            for i, b in enumerate(df["q"].unique()):
                ax.text(ax.get_xticks()[-1], i, f"n = {df.query('q == @b').shape[0]}")
            ax.axvline(0, color="grey", linestyle="--")
        ax.set(
            title=tissue,
            xlabel="Telomere length (Z-score)",
            ylabel="Biological age (bin)",
        )
        ax.yaxis.set_inverted(False)
    fig.tight_layout()
    fig.savefig(output_dir / f"tissue_{metric}_telomere_in_bins.barplot.svg", **figkws)


for metric in [
    "Age",
    "prediction_adj",
    "prediction-shuffled",
    "residuals_adj",
    "residuals-shuffled",
]:
    p = (
        res.query(f"var_1 == '{metric}' & var_2 == 'TQImean' & n > 100")
        .set_index("Tissue")
        .sort_values("value")
    )

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.scatter(p[("n")], p[("value")])
    for tissue in p.index:
        ax.text(p.loc[tissue, "n"], p.loc[tissue, "value"], tissue)
    ax.axhline(0, color="grey", linestyle="--")
    ax.axvline(0, color="grey", linestyle="--")
    ax.set(xlabel="Number of samples", ylabel="Correlation", ylim=(-0.45, 0.15))
    fig.tight_layout()
    fig.savefig(output_dir / f"tissue_{metric}_vs_n.svg", **figkws)

    fig, axes = plt.subplots(3, 6, figsize=(2 * 6, 2 * 3), sharex=True, sharey=True)
    for tissue, ax in zip(p.index, axes.flatten()):
        df = preds.query("Tissue == @tissue").join(tln.loc[tissue])
        ax.scatter(df[metric], df["TQImean"], alpha=0.5, s=2)
        sns.regplot(x=df[metric], y=df["TQImean"], ax=ax, scatter=False)
        ax.text(0.05, 0.9, f"{p.loc[tissue, 'value']:.3f}", transform=ax.transAxes)
        ax.axhline(0, color="grey", linestyle="--")
        ax.axvline(0, color="grey", linestyle="--")
        ax.set(title=tissue, xlabel="", ylabel="")
    fig.supxlabel("Tissue predicted age gap")
    fig.supylabel("Telomere length (Z-score)")
    fig.tight_layout()
    fig.savefig(output_dir / f"tissue_{metric}_vs_telomere.svg", **figkws)

    fig, axes = plt.subplots(3, 6, figsize=(2 * 6, 2 * 3), sharex=False, sharey=False)
    for tissue, ax in zip(p.index, axes.flatten()):
        df = preds.query("Tissue == @tissue").join(tln.loc[tissue])
        ax.scatter(df[metric], df["TQImean"], alpha=0.5, s=2)
        sns.regplot(x=df[metric], y=df["TQImean"], ax=ax, scatter=False)
        ax.text(0.05, 0.9, f"{p.loc[tissue, 'value']:.3f}", transform=ax.transAxes)
        ax.axhline(0, color="grey", linestyle="--")
        ax.axvline(0, color="grey", linestyle="--")
        ax.set(title=tissue, xlabel="", ylabel="")
    fig.supxlabel("Tissue predicted age gap")
    fig.supylabel("Telomere length (Z-score)")
    fig.tight_layout()
    fig.savefig(output_dir / f"tissue_{metric}_vs_telomere.free.svg", **figkws)


# Compare chronological and biological age with telomere length
for var1, var2, label in [
    ("Age", "prediction_adj", "Age_vs_Predicted_real"),
    ("prediction-shuffled", "prediction_adj", "Predicted_shuffled_vs_Predicted_real"),
    ("residuals-shuffled", "residuals_adj", "Gap_shuffled_vs_Gap_real"),
]:
    p1 = res.query(f"var_1 == '{var1}' & var_2 == 'TQImean' & n > 100").set_index(
        "Tissue"
    )["value"]
    p2 = res.query(f"var_1 == '{var2}' & var_2 == 'TQImean' & n > 100").set_index(
        "Tissue"
    )["value"]
    d = p1.to_frame(var1).join(p2.rename(var2))

    fig, axes = plt.subplots(1, 2, figsize=(2 * 5, 3))
    axes[0].scatter(data=d, x=var1, y=var2)
    for tissue in d.index:
        axes[0].text(d.loc[tissue, var1], d.loc[tissue, var2], tissue)
    axes[0].axhline(0, color="grey", linestyle="--")
    axes[0].axvline(0, color="grey", linestyle="--")
    vmin = d.min().min()
    vmax = d.max().max()
    axes[0].plot([vmin, vmax], [vmin, vmax], color="grey", linestyle="--")
    n1, n2 = label.split("_vs_")
    axes[0].set(
        xlabel=f"Correlation between\ntelomere length and {n1}",
        ylabel=f"Correlation between\ntelomere length and {n2}",
        title=label,
    )
    mean = d.mean(1)
    mean = mean.loc[mean < -0.1]
    fc = signed_fold_change(d.loc[mean.index, var1], d.loc[mean.index, var2])
    axes[1].scatter(mean, fc)
    for tissue in mean.index:
        axes[1].text(d.mean(1)[tissue], fc[tissue], tissue)
    axes[1].axhline(0, color="grey", linestyle="--")
    axes[1].set(
        xlabel="Mean correlation with telomere length",
        ylabel=f"Fold difference in correlation with telomere length\nbetween {n1} and {n2}",
        title=label,
    )
    fig.tight_layout()
    fig.savefig(output_dir / f"comparison.{label}.svg", **figkws)
