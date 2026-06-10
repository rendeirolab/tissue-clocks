from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import get_restricted_info, get_pathology_data


metadata_dir = Path("metadata")
data_dir = Path("data")
expr_dir = Path("data") / "gtex" / "gene_expression"
gaps_dir = Path("results") / "gtex" / "fine_tuned" / "_pre_2024-01-19_age_X_frac1.0"
output_dir = Path("results") / "gtex" / "predict_gaps_from_blood_expression"
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

suffix = ".age_regressed"
# target_var = "Age"
target_var = "residuals_adj"
preds = pd.read_csv(
    output_dir
    / f"tissue-specific_clocks.Ridge.KFold.predictions_residuals.predicted_from_blood_expression{suffix}.{target_var}.predictions.csv",
    index_col=0,
)
preds = preds.query("Tissue not in @exclude_entities")
preds = pd.concat(
    [
        preds.query("Shuffled == False")
        .rename(columns=dict(prediction="residuals_real"))
        .drop(["Shuffled"], axis=1),
        preds.query("Shuffled == True")
        .rename(columns=dict(prediction="residuals_shuffled"))
        .drop(["Shuffled", "Tissue"], axis=1),
    ],
    axis=1,
)
meta = pd.read_csv(data_dir / "gtex" / "GTEx Portal.csv", index_col=0)
rest, _ = get_restricted_info()
preds = preds.join(rest[["Age"]])
tissues = sorted(
    preds["Tissue"].drop_duplicates().drop(exclude_entities, errors="ignore")
)


# # prepare pathology info
path_df = get_pathology_data().join(meta[["Tissue"]])
path_df.index = path_df.index.str.extract(r"(GTEX-\w+)-\d{4}", expand=False)
p = path_df.drop("Tissue", axis=1)
gp = (
    p.groupby(level=0).sum()
    / p.groupby(level=0).apply(lambda x: (~x.isnull()).any().astype(int))
).assign(Tissue="mean")
gp["Body_pathology_burden"] = p.groupby(level=0).sum().sum(1) / p.groupby(
    level=0
).apply(lambda x: (~x.isnull()).sum()).mean(1)
path_df = pd.concat([path_df, gp], axis=0)
path_df.loc[path_df["Body_pathology_burden"].isnull(), "Body_pathology_burden"] = 0


#
exc = preds.columns.drop("Tissue").tolist()
_res = list()
for tissue in tissues:
    df = (
        preds.query("Tissue == @tissue")
        .drop(["Tissue"], axis=1)
        .join(path_df.query("Tissue == @tissue").drop(["Tissue"], axis=1))
    )
    df = df.loc[:, df.var() > 0]
    _res.append(
        df.corr().stack().reset_index().assign(Tissue=tissue, Metric="Correlation")
    )
    _res.append(
        df.drop(exc, axis=1)
        .sum()
        .to_frame(0)
        .rename_axis(index="level_0")
        .reset_index()
        .assign(level_1="__", Tissue=tissue, Metric="N")
    )
res = pd.concat(_res, axis=0)
res.columns = ["var_1", "var_2", "value", "Tissue", "Metric"]


fig, axes = plt.subplots(1, 2, figsize=(4 * 3, 4), sharex=True, sharey=True)
for metric, ax in zip(
    ["Age", "residuals_real", "residuals_shuffled"],
    [axes[0], axes[1], axes[1]],
):
    v = (
        res.query(f"var_1 == '{metric}' & ~var_2.isin(@exc) & Metric == 'Correlation'")
        .pivot_table(
            index="Tissue",
            columns="var_2",
            values="value",
        )
        .stack()
        .sort_values()
    )
    sns.histplot(v, ax=ax, label=metric)
    ax.set(xlabel="Correlation", title=metric)
    for x in [-0.1, 0, 0.1]:
        ax.axvline(x, color="grey", linestyle="--")


fig1, axes1 = plt.subplots(1, 2, figsize=(3.25 * 2, 2.75), sharex=True, sharey=True)
fig2, axes2 = plt.subplots(
    20, 2, figsize=(2.75 * 2, 2.75 * 10), sharex=True, sharey=False
)
for metric, ax1, ax2 in zip(
    ["Age", "residuals_real", "residuals_shuffled"],
    [axes1[0], axes1[1], axes1[1]],
    [axes2[:, 0], axes2[:, 1], axes2[:, 1]],
):
    color = "grey" if "shuffled" in metric else "tab:blue"
    v = (
        res.query(f"var_1 == '{metric}' & ~var_2.isin(@exc) & Metric == 'Correlation'")
        .pivot_table(
            index="Tissue",
            columns="var_2",
            values="value",
        )
        .stack()
        .sort_values()
        .rename("Correlation")
    )
    n = (
        res.query("Metric == 'N'")
        .set_index(["Tissue", "var_1"])["value"]
        .reindex(v.index)
        .rename("N")
    )
    x = pd.concat([n, v], axis=1)
    ax1.scatter(data=x, x="N", y="Correlation", color=color, alpha=0.75, s=10)
    ax1.axhline(0, color="grey", linestyle="--")
    ax1.set(xlabel="Number of samples", ylabel="Correlation to pathology", xscale="log")

    if "shuffled" not in metric:
        use = x.query("N > 100").index[-18:].tolist()
        use += x.query("N > 100").index[:2].tolist()

    for (tissue, var_2), ax in zip(use, ax2):
        ax1.text(
            x.loc[(tissue, var_2), "N"],
            x.loc[(tissue, var_2), "Correlation"],
            f"{tissue}, {var_2}",
        )
        df = (
            preds.query("Tissue == @tissue")
            .drop(["Tissue"], axis=1)
            .join(path_df.query("Tissue == @tissue").drop(["Tissue"], axis=1))
        )
        x1 = df.query(f"{var_2} == False")[[metric]].assign(pathology=False)
        x2 = df.query(f"{var_2} == True")[[metric]].assign(pathology=True)
        x3 = pd.concat([x1, x2])
        sns.boxplot(
            data=x3,
            x="pathology",
            y=metric,
            ax=ax,
            color=color,
            fill=True,
            saturation=0.5,
            whis=0.75,
            showcaps=False,
            showfliers=False,
            medianprops=dict(linewidth=2, color=color),
            meanprops=dict(linewidth=2, color=color),
            capprops=dict(linewidth=2, color=color),
            whiskerprops=dict(linewidth=2, color=color),
        )
        # sns.barplot(
        #     data=x3,
        #     x="pathology",
        #     y=metric,
        #     ax=ax,
        #     color=color,
        #     fill=False,
        #     saturation=0.5,
        # )
        nn = x3["pathology"].value_counts()
        if "shuffled" not in metric:
            ax.text(1, x3[metric].mean(), f"{nn[True]}/{nn.sum()}", ha="center")
        ax.set(title=f"{tissue}, {var_2}")
# fig1.tight_layout()
# fig2.tight_layout()
fig1.savefig(output_dir / "correlation_to_pathology_vs_sample_size.svg", **figkws)
fig2.savefig(output_dir / "correlation_to_pathology.examples.svg", **figkws)

for ax in fig1.axes:
    ax.set(xscale="linear")
fig1.savefig(
    output_dir / "correlation_to_pathology_vs_sample_size.linear_x.svg", **figkws
)


# Compare chronological and biological age with pathology incidence
for var1, var2, label in [
    ("Age", "residuals_real", "Age_vs_Gap_real"),
    ("residuals_shuffled", "residuals_real", "Gap_shuffled_vs_Gap_real"),
]:
    p1 = res.query(
        f"var_1 == '{var1}' & ~var_2.isin(@exc) & Metric == 'Correlation'"
    ).set_index(["Tissue", "var_2"])["value"]
    p2 = res.query(
        f"var_1 == '{var2}' & ~var_2.isin(@exc) & Metric == 'Correlation'"
    ).set_index(["Tissue", "var_2"])["value"]
    d = p1.to_frame(var1).join(p2.rename(var2)).astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(2 * 3.3, 3))
    axes[0].scatter(data=d, x=var1, y=var2, alpha=0.5, s=10)
    sns.regplot(data=d, x=var1, y=var2, ax=axes[0], scatter=False)
    # for tissue in d.index:
    #     axes[0].text(d.loc[tissue, var1], d.loc[tissue, var2], tissue)
    axes[0].axhline(0, color="grey", linestyle="--")
    axes[0].axvline(0, color="grey", linestyle="--")
    vmin = d.min().min()
    vmax = d.max().max()
    axes[0].plot([vmin, vmax], [vmin, vmax], color="grey", linestyle="--")
    n1, n2 = label.split("_vs_")
    axes[0].set(
        xlabel=f"Correlation between\npathology incidence and {n1}",
        ylabel=f"Correlation between\npathology incidence and {n2}",
        title=label,
    )

    base = var2 if var1 != "Age" else var1
    d = d.loc[d[base] > 0.1]
    d["fc"] = signed_fold_change(d.loc[d.index, var2], d.loc[d.index, var1])
    axes[1].scatter(data=d, x=base, y="fc", alpha=0.5, s=10)
    for tissue in d["fc"].sort_values().index[-10:]:
        axes[1].text(d.loc[tissue, base], d.loc[tissue, "fc"], ", ".join(tissue))
    for tissue in d[base].sort_values().index[-10:]:
        axes[1].text(d.loc[tissue, base], d.loc[tissue, "fc"], ", ".join(tissue))
    axes[1].axhline(0, color="grey", linestyle="--")
    axes[1].set(
        xlabel=f"{base} correlation with pathology incidence",
        ylabel=f"Fold difference in correlation with pathology incidence\nbetween {n1} and {n2}",
        title=label,
    )
    if var1 != "Age":
        axes[1].set(yscale="log")
    # fig.tight_layout()
    fig.savefig(output_dir / f"pathology_comparison.{label}.svg", **figkws)
