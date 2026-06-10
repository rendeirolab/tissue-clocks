from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import get_restricted_info, get_telomere_lengths


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
p = (
    res.query("var_1 == 'residuals_real' & var_2 == 'TQImean' & n > 100")
    .set_index("Tissue")
    .sort_values("value")
)

fig, ax = plt.subplots(figsize=(4, 4))
ax.scatter(p[("n")], p[("value")])
for tissue, x, y in zip(p.index, p[("n")], p[("value")]):
    ax.text(x, y, tissue)
ax.axhline(0, color="grey", linestyle="--")
ax.set(xlabel="Number of samples", ylabel="Correlation to telomere length")
fig.tight_layout()
fig.savefig(output_dir / "blood_vs_telomere.correlation.svg", **figkws)

fig, axes = plt.subplots(3, 6, figsize=(1.8 * 6, 2 * 3), sharex=True, sharey=True)
for tissue, ax in zip(p.index, axes.flatten()):
    df = preds.query("Tissue == @tissue").join(tln.loc[tissue])
    ax.scatter(df["residuals_real"], df["TQImean"], alpha=0.5, s=2)
    sns.regplot(x=df["residuals_real"], y=df["TQImean"], ax=ax, scatter=False)
    ax.text(0.05, 0.9, f"{p.loc[tissue, 'value']:.3f}", transform=ax.transAxes)
    ax.axhline(0, color="grey", linestyle="--")
    ax.set(title=tissue, xlabel="", ylabel="")
fig.supxlabel("Blood predicted age gap")
fig.supylabel("Telomere length (Z-score)")
fig.tight_layout()
fig.savefig(output_dir / "blood_vs_telomere.svg", **figkws)
