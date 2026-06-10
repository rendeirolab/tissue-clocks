from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from seaborn_extensions import clustermap
import pingouin as pg


metadata_dir = Path("metadata")
data_dir = Path("data") / "gtex" / "svs"
processed_dir = Path("processed") / "histopathology"
processed_dir.mkdir(exist_ok=True, parents=True)
results_dir = Path("results") / "tissue_clocks_revision"
results_dir.mkdir(exist_ok=True, parents=True)
output_dir = results_dir / "correlate_histology_dname"
figkws = dict(dpi=300, bbox_inches="tight")

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
]
mpp: float = 0.5
tile_width: int = 224

cohort_name = "lungaging-schiller"


# Get histology gaps
histo_gaps_f = (
    results_dir
    / "clock"
    / f"cohort_{cohort_name}.predictions.{mpp}mpp.{tile_width}px.all_models.csv"
)
histo_gaps = pd.read_csv(histo_gaps_f).query("Cohort==@cohort_name & Organ=='Lung'")

# add more IDs
f = Path("~/projects") / cohort_name / "metadata" / "original" / "HE_lung_aging.xlsx"
histo_ids = pd.read_excel(f)
histo_gaps = histo_gaps.query("Model == 'virchow2'").merge(
    histo_ids[["HE_slide_ID", "Sample_ID"]],
    left_on="slide_ID",
    right_on="HE_slide_ID",
    how="left",
)


# Get DNAme gaps
f = Path("~/projects") / cohort_name / "data" / "Epigenetic_Results_with_Difference.csv"
dname_gaps = pd.read_csv(f)


# Join
df = histo_gaps.merge(
    dname_gaps, left_on=["Sample_ID", "Age"], right_on=["Lung_number", "Age"]
)

# Compare all to all
dname_clocks = [
    "mAge_Horvath",
    "mAge_Hannum",
    "PhenoAge",
    "Difference_from_Regression_mAge_Horvath",
    "Difference_from_Regression_mAge_Hannum",
    "Difference_from_Regression_PhenoAge",
]
metrics = ["Age", "pred", "Error"]
n, m = len(metrics), len(dname_clocks)

for cmap in ["inferno", "coolwarm"]:
    fig, axes = plt.subplots(n, m, figsize=(m * 3 * 1.2, n * 3))
    for metric, axs in zip(metrics, axes):
        for clock, ax in zip(dname_clocks, axs):
            r = pg.corr(df[metric], df[clock]).loc["pearson", "r"]
            s = ax.scatter(
                df[metric],
                df[clock],
                c=df["Age"] if cmap == "inferno" else df[[metric, clock]].mean(axis=1),
                cmap=cmap,
                s=15,
                alpha=0.75,
                vmin=0 if cmap == "inferno" else -20,
                vmax=100 if cmap == "inferno" else 20,
            )
            plt.colorbar(
                s, ax=ax, label="Age" if cmap == "inferno" else "Age gap", shrink=0.5
            )
            ax.text(0, 20, s=f"r = {r:.3f}")
            ax.set(xlabel=metric + " (Histo)", ylabel=clock + " (DNAme)")

            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            min_val = min(x_min, y_min)
            max_val = max(x_max, y_max)
            ax.plot(
                (min_val, max_val),
                (min_val, max_val),
                linestyle="--",
                color="grey",
                zorder=-1000,
            )
    fig.savefig(output_dir / f"scatter_plots.DNAme_vs_histology.{cmap=}.svg", **figkws)


# Plot consistency
histo_gaps = pd.read_csv(histo_gaps_f).query("Cohort==@cohort_name & Organ=='Lung'")

q = histo_gaps.pivot_table(index="Model", columns="slide_ID", values="Error")
g = clustermap(
    q, cmap="coolwarm", vmin=-25, vmax=25, figsize=(8, 4.5), dendrogram_ratio=0.05
)
g.fig.savefig(output_dir / "lungaging-schiller.consistency_models.svg", **figkws)
