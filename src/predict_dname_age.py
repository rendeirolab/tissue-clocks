"""
Predict age of GTEx samples using DNA methylation data.

Uses a local fork of pyaging to added the remove_added_features flag to the
predict_age function to prevent the removal of added (0) missing features
and speedup the process.

Run on CPUS with CUDA_VISIBLE_DEVICES="" python src/gtex_predict_dname_age.py
"""

from pathlib import Path

from tqdm import tqdm
import pandas as pd
import pyaging as pya
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib
import matplotlib.pyplot as plt

from src.utils import get_restricted_info

matplotlib.use("Agg")

# GTEx
data_dir = Path("data") / "gtex" / "dna_methylation"
results_dir = Path("results") / "gtex" / "dna_methylation" / "pyage"
results_dir.mkdir(exist_ok=True, parents=True)
dname_file = (
    data_dir / "GSE213478_methylation_DNAm_noob_final_BMIQ_all_tissues_987.txt.gz"
)

# Load entire data (takes long but it's needed)
print("Loading data...")
x = pd.read_csv(dname_file, index_col=0, engine="pyarrow").T

# Get Age and Individual ID
print("Adding metadata (obs)...")
y, _ = get_restricted_info()
voi = ["Age", "Sex", "Cohort"]
o = x.index.str.extract(r"(?P<SUBJID>GTEX-\w+)-\w+", expand=False).to_frame(name="asd")
obs = o.join(y[voi], on="SUBJID").drop("asd", axis=1)
obs.index = x.index

# Get Tissue type
meta = pd.read_csv(data_dir.parent / "GTEx Portal.csv", index_col=0)
obs["Tissue ID"] = obs.index.str.extract(r"(GTEX-\w+-\w+).*", expand=False)
obs["Tissue"] = obs["Tissue ID"].map(meta["Tissue"]).fillna("Blood")
voi += ["Tissue"]

adata = pya.pp.df_to_adata(x.join(obs.drop("Tissue ID", axis=1)), metadata_cols=voi)

clocks = [
    "altumage",
    "dnamphenoage",
    "dnamtl",
    "dunedinpace",
    "hannum",
    "horvath2013",
    "hrsinchphenoage",
    "knight",
    "leecontrol",
    "leerefinedrobust",
    "leerobust",
    "lin",
    "mammalian1",
    "mammalian2",
    "mammalian3",
    "mammalianlifespan",
    "pcdnamtl",
    "pcgrimage",
    "pchannum",
    "pchorvath2013",
    "pcphenoage",
    "pcskinandblood",
    "pedbe",
    "replitali",
    "skinandblood",
    "zhangblup",
    "zhangen",
    "zhangmortality",
]
n_clocks = len(clocks)
print(f"Predicting age using {n_clocks} clocks...")
# adata = pya.pred.predict_age(adata, clocks, verbose=True)
for clock in tqdm(clocks):
    if clock in adata.obs:
        continue
    adata = pya.pred.predict_age(
        adata, clock, verbose=False, remove_added_features=False
    )
    adata.obs.to_csv(results_dir / "gtex_age_prediction.csv")
    clock_meta = pd.DataFrame(
        {k: v for k, v in adata.uns.items() if k.endswith("_metadata")}
    ).T
    clock_meta["percent_na"] = pd.Series(
        {
            k.replace("_percent_na", "_metadata"): v
            for k, v in adata.uns.items()
            if k.endswith("_percent_na")
        }
    ).T
    clock_meta.to_csv(results_dir / "gtex_age_prediction.uns.csv")

print("Writing adata to disk...")
adata.write(results_dir / "gtex_age_prediction.h5ad")

obs = pd.read_csv(results_dir / "gtex_age_prediction.csv", index_col=0)
uns = pd.read_csv(results_dir / "gtex_age_prediction.uns.csv", index_col=0)
metrics = dict()
for m in ["pearson", "spearman", mean_absolute_error, mean_squared_error]:
    metrics[m] = obs.iloc[:, -n_clocks:].corr(m)
if "Age" in obs:
    metrics["ground_truth_mean_absolute_error"] = (
        obs.iloc[:, -n_clocks:] - obs["Age"].values.reshape(-1, 1)
    ).mean(0)
    metrics["ground_truth_mean_squared_error"] = (
        (obs.iloc[:, -n_clocks:] - obs["Age"].values.reshape(-1, 1)) ** 2
    ).mean(0)

# Plot
print("Plotting...")
n_tissues = len(obs["Tissue"].unique())
fig_grid = (n_clocks, n_tissues)
lims = (15, 75)

fig, axes = plt.subplots(*fig_grid, figsize=(n_tissues * 3, n_clocks * 3))
for i, tissue in enumerate(adata.obs["Tissue"].unique()):
    for j, clock in enumerate(clocks):
        ax = axes[j, i]
        ax.plot(
            lims,
            lims,
            color="black",
            linestyle="--",
            linewidth=1,
            alpha=0.5,
            zorder=0,
        )
        ax.scatter(
            adata.obs.loc[adata.obs["Tissue"] == tissue, "Age"],
            adata.obs.loc[adata.obs["Tissue"] == tissue, clock.lower()],
            alpha=0.75,
            s=5,
        )
        ax.set(
            title=f"{clock} ({tissue})",
            ylabel=("Predicted Age"),
            xlabel=("Chronological Age"),
            xlim=lims,
        )
fig.tight_layout()
fig.savefig(results_dir / "gtex_age_prediction.svg", dpi=300, bbox_inches="tight")


fig, axes = plt.subplots(*fig_grid, figsize=(n_tissues * 3, n_clocks * 3))
for i, tissue in enumerate(obs["Tissue"].unique()):
    for j, clock in enumerate(clocks):
        ax = axes[j, i]
        # ax.plot(
        #     lims,
        #     lims,
        #     color="black",
        #     linestyle="--",
        #     linewidth=1,
        #     alpha=0.5,
        #     zorder=0,
        # )
        ax.scatter(
            obs.loc[obs["Tissue"] == tissue, "Age"],
            obs.loc[obs["Tissue"] == tissue, clock.lower()],
            alpha=0.75,
            s=5,
        )
        ax.set(
            title=f"{clock} ({tissue})",
            ylabel=("Predicted Age"),
            xlabel=("Chronological Age"),
            xlim=lims,
        )
fig.tight_layout()
fig.savefig(
    results_dir / "gtex_age_prediction.free_y_scale.svg", dpi=300, bbox_inches="tight"
)
fig.savefig(
    results_dir / "gtex_age_prediction.free_y_scale.png", dpi=96, bbox_inches="tight"
)

print("Done.")
