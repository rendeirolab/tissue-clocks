#!/usr/bin/env uv --script

# /// script
# dependencies = [
#   "numpy",
#   "pandas",
#   "spatialdata>=0.4.0",
#   "lazyslide>=0.7.2",
#   "scanpy",
#   "conch",
# ]
# [tool.uv.sources]
# conch = { git = "https://github.com/mahmoodlab/CONCH.git" }
# ///

"""
Build clocks de novo.
"""

from pathlib import Path

from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn.metrics import r2_score


metadata_dir = Path("metadata")
data_dir = Path("data") / "gtex" / "svs"
processed_dir = Path("processed") / "histopathology"
processed_dir.mkdir(exist_ok=True, parents=True)
results_dir = Path("results") / "tissue_clocks_revision" / "new_cohorts_clocks"
results_dir.mkdir(exist_ok=True, parents=True)
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

cohorts = ["skinpath", "neuropath", "lungaging-schiller"]
cohort_attrs = {
    "skinpath": {"age": "Age at sampling", "group": "Individual ID"},
    "neuropath": {"age": "Age at sampling", "group": "Autopsy_ID"},
    "lungaging-schiller": {"age": "age", "group": "sample_ID"},
}


def main() -> None:
    for model_name in model_names:
        for cohort_name in cohorts:
            make_clock(cohort_name, model_name)

    summarize_performance()


def summarize_performance(
    mpp: float = 0.5,
    tile_width: int = 224,
):
    # mpp: float = 0.5; tile_width: int = 224
    _res = list()

    # Other cohorts
    for model_name in model_names:
        for cohort_name in cohorts:
            f = (
                results_dir
                / "clock"
                / f"{cohort_name}.ridgecv.{mpp}mpp.{tile_width}px.{model_name}.predictions.csv"
            )
            d = pd.read_csv(f, index_col=0).assign(Cohort=cohort_name, Model=model_name)
            d["MAE"] = (d["Age"] - d["pred"]).abs().mean()
            d["R2"] = r2_score(d["Age"], d["pred"])
            d["Pearson"] = d[["Age"]].corrwith(d["pred"]).loc["Age"]
            _res.append(d)
    res = pd.concat(_res)
    res.to_csv(
        results_dir / f"all_cohorts.statistics.{mpp}mpp.{tile_width}px.all_models.csv",
        index=False,
    )

    lims = dict(R2=(0, 1), Pearson=(-1, 1), MAE=(0, 30))
    cmaps = dict(R2="vlag", Pearson="vlag", MAE="Reds_r")
    fig, axes = plt.subplots(3, 1, figsize=(10, 3 * 1.5), sharex=True)
    for metric, ax in zip(["R2", "Pearson", "MAE"][::-1], axes):
        resp = res.pivot_table(index=["Cohort"], columns="Model", values=metric)
        sns.heatmap(
            resp.join(resp.mean(1).rename("mean")),
            ax=ax,
            vmin=lims[metric][0],
            vmax=lims[metric][1],
            annot=True,
            fmt=".2f",
            cmap=cmaps[metric],
        )
        ax.set(title=metric)
    fig.tight_layout()
    fig.savefig(
        results_dir
        / f"all_cohorts.predictions.{mpp}mpp.{tile_width}px.all_models.heatmaps.svg",
        **figkws,
    )


def make_clock(
    cohort_name: str,
    model_name: str = "virchow2",
    mpp: float = 0.5,
    tile_width: int = 224,
):
    # cohort_name: str = "skinpath"; model_name: str = "virchow2"; mpp: float = 0.5; tile_width: int = 224
    suffix = f".{mpp}mpp.{tile_width}px.{model_name}"
    output_dir = results_dir / "clock"
    output_dir.mkdir(exist_ok=True, parents=True)

    if (output_dir / f"ridgecv{suffix}.svg").exists():
        return

    h = (
        Path().absolute().parent
        / Path(cohort_name)
        / "results"
        / "tissue_clocks_revision"
        / f"anndata.{suffix}.h5ad"
    )
    a = sc.read_h5ad(h)

    cv = GroupKFold(5)

    x = a.to_df()
    y = a.obs[cohort_attrs[cohort_name]["age"]]
    groups = a.obs[cohort_attrs[cohort_name]["group"]]

    alphas = np.logspace(-2, 3, 20)
    ridgecv = RidgeCV(alphas=alphas, fit_intercept=True)
    model = make_pipeline(StandardScaler(), ridgecv)
    y_pred = cross_val_predict(model, x, y, cv=cv, groups=groups, n_jobs=-1)
    pd.Series(y_pred, index=y.index).to_frame("pred").join(y.rename("Age")).rename_axis(
        index=None
    ).to_csv(output_dir / f"{cohort_name}.ridgecv{suffix}.predictions.csv")

    r2 = r2_score(y, y_pred)
    mae = (y_pred - y).abs().mean()
    fig, ax = plt.subplots(1, 1, figsize=(4 * 1, 4))
    d = y_pred - y
    sns.regplot(x=y, y=y_pred, scatter=False, ax=ax, color="grey")
    ax.scatter(
        y,
        y_pred,
        alpha=0.5,
        c=d,
        cmap="coolwarm",
        vmin=-abs(d).max(),
        vmax=abs(d).max(),
        rasterized=True,
    )
    vmin, vmax = pd.Series(y.tolist() + y_pred.tolist()).describe().loc[["min", "max"]]
    ax.plot((vmin, vmax), (vmin, vmax), linestyle="--", color="grey")
    ax.set(
        xlabel="Chronological age",
        ylabel="Biological age",
        title=f"MAE={mae:.2f}, R2={r2:.2f}",
    )
    fig.savefig(output_dir / f"{cohort_name}.ridgecv{suffix}.svg", **figkws)

    print(model_name, x.shape[0], x.shape[1], r2, mae)


def fit_scaler(x):
    scaler = StandardScaler()
    scaler.fit(x)
    return scaler


def fit_calibrator(y_pred, y_true):
    calibrator = LinearRegression()
    calibrator.fit(y_pred.reshape(-1, 1), y_true)
    return calibrator


if __name__ == "__main__":
    main()
