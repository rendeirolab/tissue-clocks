# SPDX-FileCopyrightText: Copyright (c) 2026 Rendeiro Lab, CeMM - Research Center for Molecular Medicine, Austrian Academy of Sciences
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#
# Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE).

"""
Make tissue clocks from imagenet model features.

NOTE: This script references hardcoded machine-specific paths for
ImageNet feature directories. It is included for transparency and is
not expected to run outside the original machine.
"""

from pathlib import Path
from collections import defaultdict

import parmap
import numpy as np
import pandas as pd
from anndata import AnnData
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
import seaborn as sns

from src.utils import get_restricted_info


data_dir = Path("data") / "gtex" / "svs"
dirs = defaultdict(lambda: data_dir)
dirs["alexnet"] = (
    Path("/research") / "lab_rendeiro" / "projects" / "histopath" / "features"
)
dirs["resnet50"] = (
    Path("/research") / "lab_rendeiro" / "projects" / "histopath" / "features"
)
features_dir = data_dir.parent / "features"
features_dir.mkdir(parents=True, exist_ok=True)

results_dir = Path("results") / "tissue_clocks_revision" / "gtex_imagenet_clocks"
results_dir.mkdir(parents=True, exist_ok=True)

meta = pd.read_csv(data_dir.parent / "GTEx Portal.csv", index_col=0)
meta["Organ"] = meta["Tissue"].str.split(" - ").str[0]
rest, _ = get_restricted_info()
meta = meta.merge(rest[["Age"]], left_on="Subject ID", right_index=True, how="left")

figkws = dict(bbox_inches="tight", dpi=300, transparent=True, pad_inches=0.1)

model_names = [
    "alexnet",
    "vgg16",
    # "resnet28",
    "resnet50",
    # "resnet152",
    "convnext_tiny",
    "convnext_base",
    "convnext_large",
    "maxvit_t",
]
tile_size = 224
random_state = 42


def main():
    for model_name in model_names:
        collect(model_name)
        make_clock(model_name)
    compare_models()


def compare_models():
    all_metrics = list()
    for model_name in model_names:
        suffix = f"{model_name}.{tile_size}px"
        f = results_dir / f"ridgecv_{suffix}.metrics.csv"
        if not f.exists():
            continue
        all_metrics.append(pd.read_csv(f))
    all_metrics = pd.concat(all_metrics)
    all_metrics.to_csv(
        results_dir / "ridgecv_imagenet_models.all_models.csv", index=False
    )

    params = {
        "r2": {"vmin": -0.25, "center": 0, "vmax": 1.0, "cmap": "vlag"},
        "mae": {"vmin": 0, "vmax": 15, "cmap": "inferno_r"},
        "n_samples": {"vmin": 0, "vmax": 2000, "cmap": "viridis"},
    }

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(4.5 * 2, 6),
        sharey=True,
        gridspec_kw={"width_ratios": [1, 1, 0.05]},
    )
    for ax, metric in zip(axes, ["mae", "r2", "n_samples"]):
        p = all_metrics.pivot(index="organ", columns="model", values=metric)
        p = p.reindex(columns=model_names)
        sns.heatmap(
            p,
            annot=metric != "n_samples",
            fmt=".2f",
            ax=ax,
            cbar_kws={"shrink": 0.5, "location": "top", "label": metric},
            xticklabels=metric != "n_samples",
            yticklabels=True,
            **params[metric],
        )
    fig.savefig(results_dir / "ridgecv_imagenet_models.comparison.svg", **figkws)


def make_clock(model_name: str, tile_width: int = 224):
    # model_name: str = "alexnet"; tile_width: int = 224

    output_dir = results_dir / "clock"
    output_dir.mkdir(exist_ok=True, parents=True)
    suffix = f"{model_name}.{tile_width}px"

    f = features_dir / f"features_{suffix}.mean.pq"
    x = pd.read_parquet(f)
    a = AnnData(x, obs=meta.loc[x.index])

    cv = GroupKFold(5, shuffle=True, random_state=random_state)
    _metrics = list()
    for organ in sorted(a.obs["Organ"].unique()):
        if (output_dir / f"ridgecv_{suffix}.{organ}.csv").exists():
            continue
        _a = a[a.obs["Organ"] == organ]
        x = _a.to_df().groupby(level=0).mean()
        if "Age" not in a.obs.columns:
            y = _a.obs.join(meta[["Age"]]).reindex(x.index)["Age"]
        else:
            y = _a.obs.reindex(x.index)["Age"]

        # alphas = np.logspace(-2, 3, 20)
        # model = RidgeCV(alphas=alphas, fit_intercept=True)
        model = Ridge(alpha=2, fit_intercept=True)
        model = make_pipeline(StandardScaler(), model)
        y_pred = cross_val_predict(
            model, x, y, cv=cv, groups=_a.obs["Subject ID"], n_jobs=-1
        )
        r2 = r2_score(y, y_pred)
        mae = (y_pred - y).abs().mean()

        yrandom = y.copy()
        np.random.shuffle(yrandom.values)
        y_pred_random = cross_val_predict(
            model, x, yrandom, cv=cv, groups=_a.obs["Subject ID"], n_jobs=-1
        )
        r2_random = r2_score(y, y_pred_random)
        mae_random = (y_pred_random - y).abs().mean()

        pd.Series(y_pred, index=y.index).to_frame("pred").join(y).rename_axis(
            index=meta.index.name
        ).to_csv(output_dir / f"ridgecv_{suffix}.{organ}.predictions.csv")

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
        vmin, vmax = (
            pd.Series(y.tolist() + y_pred.tolist()).describe().loc[["min", "max"]]
        )
        ax.plot((20, 70), (vmin, vmax), linestyle="--", color="grey")
        ax.set(
            xlabel="Chronological age",
            ylabel="Biological age",
            title=f"MAE={mae:.2f}, R2={r2:.2f}",
        )
        fig.savefig(output_dir / f"ridgecv_{suffix}.{organ}.predictions.svg", **figkws)

        print(model_name, organ, x.shape[0], x.shape[1], r2, r2_random, mae, mae_random)
        _metrics.append(
            {
                "model": model_name,
                "organ": organ,
                "n_samples": x.shape[0],
                "n_features": x.shape[1],
                "r2": r2,
                "mae": mae,
                "r2_random": r2_random,
                "mae_random": mae_random,
            }
        )
    metrics = pd.DataFrame(_metrics)
    metrics.to_csv(results_dir / f"ridgecv_{suffix}.metrics.csv", index=False)


def collect(model_name: str):
    fs = sorted(features_dir.glob(f"features_{model_name}*.pq"))
    if fs:
        print(f"Features for {model_name} already collected.")
        return
    reduce_dims = False
    files = sorted(dirs[model_name].glob(f"*{model_name}.{tile_size}px*.mean.npy"))
    files = [f for f in files if "fine_tuned" not in f.name]
    # assert all([f.name.endswith('mean.npy') for f in files])
    if not files:
        files = sorted(dirs[model_name].glob(f"*{model_name}.{tile_size}px*.npy"))
        files = [f for f in files if "fine_tuned" not in f.name]
        reduce_dims = True
    if not files:
        print("Could not find files.")
        return

    _features = parmap.map(load_one, files, reduce_dims=reduce_dims, pm_pbar=True)
    x = pd.DataFrame(_features).dropna()
    name = x.index[:2].str.extract(rf".({model_name}.*)")[0][0]
    x.index = x.index.str.replace(rf".{model_name}.*", "", regex=True)
    print(f"Collected {x.shape} for {model_name}.")
    x.to_parquet(features_dir / f"features_{name}.mean.pq")


def load_one(f: Path, reduce_dims: bool = False):
    name = f.stem.replace(".mean", "")
    try:
        d = np.load(f)
        if reduce_dims:
            return pd.Series(d.mean(axis=0), name=name)
        return pd.Series(d, name=name)
    except EOFError:
        return pd.Series(name=name)


if __name__ == "__main__":
    main()
