#!/usr/bin/env uv --script

# /// script
# dependencies = [
#   "tqdm",
#   "joblib",
#   "parmap",
#   "numpy",
#   "pandas",
#   "anndata",
#   "matplotlib",
#   "seaborn",
#   "scikit-learn",
#   "lightgbm",
# ]
# [tool.uv.sources]
# conch = { git = "https://github.com/mahmoodlab/CONCH.git" }
# ///

"""
Cross-apply the clocks trained on GTEx to the other cohorts.

cd ~/projects/histopath
srun --qos interactiveq --partition interactiveq --mem 96000 -c 16 --x11 --pty -J IPython \
uv run --with ipython ipython

"""

from pathlib import Path

from tqdm.auto import tqdm
from joblib import dump, load
import parmap
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    RidgeCV,
    Lasso,
    LassoCV,
    ElasticNet,
    ElasticNetCV,
)
from sklearn.linear_model import HuberRegressor, QuantileRegressor
from sklearn.linear_model import BayesianRidge, ARDRegression
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import TweedieRegressor, GammaRegressor
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn.metrics import r2_score

from src.utils import get_restricted_info


metadata_dir = Path("metadata")
data_dir = Path("data") / "gtex" / "svs"
processed_dir = Path("processed") / "histopathology"
processed_dir.mkdir(exist_ok=True, parents=True)
results_dir = Path("results") / "tissue_clocks_revision"
results_dir.mkdir(exist_ok=True, parents=True)
figkws = dict(dpi=300, bbox_inches="tight")

meta_file = Path("data") / "gtex" / "GTEx Portal.csv"
meta = pd.read_csv(meta_file, index_col=0)
meta = meta.query(
    "Tissue.str.startswith('Skin') | (Tissue == 'Brain - Cortex') | Tissue.str.contains('Colon') | Tissue.str.startswith('Lung')"
)
rest, _ = get_restricted_info()
meta = meta.merge(
    rest[["Age", "Cohort"]], left_on="Subject ID", right_index=True, how="left"
)
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

model_names = ["virchow2"]

ml_types = [
    "linearmodel",
    "lassocv",
    "ridgecv",
    "elasticnetcv",
    "bayesianridge",
    "gamma",
    "tweedie",
    "svr",
    "linearsvr",
    # "ard",
    "huber",
    "quantile",
    "randomforest",
    "mlp",
    "l1mlp",
    "lgbm",
    # "lgbm_huber",
    # "lgbm_quantile",
]

cohorts = [
    "skinpath",
    "neuropath",
    "lungaging-schiller",
    "colonaging-fennell",
    "brain-healthy-histo",
    # "kidney-healthy-histo",
]


def main() -> None:
    for model_name in model_names:
        collect_features(model_name)
        for ml_type in ml_types:
            make_clock(model_name, ml_type=ml_type)
    for model_name in ["prism", "titan"]:
        collect_aggregated(model_name)
        make_clock(model_name)

    for ml_type in ml_types:
        for cohort in cohorts:
            apply_clock(cohort, ml_type=ml_type)
    summarize_performance()


def summarize_mltypes():
    _res = list()
    files = sorted((results_dir / "clock").glob("*.statistics.*.csv"))
    for f in files:
        _res.append(
            pd.read_csv(f).assign(
                Cohort=f.stem.split(".")[0], Regressor=f.stem.split(".")[2]
            )
        )
    res = pd.concat(_res)

    res.groupby(["Cohort", "Organ", "Regressor"]).agg(
        {"R2": "mean", "Pearson": "mean", "MAE": "mean"}
    ).unstack()["R2"]


def correlate_mltypes():
    mapping = {
        "cohort_skinpath": "Skin",
        "cohort_neuropath": "Brain",
        "cohort_lungaging-schiller": "Lung",
        "cohort_colonaging-fennell": "Colon",
        "cohort_brain-healthy-histo": "Brain",
        # "kidney-healthy-histo": "Kidney",
    }
    _res = list()
    _res2 = list()
    files = sorted((results_dir / "clock").glob("*.predictions.*.csv"))
    for f in files:
        d = pd.read_csv(f, index_col=0).assign(
            Cohort=f.stem.split(".")[0], Regressor=f.stem.split(".")[2]
        )
        cohort = d["Cohort"].iloc[0]
        if cohort in mapping:
            d = d.query(f"Organ == '{mapping[cohort]}'")
            _res.append(d)
    res = pd.concat(_res).reset_index(names="Sample")
    res["Error"] = res["Error"].abs()

    e = (
        res.pivot_table(
            index=["Cohort", "Organ", "Sample"], columns=["Regressor"], values="Error"
        )
        .groupby("Cohort")
        .mean()
    )
    c = res.pivot_table(
        index=["Cohort", "Organ", "Sample"], columns=["Regressor"], values="pred"
    ).corr()

    g = sns.clustermap(
        c,
        annot=True,
        annot_kws={"fontsize": 5},
        cmap="vlag",
        vmin=-1,
        vmax=1,
        col_colors=e.mean().map(lambda x: plt.cm.Reds_r(x)).rename("Error"),
        figsize=(4, 4),
    )
    g.fig.savefig(
        results_dir / "all_cohorts.predictions.correlation_between_ml_types.svg",
        **figkws,
    )

    fig, ax = plt.subplots(figsize=(4, 1))
    eo = e.mean().iloc[g.dendrogram_col.reordered_ind]
    sns.barplot(x=eo.index, y=eo.values, ax=ax, hue=eo.values, palette="Reds")
    for t in eo.index:
        ax.text(
            t,
            1,
            f"{eo[t]:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set(title="Mean absolute error of each regressor")
    fig.savefig(
        results_dir / "all_cohorts.predictions.mean_error_by_ml_type.barplot.svg",
        **figkws,
    )

    fig, ax = plt.subplots(figsize=(4, 1))
    eo = e.iloc[:, g.dendrogram_col.reordered_ind]
    sns.heatmap(eo, cmap="Reds")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center", va="top")
    ax.set(title="Mean absolute error of each regressor")
    fig.savefig(
        results_dir / "all_cohorts.predictions.mean_error_by_ml_type.heatmap.svg",
        **figkws,
    )


def summarize_performance(
    mpp: float = 0.5,
    tile_width: int = 224,
):
    # mpp: float = 0.5; tile_width: int = 224
    _res = list()
    for cohort_name in cohorts:
        f = (
            results_dir
            / "clock"
            / f"cohort_{cohort_name}.statistics.{mpp}mpp.{tile_width}px.all_models.csv"
        )
        _res.append(pd.read_csv(f).assign(Cohort=cohort_name, Regressor="Ridge"))
        f = (
            results_dir
            / "clock_lgbm"
            / f"cohort_{cohort_name}.statistics.{mpp}mpp.{tile_width}px.all_models.csv"
        )
        _res.append(pd.read_csv(f).assign(Cohort=cohort_name, Regressor="LGBM"))
    res = pd.concat(_res)

    # GTEx
    _res2 = list()
    for model_name in model_names + ["prism", "titan"]:
        suffix = f".{mpp}mpp.{tile_width}px.{model_name}"
        for organ in ["Brain", "Colon", "Lung", "Skin"]:
            f = results_dir / "clock" / f"ridgecv{suffix}.{organ}.csv"
            pred = pd.read_csv(f, index_col=0)
            pred["Error"] = pred["pred"] - pred["Age"]
            _res2.append(
                dict(
                    Model=model_name,
                    Organ=organ,
                    R2=r2_score(pred["Age"], pred["pred"]),
                    Pearson=np.corrcoef(pred["Age"], pred["pred"])[0, 1],
                    MAE=pred["Error"].abs().mean(),
                    Cohort="gtex",
                    Regressor="Ridge",
                )
            )
            f = results_dir / "clock_lgbm" / f"lgbm{suffix}.{organ}.csv"
            if not f.exists():
                continue
            pred = pd.read_csv(f, index_col=0)
            pred["Error"] = pred["pred"] - pred["Age"]
            _res2.append(
                dict(
                    Model=model_name,
                    Organ=organ,
                    R2=r2_score(pred["Age"], pred["pred"]),
                    Pearson=np.corrcoef(pred["Age"], pred["pred"])[0, 1],
                    MAE=pred["Error"].abs().mean(),
                    Cohort="gtex",
                    Regressor="LGBM",
                )
            )
    res2 = pd.DataFrame(_res2)

    # Combine
    dfj = pd.concat([res, res2], ignore_index=True)
    dfj.to_csv(
        results_dir / f"all_cohorts.statistics.{mpp}mpp.{tile_width}px.all_models.csv",
        index=False,
    )

    lims = dict(R2=(0, 1), Pearson=(-1, 1), MAE=(0, 30))
    cmaps = dict(R2="vlag", Pearson="vlag", MAE="Reds_r")
    fig, axes = plt.subplots(3, 1, figsize=(12, 3 * 4), sharex=True)
    for metric, ax in zip(["R2", "Pearson", "MAE"], axes):
        resp = dfj.query("Regressor == 'Ridge'").pivot_table(
            index=["Cohort", "Organ"], columns="Model", values=metric
        )
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

    # Make version without GTEx and transposed
    fig, axes = plt.subplots(3, 1, figsize=(12, 3 * 4), sharex=True)
    for metric, ax in zip(["R2", "Pearson", "MAE"], axes):
        resp = dfj.query("Regressor == 'Ridge'").pivot_table(
            index=["Cohort", "Organ"], columns="Model", values=metric
        )
        resp = resp.drop(index="gtex", level=0)[["uni2", "virchow2", "h-optimus-1"]]
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
        / f"validation_cohorts.predictions.{mpp}mpp.{tile_width}px.selected_models.heatmaps.svg",
        **figkws,
    )

    # Make version with only GTEx and transposed
    fig, axes = plt.subplots(3, 1, figsize=(1.2 * 3, 14), sharex=True)
    for metric, ax in zip(["MAE", "R2", "Pearson"], axes):
        resp = (
            dfj.query("Regressor == 'Ridge'")
            .query("Cohort == 'gtex'")
            .pivot_table(index="Model", columns="Organ", values=metric)
        )
        sns.heatmap(
            resp.reindex(resp.mean(axis=1).sort_values().index),
            ax=ax,
            vmin=lims[metric][0],
            vmax=lims[metric][1],
            annot=True,
            fmt=".2f",
            cmap=cmaps[metric],
            cbar_kws={"shrink": 0.5, "label": metric},
        )
        ax.set(title=metric)
    fig.tight_layout()
    fig.savefig(
        results_dir
        / f"gtex.predictions.{mpp}mpp.{tile_width}px.all_models.heatmaps.svg",
        **figkws,
    )

    # Make version with only GTEx and transposed, only for slide-level models
    fig, axes = plt.subplots(3, 1, figsize=(1.2 * 3, 3), sharex=True)
    for metric, ax in zip(["MAE", "R2", "Pearson"], axes):
        resp = (
            dfj.query("Regressor == 'Ridge'")
            .query("Cohort == 'gtex' & Model.isin(['titan', 'prism'])")
            .pivot_table(index="Model", columns="Organ", values=metric)
        )
        sns.heatmap(
            resp.reindex(resp.mean(axis=1).sort_values().index),
            ax=ax,
            vmin=lims[metric][0],
            vmax=lims[metric][1],
            annot=True,
            fmt=".2f",
            cmap=cmaps[metric],
            cbar_kws={"shrink": 1.5, "label": metric},
        )
        ax.set(title=metric)
    fig.tight_layout()
    fig.savefig(
        results_dir
        / f"gtex.predictions.{mpp}mpp.{tile_width}px.slide-level.heatmaps.svg",
        **figkws,
    )

    # Make version contrasting Ridge and LGBM for Virchow2 only
    fig, axes = plt.subplots(3, 1, figsize=(3 * 3, 1.2 * 3), sharex=True)
    for metric, ax in zip(["MAE", "R2", "Pearson"], axes):
        resp = res.query("Cohort != 'gtex' & Model.isin(['virchow2'])").pivot_table(
            index=["Regressor", "Model"], columns=["Cohort", "Organ"], values=metric
        )
        sns.heatmap(
            resp.reindex(resp.mean(axis=1).sort_values().index),
            ax=ax,
            vmin=lims[metric][0],
            vmax=lims[metric][1],
            annot=True,
            fmt=".2f",
            cmap=cmaps[metric],
            cbar_kws={"shrink": 1.5, "label": metric},
        )
        ax.set(title=metric)
    fig.tight_layout()
    fig.savefig(
        results_dir
        / f"all_cohorts.predictions.{mpp}mpp.{tile_width}px.all_models.ridge_vs_lgbm.heatmaps.svg",
        **figkws,
    )


def apply_clock(
    cohort_name: str,
    mpp: float = 0.5,
    tile_width: int = 224,
    ml_type: str = "ridgecv",
):
    # model_name: str = "virchow2"; mpp: float = 0.5; tile_width: int = 224

    output_dir = results_dir / "clock"
    output_dir.mkdir(exist_ok=True, parents=True)

    h5ads = list(
        (results_dir / "other_cohort_data" / cohort_name).glob(
            f"anndata.{mpp}mpp.{tile_width}px.*.h5ad"
        )
    )
    ms = [f.stem.split(".")[-1] for f in h5ads]
    _model_names = [m for m in model_names if m in ms]

    organs = ["Brain", "Colon", "Lung", "Skin"]

    _res = list()
    _preds = list()
    m, n = len(_model_names), 4
    fig, axes = plt.subplots(
        nrows=n,
        ncols=m,
        figsize=(4 * m, 4 * n),
        sharex="col",
        sharey="col",
        squeeze=False,
    )
    for model_name, axs in tqdm(zip(_model_names, axes.T)):
        suffix = f"{mpp}mpp.{tile_width}px.{model_name}"
        h = results_dir / "other_cohort_data" / cohort_name / f"anndata.{suffix}.h5ad"
        b = ad.read_h5ad(h)
        if "Age at sampling" not in b.obs.columns:
            if "age" in b.obs.columns:
                b.obs["Age at sampling"] = b.obs["age"]
            elif "Age" in b.obs.columns:
                b.obs["Age at sampling"] = b.obs["Age"]
        b = b[~b.obs["Age at sampling"].isnull()]
        b = b[b.obs["Age at sampling"] > 0]
        if "Tissue_Source" in b.obs.columns:
            b = b[b.obs["Tissue_Source"].isin(["Colon"])]
        x = b.to_df().groupby(level=0).mean()
        y = b.obs["Age at sampling"].groupby(level=0).mean().reindex(x.index)

        for organ, ax in zip(organs, axs):
            final_model = load(
                output_dir / f"{ml_type}-final_model.{suffix}.{organ}.joblib"
            )
            # scaler = load(output_dir / f"{ml_type}-scaler.{suffix}.{organ}.joblib")
            scaler = StandardScaler()
            calibrator = load(
                output_dir / f"{ml_type}-calibrator.{suffix}.{organ}.joblib"
            )
            x_scaled = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)
            pred = pd.Series(
                calibrator.predict(final_model.predict(x_scaled).reshape(-1, 1)),
                index=x.index,
                name="pred",
            ).to_frame()
            pred["Age"] = y
            pred["Error"] = pred["pred"] - pred["Age"]
            _preds.append(
                pred.assign(Model=model_name, Organ=organ, Cohort=cohort_name)
            )
            _res.append(
                dict(
                    Model=model_name,
                    Organ=organ,
                    N=pred.shape[0],
                    R2=r2_score(pred["Age"], pred["pred"]),
                    Pearson=np.corrcoef(pred["Age"], pred["pred"])[0, 1],
                    MAE=pred["Error"].abs().mean(),
                )
            )
            d = pred["pred"] - pred["Age"]
            print(d.abs().mean())

            sns.regplot(
                x=pred["Age"],
                y=pred["pred"],
                scatter=False,
                ax=ax,
                color="grey",
            )
            ax.scatter(
                pred["Age"],
                pred["pred"],
                alpha=0.5,
                c=d,
                cmap="coolwarm",
                vmin=-abs(d).max(),
                vmax=abs(d).max(),
                rasterized=True,
            )
            ax.plot(
                (pred["Age"].min(), pred["Age"].max()),
                (pred["Age"].min(), pred["Age"].max()),
                label="y=x",
                linestyle="--",
                color="grey",
            )
            ax.text(
                0.05,
                0.95,
                f"R2={_res[-1]['R2']:.2f}\nr={_res[-1]['Pearson']:.2f}\nMAE={_res[-1]['MAE']:.2f}",
                transform=ax.transAxes,
                va="top",
            )
            ax.set(
                xlabel="Chronological age",
                ylabel="Biological age",
                title=f"{model_name} {organ}",
            )

    preds = pd.concat(_preds)
    preds.to_csv(
        output_dir
        / f"cohort_{cohort_name}.predictions.{ml_type}.{suffix}.all_models.csv",
        index=True,
    )

    res = pd.DataFrame(_res)
    res.to_csv(
        output_dir
        / f"cohort_{cohort_name}.statistics.{ml_type}.{suffix}.all_models.csv",
        index=False,
    )

    fig.tight_layout()
    fig.savefig(
        output_dir
        / f"cohort_{cohort_name}.predictions.{ml_type}.{suffix}.all_models.svg",
        **figkws,
    )


def make_clock(
    model_name: str = "virchow2",
    mpp: float = 0.5,
    tile_width: int = 224,
    ml_type: str = "ridgecv",
):
    # model_name: str = "virchow2"; mpp: float = 0.5; tile_width: int = 224

    output_dir = results_dir / "clock"
    output_dir.mkdir(exist_ok=True, parents=True)

    suffix = f"{mpp}mpp.{tile_width}px.{model_name}"
    a = ad.read_h5ad(results_dir / f"gtex.4_tissues.{suffix}.h5ad")
    a.obs["Organ"] = a.obs["Tissue"].str.split(" - ").str[0]

    cv = GroupKFold(5)

    for organ in sorted(a.obs["Organ"].unique()):
        if (output_dir / f"{ml_type}-calibrator.{suffix}.{organ}.joblib").exists():
            continue
        _a = a[a.obs["Organ"] == organ]
        x = _a.to_df().groupby(level=0).mean()
        if "Age" not in a.obs.columns:
            y = _a.obs.join(meta[["Age"]]).reindex(x.index)["Age"]
        else:
            y = _a.obs.reindex(x.index)["Age"]

        alphas = np.logspace(-2, 3, 20)
        if ml_type == "linearmodel":
            predictor = LinearRegression(fit_intercept=True)
        elif ml_type == "ridge":
            predictor = Ridge(alpha=2.0, fit_intercept=True)
        elif ml_type == "ridgecv":
            predictor = RidgeCV(alphas=alphas, fit_intercept=True)
        elif ml_type == "lasso":
            predictor = Lasso(alpha=2.0, fit_intercept=True)
        elif ml_type == "lassocv":
            base_model = LassoCV(alphas=alphas, fit_intercept=True, max_iter=5000)
            predictor = TransformedTargetRegressor(
                regressor=base_model,
                transformer=StandardScaler(),
            )
        elif ml_type == "elasticnetcv":
            base_model = ElasticNetCV(
                l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99],
                alphas=100,
                cv=5,
                max_iter=50000,
                tol=1e-3,  # Slightly looser tolerance
                selection="random",
                random_state=0,
            )
            predictor = TransformedTargetRegressor(
                regressor=base_model,
                transformer=StandardScaler(),
            )
        elif ml_type == "huber":
            predictor = HuberRegressor(epsilon=1.35, alpha=0.01, max_iter=5000)
        elif ml_type == "quantile":
            predictor = QuantileRegressor(quantile=0.5, alpha=0.01, solver="highs")
        elif ml_type == "bayesianridge":
            predictor = BayesianRidge(
                max_iter=5000,
                tol=1e-4,
                alpha_1=1e-6,
                alpha_2=1e-6,
                lambda_1=1e-6,
                lambda_2=1e-6,
            )
        elif ml_type == "ard":
            predictor = ARDRegression(max_iter=5000, tol=1e-4)
        elif ml_type == "randomforest":
            predictor = RandomForestRegressor(
                n_estimators=250, random_state=0, n_jobs=-1
            )
        elif ml_type == "lgbm":
            predictor = lgb.LGBMRegressor(
                objective="regression",
                metric="mae",
                boosting_type="gbdt",
                num_leaves=15,  # Smaller trees
                max_depth=5,  # Limit depth
                learning_rate=0.02,  # Slower learning
                feature_fraction=0.5,  # More aggressive subsampling
                bagging_fraction=0.7,
                bagging_freq=5,
                n_estimators=1000,
                reg_alpha=0.1,  # L1 on weights
                reg_lambda=1.0,  # L2 on weights
                min_child_samples=20,  # Require more samples per leaf
                verbose=-1,
                random_state=0,
            )
        elif ml_type == "lgbm_huber":
            predictor = lgb.LGBMRegressor(
                objective="huber",
                metric="mae",
                num_leaves=31,
                learning_rate=0.05,
                n_estimators=500,
                verbose=-1,
                random_state=0,
            )
        elif ml_type == "lgbm_quantile":
            predictor = lgb.LGBMRegressor(
                objective="quantile",
                alpha=0.5,  # Median
                metric="mae",
                num_leaves=31,
                learning_rate=0.05,
                n_estimators=500,
                verbose=-1,
                random_state=0,
            )
        elif ml_type == "svr":
            predictor = SVR(kernel="rbf", C=1.0, epsilon=0.1, max_iter=5000)
        elif ml_type == "linearsvr":
            predictor = LinearSVR(C=1.0, epsilon=0.1, max_iter=5000)
        elif ml_type == "tweedie":
            predictor = TweedieRegressor(
                power=1.5,  # 1=Poisson, 2=Gamma, between=compound
                alpha=0.1,
                max_iter=1000,
            )
        elif ml_type == "gamma":
            predictor = GammaRegressor(alpha=0.1, max_iter=1000)
        elif ml_type == "mlp":
            predictor = MLPRegressor(
                hidden_layer_sizes=(256, 128, 64),
                activation="relu",
                solver="adam",
                max_iter=2000,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                learning_rate="adaptive",
                learning_rate_init=0.001,
                alpha=0.01,
                batch_size=32,
                random_state=0,
                warm_start=False,
            )
        elif ml_type == "l1mlp":
            predictor = L1MLPRegressor(hidden_dims=(256, 128), lambda_l1=1.0)

        model = make_pipeline(StandardScaler(), predictor)
        y_pred = cross_val_predict(
            model, x, y, cv=cv, groups=_a.obs["Subject ID"], n_jobs=-1
        )
        r2 = r2_score(y, y_pred)
        mae = (y_pred - y).abs().mean()
        pd.Series(y_pred, index=y.index).to_frame("pred").join(y).to_csv(
            output_dir / f"{ml_type}-cv_preds.{suffix}.{organ}.csv"
        )

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
        ax.plot((vmin, vmax), (vmin, vmax), linestyle="--", color="grey")
        ax.set(
            xlabel="Chronological age",
            ylabel="Biological age",
            title=f"MAE={mae:.2f}, R2={r2:.2f}",
        )
        fig.savefig(output_dir / f"{ml_type}.{suffix}.{organ}.svg", **figkws)

        # Make final model
        scaler = fit_scaler(x)
        xt = pd.DataFrame(scaler.transform(x), index=x.index, columns=x.columns)
        if isinstance(predictor, RidgeCV):
            predictor.fit(xt, y)
            predictor = Ridge(alpha=predictor.alpha_, fit_intercept=True)
        if isinstance(predictor, LassoCV):
            predictor.fit(xt, y)
            predictor = Lasso(alpha=predictor.alpha_, fit_intercept=True)
        elif isinstance(predictor, ElasticNetCV):
            predictor.fit(xt, y)
            predictor = ElasticNet(
                alpha=predictor.alpha_,
                l1_ratio=predictor.l1_ratio,
                fit_intercept=True,
                max_iter=5000,
            )
        predictor.fit(xt, y)
        y_pred = pd.Series(predictor.predict(xt), index=xt.index, name="pred")
        calibrator = fit_calibrator(y_pred.values, y)
        y_pred_calib = pd.Series(
            calibrator.predict(y_pred.values.reshape(-1, 1)).flatten(), xt.index
        )
        y_pred_calib.to_csv(output_dir / f"{ml_type}-final_preds.{suffix}.{organ}.csv")
        if isinstance(
            predictor,
            (
                LinearRegression,
                Ridge,
                RidgeCV,
                Lasso,
                LassoCV,
                ElasticNet,
                ElasticNetCV,
                HuberRegressor,
                QuantileRegressor,
                BayesianRidge,
                ARDRegression,
            ),
        ):
            coef = pd.Series(predictor.coef_, index=x.columns)
            coef["intercept"] = predictor.intercept_
        elif isinstance(predictor, (RandomForestRegressor, lgb.LGBMRegressor)):
            coef = pd.Series(predictor.feature_importances_, index=x.columns)
        else:
            coef = pd.Series(index=x.columns, data=np.nan)
        coef.to_csv(output_dir / f"{ml_type}-final_coefs.{suffix}.{organ}.csv")
        dump(scaler, output_dir / f"{ml_type}-scaler.{suffix}.{organ}.joblib")
        dump(predictor, output_dir / f"{ml_type}-final_model.{suffix}.{organ}.joblib")
        dump(calibrator, output_dir / f"{ml_type}-calibrator.{suffix}.{organ}.joblib")

        print(model_name, organ, x.shape[0], x.shape[1], r2, mae)


def fit_scaler(x):
    scaler = StandardScaler()
    scaler.fit(x)
    return scaler


def fit_calibrator(y_pred, y_true):
    calibrator = LinearRegression()
    calibrator.fit(y_pred.reshape(-1, 1), y_true)
    return calibrator


def collect_features(
    model_name: str = "virchow2", mpp: float = 0.5, tile_width: int = 224
):
    if processed_dir.glob(f"*{model_name}*.h5ad"):
        collect_features_h5ad(model_name, mpp, tile_width)
    elif processed_dir.glob(f"*.zarr/tables/{model_name}_tiles/"):
        collect_features_zarr(model_name, mpp, tile_width)
    else:
        raise ValueError(f"No files found for model {model_name}")


def collect_features_h5ad(
    model_name: str = "virchow2", mpp: float = 0.5, tile_width: int = 224
):
    # model_name: str = "virchow2"; mpp: float = 0.5; tile_width: int = 224
    suffix = f".{mpp}mpp.{tile_width}px.{model_name}"
    output_file = results_dir / f"gtex.4_tissues.{suffix}.h5ad"
    if output_file.exists():
        return
    files = sorted(processed_dir.glob(f"*{suffix}.h5ad"))
    _ds = dict()
    for f in tqdm(files):
        print(f)
        d = ad.read_h5ad(f).to_df().mean().rename(f.stem.replace(suffix, ""))
        _ds[d.name] = d.values.tolist()
    df = pd.DataFrame(_ds).T
    obs = meta.reindex(df.index).dropna(subset="Tissue")
    a = ad.AnnData(df, obs=obs)
    a.write(output_file)


def collect_features_zarr(
    model_name: str = "virchow2", mpp: float = 0.5, tile_width: int = 224
):
    # model_name: str = "virchow2"; mpp: float = 0.5; tile_width: int = 224
    import anndata as ad

    suffix = f".{mpp}mpp.{tile_width}px.{model_name}"
    output_file = results_dir / f"gtex.4_tissues.{suffix}.h5ad"
    if output_file.exists():
        return
    files = sorted(processed_dir.glob(f"*.zarr/tables/{model_name}_tiles/"))
    _ds = dict()
    for f in tqdm(files):
        print(f)
        name = f.parent.parent.stem
        d = ad.read_zarr(f).to_df().mean().rename(name)
        _ds[d.name] = d.values.tolist()
    df = pd.DataFrame(_ds).T
    obs = meta.reindex(df.index).dropna(subset="Tissue")
    a = ad.AnnData(df, obs=obs)
    a.write(output_file)


def collect_aggregated(
    model_name: str = "prism", mpp: float = 0.5, tile_width: int = 224
):
    """
    Output from process_lz_aggregate.py
    """
    # model_name: str = "prism"; mpp: float = 0.5; tile_width: int = 224
    suffix = f".{mpp}mpp.{tile_width}px.{model_name}"
    output_file = results_dir / f"gtex.4_tissues.{suffix}.h5ad"

    def read(f: Path):
        return pd.Series(np.load(f).squeeze(), name=f.stem.split(".")[0])

    npys = sorted(
        processed_dir.glob(f"*.{mpp}mpp.{tile_width}px.{model_name}_aggregated.npy")
    )
    _data = parmap.map(read, npys, pm_pbar=True, pm_processes=8)
    df = pd.DataFrame(_data)
    obs = meta.reindex(df.index).dropna(subset="Tissue")
    a = ad.AnnData(df, obs=obs)
    a.write(output_file)


# def check_features():
#     model_name: str = "virchow2"
#     mpp: float = 0.5
#     tile_width: int = 224

#     _h5ads = list()
#     for cohort_name in cohorts:
#         suffix = f"{mpp}mpp.{tile_width}px.{model_name}"
#         h = results_dir / "other_cohort_data" / cohort_name / f"anndata.{suffix}.h5ad"
#         b = ad.read_h5ad(h)
#         if "Age at sampling" not in b.obs.columns:
#             if "age" in b.obs.columns:
#                 b.obs["Age at sampling"] = b.obs["age"]
#             elif "Age" in b.obs.columns:
#                 b.obs["Age at sampling"] = b.obs["Age"]
#         b = b[~b.obs["Age at sampling"].isnull()]
#         b.obs["Cohort"] = cohort_name
#         _h5ads.append(b)
#     a = ad.concat(_h5ads)

#     sc.pp.pca(a)
#     sc.pp.neighbors(a)
#     sc.tl.umap(a)

#     for emb in a.obsm:
#         fig = sc.pl.embedding(a, basis=emb, color=["Cohort", "Age at sampling"])
#         fig.savefig(
#             results_dir / f"all_cohorts.{model_name}.{emb}.png",
#             **figkws,
#         )

#     cm = a.to_df().groupby(a.obs["Cohort"]).mean()
#     cv = a.to_df().groupby(a.obs["Cohort"]).var()


class L1MLPRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        hidden_dims=(256, 128),
        lambda_l1=0.01,
        lr=0.001,
        max_iter=1000,
        patience=20,
    ):
        self.hidden_dims = hidden_dims
        self.lambda_l1 = lambda_l1
        self.lr = lr
        self.max_iter = max_iter
        self.patience = patience

    def fit(self, X, y):
        import torch
        import torch.nn as nn

        X_t = torch.FloatTensor(X.values if hasattr(X, "values") else X)
        y_t = torch.FloatTensor(y.values if hasattr(y, "values") else y)

        # Build network
        layers = []
        prev_dim = X_t.shape[1]
        for h in self.hidden_dims:
            layers.extend([nn.Linear(prev_dim, h), nn.ReLU()])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.model_ = nn.Sequential(*layers)

        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        best_loss, patience_counter = float("inf"), 0

        for epoch in range(self.max_iter):
            optimizer.zero_grad()
            pred = self.model_(X_t).squeeze()
            mse = nn.functional.mse_loss(pred, y_t)
            l1 = sum(p.abs().sum() for p in self.model_.parameters())
            loss = mse + self.lambda_l1 * l1
            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break
        return self

    def predict(self, X):
        import torch

        X_t = torch.FloatTensor(X.values if hasattr(X, "values") else X)
        with torch.no_grad():
            return self.model_(X_t).squeeze().numpy()


if __name__ == "__main__":
    main()
