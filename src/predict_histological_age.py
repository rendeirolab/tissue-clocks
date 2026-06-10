"""
Use latent representations of tissue images to study aging.
"""

from pathlib import Path
import typing as tp
from timeit import default_timer as timer

from tqdm import tqdm
import numpy as np
import pandas as pd
from anndata import AnnData
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn_extensions import clustermap, swarmboxenplot
import pingouin as pg
from sklearn.linear_model import (
    LinearRegression,
    Lasso,
    LassoCV,
    Ridge,
    RidgeCV,
    ElasticNet,
    ElasticNetCV,
    BayesianRidge,
)
from sklearn.ensemble import RandomForestRegressor  # from xgboost import XGBRegressor
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

# from sklearnex import patch_sklearn; patch_sklearn()  # only for Intel CPUs

from src import config
from src.utils import (
    get_restricted_info,
    get_engineered_info,
    get_telomere_lengths,
    get_somatic_mutation_counts,
)


def main():
    feature_model_name = "fine_tuned"
    # feature_model_name = "uni_features"
    a = sc.read_h5ad(config.results_dir / feature_model_name / "anndata.h5ad")
    a.obs.index.name = "Tissue Sample ID"
    var, var_annot = get_restricted_info()
    a.obs = (
        a.obs.reset_index()
        .merge(
            var[["Age", "Ischemic Time (Minutes)", "Cohort"]],
            left_on="Subject ID",
            right_index=True,
        )
        .set_index(a.obs.index.name)
    )

    predict_age(a, feature_model_name=feature_model_name)
    inspect_performance(feature_model_name=feature_model_name)
    models = [
        "LinearRegression",
        "Lasso",
        "LassoCV",
        "Ridge",
        "RidgeCV",
        "BayesianRidge",
        "ElasticNet",
        "ElasticNetCV",
        "RandomForestRegressor",
    ]
    cvs = ["GroupKFold", "StratifiedGroupKFold"]
    # exclude tissues with low number of samples - bad performance
    exclude_entities = [
        "Bladder",
        "Cervix - Ectocervix",
        "Cervix - Endocervix",
        "Fallopian Tube",
        "Kidney - Medulla",
    ]

    for cv in cvs:
        for model in models:
            inspect_model_output(
                model_name=model,
                feature_model_name=feature_model_name,
                cv_name=cv,
            )
            cross_prediction(
                a,
                feature_model_name=feature_model_name,
                model_name=model,
                cv_name=cv,
                exclude_entities=exclude_entities,
            )

    aging_at_individual_level()


def get_age_influence(a, feature_model_name: str = "fine_tuned"):
    # determine % variance explained by age in each tissue
    import statsmodels.api as sm

    covariates = ["Sex", "Ischemic Time (Minutes)"]
    target = "Age"

    _coefs = list()
    _stats = list()
    for tissue in tqdm(sorted(a.obs["Tissue"].unique())):
        b = a[a.obs.query("Tissue == @tissue").index, :]
        sc.pp.scale(b)
        sc.pp.pca(b)

        y = b.obs[target]
        x = pd.DataFrame(b.obsm["X_pca"], index=b.obs.index)
        x.columns = "PCA_" + x.columns.astype(str)
        dummies = pd.get_dummies(b.obs[covariates]).astype(float)
        x = x.join(dummies)
        x = x.loc[:, x.var().gt(0)]
        x = sm.add_constant(x)

        model = sm.OLS(y, x).fit()
        c = model.summary2().tables[1]
        c["variance_ratio"] = (
            [np.nan]
            + b.uns["pca"]["variance_ratio"].tolist()
            + [np.nan] * dummies.var().gt(0).sum()
        )
        _coefs.append(c.assign(Tissue=tissue))
        _stats.append(
            pd.DataFrame(model.summary2().tables[0].values.reshape(-1, 2)).assign(
                Tissue=tissue
            )
        )
    coefs = pd.concat(_coefs)
    stats = pd.concat(_stats)
    stats.replace("nan", np.nan, inplace=True)

    r = (
        stats.loc[stats[0] == "Adj. R-squared:"]
        .set_index("Tissue")[1]
        .astype(float)
        .sort_values(ascending=False)
        .dropna()
    )
    p = (
        stats.loc[stats[0] == "Prob (F-statistic):"]
        .set_index("Tissue")[1]
        .astype(float)
        .reindex(r.index)
        .dropna()
    )
    p = -np.log10(p)
    n = (
        stats.loc[stats[0] == "No. Observations:"]
        .set_index("Tissue")[1]
        .astype(float)
        .reindex(r.index)
        .dropna()
    )

    fig, axes = plt.subplots(3, 1, figsize=(6, 3 * 3), sharex=True)
    r.plot.bar(ax=axes[0])
    axes[0].axhline(r.mean(), ls="--", c="k")
    axes[0].set(ylabel="Adjusted R-squared", ylim=(0, 1))
    p.plot.bar(ax=axes[1])
    axes[1].axhline(p.mean(), ls="--", c="k")
    axes[1].set(ylabel="-log10(p-value)")
    n.plot.bar(ax=axes[2])
    axes[2].axhline(n.mean(), ls="--", c="k")
    axes[2].set(ylabel="No. observations")
    fig.tight_layout()
    fig.savefig(
        config.results_dir / feature_model_name / "age_influence.svg",
        **config.figkws,
    )

    fig, axes = plt.subplots(3, 1, figsize=(6, 3 * 3), sharey=True)
    sns.barplot(x=r, y=r.index, ax=axes[0], orient="horiz")
    axes[0].axvline(r.mean(), ls="--", c="k")
    axes[0].set(xlabel="Adjusted R-squared", xlim=(0, 1))
    sns.barplot(x=p, y=p.index, ax=axes[1], orient="horiz")
    axes[1].axvline(p.mean(), ls="--", c="k")
    axes[1].set(xlabel="-log10(p-value)")
    sns.barplot(x=n, y=n.index, ax=axes[2], orient="horiz")
    axes[2].axvline(n.mean(), ls="--", c="k")
    axes[2].set(xlabel="No. observations")
    fig.tight_layout()
    fig.savefig(
        config.results_dir / feature_model_name / "age_influence.horiz.svg",
        **config.figkws,
    )


def regression_metric(true, pred):
    from sklearn.linear_model import LinearRegression

    fit = LinearRegression(fit_intercept=True).fit(true.values.reshape(-1, 1), pred)
    return fit.coef_[0], fit.intercept_


def get_metrics(train, valid, time, train_y, train_pred, valid_y, valid_pred):
    from sklearn.metrics import r2_score, mean_absolute_error, explained_variance_score
    from pingouin import corr

    tr2 = r2_score(train_y, train_pred)
    tmae = mean_absolute_error(train_y, train_pred)
    tevs = explained_variance_score(train_y, train_pred)
    tpearson = corr(train_y, train_pred).loc["pearson", "r"]
    tm, tb = regression_metric(train_y, train_pred)

    r2 = r2_score(valid_y, valid_pred)
    mae = mean_absolute_error(valid_y, valid_pred)
    evs = explained_variance_score(train_y, train_pred)
    pearson = corr(train_y, train_pred).loc["pearson", "r"]
    m, b = regression_metric(valid_y, valid_pred)

    metric = (
        [len(train), len(valid), time, train_y.mean(), valid_y.mean()]
        + [tr2, tmae, tevs, tpearson, tm, tb]
        + [r2, mae, evs, pearson, m, b]
    )
    metric_names = (
        ["n_train", "n_valid", "time", "mean_y_train", "mean_y_valid"]
        + ["train_r_squared", "train_mean_absolute_error", "train_explained_variance"]
        + ["train_pearson", "train_coefficient", "train_intercept"]
        + ["r_squared", "mean_absolute_error", "explained_variance"]
        + ["pearson", "coefficient", "intercept"]
    )
    return pd.Series(metric, index=metric_names)


def cross_validated_regression(
    model: tp.Any,
    X: pd.DataFrame,
    y: pd.Series,
    splitter: tp.Any,
    groups: pd.Series,
    *,
    shuffle: bool = False,
    preprocessing: bool = False,
    # target_transform: str = "no_transformation",
    # hyperparameter_tuner: tp.Any = None,
):
    from sklearn import set_config

    set_config(transform_output="pandas")
    # tt = TargetTransformer(target_transform)

    def fold(train, valid, model):
        xt = X.iloc[train]
        yt = y.iloc[train]
        xv = X.iloc[valid]
        yv = y.iloc[valid]

        if preprocessing:
            xscaler = StandardScaler()
            xt = xscaler.fit_transform(xt)
            xv = xscaler.transform(xv)
            # if target_transform == "no_transformation":
            yscaler = StandardScaler()
            yt = yscaler.fit_transform(yt.to_frame()).squeeze(axis=1)
            yv = yscaler.transform(yv.to_frame()).squeeze(axis=1)

        # yt = tt.transform(yt)
        # yv = tt.transform(yv)

        # Fit training data
        start = timer()

        # # Tune hyperparameters
        # if hyperparameter_tuner is not None:
        #     model = hyperparameter_tuner(xt, yt, xv, yv)
        model.fit(xt, yt)

        # Predict training data
        tpred = pd.Series(model.predict(xt), xt.index)

        # Predict validation data
        vpred = pd.Series(model.predict(xv), xv.index)
        time = timer() - start

        if preprocessing:  #  and (target_transform == "no_transformation")
            yt = yscaler.inverse_transform(yt.to_frame()).squeeze(axis=1)
            yv = yscaler.inverse_transform(yv.to_frame()).squeeze(axis=1)
            # tmp fix (https://github.com/scikit-learn/scikit-learn/issues/25592):
            yt = pd.Series(yt, index=xt.index)
            yv = pd.Series(yv, index=xv.index)
            tpred = yscaler.inverse_transform(tpred.to_frame()).squeeze(axis=1)
            vpred = yscaler.inverse_transform(vpred.to_frame()).squeeze(axis=1)
            tpred = pd.Series(tpred, index=xt.index)
            vpred = pd.Series(vpred, index=xv.index)

        # yt = tt.inverse_transform(yt)
        # yv = tt.inverse_transform(yv)

        # Calculate residuals
        res = vpred - yv

        # Get metrics
        metric = get_metrics(train, valid, time, yt, tpred, yv, vpred)

        # Get coefficients
        coef = np.nan
        if hasattr(model, "coef_"):
            coef = model.coef_
        elif hasattr(model, "feature_importances_"):
            coef = model.feature_importances_
        if hasattr(model, "intercept_"):
            coef = coef.tolist() + [model.intercept_]
        extra = ["fit_intercept"] if hasattr(model, "intercept_") else []
        return (vpred, res, metric, pd.Series(coef, index=X.columns.tolist() + extra))

    if shuffle:
        # Make sure not to overwrite values if `y` is a reference to an existing dataframe
        y = y.copy()
        y[:] = y.sample(frac=1).values

    # To use a stratified splitter for regression we bin Y just for splitting
    if splitter.__str__().split("(")[0] == "StratifiedGroupKFold":
        _y = pd.cut(y, bins=list(np.arange(0, 100, 15))).cat.codes
    elif splitter.__str__().split("(")[0] == "TimeSeriesSplit":
        _y = y.sort_values()
        X = X.reindex(_y.index)
    else:
        _y = y

    _preds = list()
    _ress = list()
    _metrics = list()
    _coefs = list()
    for train, valid in tqdm(
        splitter.split(X, _y, groups),
        total=splitter.get_n_splits(),
        leave=False,
        position=1,
        disable=splitter.get_n_splits() == 1,
    ):
        # print(len(train), _y[train].mean(), len(valid), _y[valid].mean())  # to check TimeSeriesSplit
        pred, res, metric, coef = fold(train, valid, model)
        _preds.append(pred)
        _ress.append(res)
        _metrics.append(metric)
        _coefs.append(coef)

    preds = pd.concat(_preds).to_frame("prediction")
    res = pd.concat(_ress).to_frame("residuals")
    preds_res = preds.join(res).assign(shuffled=shuffle)
    coefs = pd.DataFrame(_coefs).assign(shuffled=shuffle)
    metrics = pd.DataFrame(_metrics).assign(shuffled=shuffle)
    return preds_res, metrics, coefs


def counter_regression_to_mean(
    res: pd.DataFrame, model, splitter, groups
) -> tuple[pd.DataFrame, pd.Series]:
    res["Age"] = res["prediction"] - res["residuals"]
    X = res["Age"].to_frame()
    y = res["residuals"]
    args = [model, X, y, splitter, groups]
    a_preds_res, a_metrics, a_coefs = cross_validated_regression(*args)
    res["residuals_adj"] = res["residuals"] - a_preds_res["prediction"]
    res["prediction_adj"] = res["Age"] + res["residuals_adj"]

    metrics_adj = get_metrics(
        train=y,
        valid=y,
        time=None,
        train_y=res["Age"],
        train_pred=res["prediction_adj"],
        valid_y=res["Age"],
        valid_pred=res["prediction_adj"],
    )
    metrics_adj = metrics_adj.loc[
        [
            "r_squared",
            "mean_absolute_error",
            "explained_variance",
            "pearson",
            "coefficient",
            "intercept",
        ]
    ]
    metrics_adj.index += "_adj"
    metrics_adj["regression_to_mean_Age_coef"] = a_coefs["Age"].mean()
    if "fit_intercept" in a_coefs.index:
        metrics_adj["regression_to_mean_Age_fit_intercept"] = a_coefs[
            "fit_intercept"
        ].mean()
    return res, metrics_adj


def predict_age(
    a: AnnData,
    feature_model_name: str = "fine_tuned",
    feature_space: str = "X",
    frac: float = 1.0,
    target_variables: list[str] = ["Age"],
    covariates: list[str] = ["Sex", "Ischemic Time (Minutes)", "Cohort"],
    grouping_variable: str = "Subject ID",
    per: str = "Tissue",
    output_dir: Path | None = None,
):
    assert len(target_variables) == 1

    if output_dir is None:
        output_dir = (
            config.results_dir
            / feature_model_name
            / (
                f"age_{feature_space}"
                + (f"_frac{frac}" if feature_space == "X" else "")
            )
        )
        output_dir.mkdir(exist_ok=True, parents=True)

    entities = a.obs[per].cat.categories

    models = [
        LinearRegression(fit_intercept=True, n_jobs=-1),
        Lasso(2, fit_intercept=True),
        LassoCV(alphas=np.logspace(-2, 6, 20), fit_intercept=True, cv=5),
        Ridge(2, fit_intercept=True),
        RidgeCV(alphas=np.logspace(-2, 6, 20), fit_intercept=True, cv=5),
        BayesianRidge(),
        ElasticNet(2, l1_ratio=0.5, selection="random", fit_intercept=True),
        RandomForestRegressor(n_jobs=-1),
        ElasticNetCV(
            l1_ratio=0.5,
            alphas=np.logspace(-2, 6, 20),
            cv=5,
            selection="random",
            fit_intercept=True,
        ),
        # XGBRegressor(n_jobs=-1),
        # ARDRegression(fit_intercept=True),
        # SGDRegressor(),
    ]
    splitters = [
        GroupKFold(5),
        StratifiedGroupKFold(5),
        # BalancedStratifiedGroupKFold(bins=np.arange(20, 85, 15)),
        # BalancedStratifiedGroupKFold(bins=np.arange(0, 100, 10)),
        # BalancedStratifiedGroupKFold(bins=np.arange(0, 100, 5)),
        # TimeSeriesSplit(5),
    ]

    # For speed, pandas is faster than AnnData
    obs = a.obs.copy()
    a = a.raw[:, a.var.index].to_adata().copy().to_df()

    # # Tissue-specific clocks
    for model in models:
        model_name = model.__str__().split("(")[0]
        for splitter in splitters:
            cv_name = splitter.__str__().split("(")[0]
            output_prefix = (
                output_dir / f"tissue-specific_clocks.{model_name}.{cv_name}.SUFFIX"
            )
            if output_prefix.with_suffix(".coefficients.pq").exists():
                continue
            _preds_reds = list()
            _metrics = list()
            _coefs = list()
            _tqdm0 = tqdm(entities, position=0, leave=True)
            for entity in _tqdm0:
                sel = obs[per] == entity
                _tqdm0.set_description(f"{entity}, n = {sel.sum()}")

                # N.B. make sure to use the `raw` values to avoid information leakage from normalization
                if feature_space == "X":
                    X = a.loc[sel, :].sample(frac=frac, axis=1).sort_index(axis=1)
                else:
                    # N.B. dimensionality reduction should be done inside each fold
                    raise NotImplementedError

                X = X.join(pd.get_dummies(obs[covariates]))
                y = pd.get_dummies(obs.loc[sel, target_variables]).squeeze()
                groups = obs.reindex(y.index)[grouping_variable]

                try:
                    next(splitter.split(X, y, groups=groups))
                except ValueError:
                    print(f"Not enough samples for '{entity}'.")
                    continue

                args = [model, X, y, splitter, groups]
                preds_res, metrics, coefs = cross_validated_regression(
                    *args, preprocessing=True
                )
                r_preds_res, r_metrics, r_coefs = cross_validated_regression(
                    *args, shuffle=True, preprocessing=True
                )
                # adjust regression-to-the mean
                preds_res, metrics_adj = counter_regression_to_mean(
                    preds_res, model, splitter, groups
                )
                r_preds_res, r_metrics_adj = counter_regression_to_mean(
                    r_preds_res, model, splitter, groups
                )
                metrics = metrics.assign(**metrics_adj.to_dict())
                r_metrics = r_metrics.assign(**r_metrics_adj.to_dict())

                ent = {per: entity}
                _preds_reds.append(preds_res.assign(**ent))
                _preds_reds.append(r_preds_res.assign(**ent))
                _metrics.append(metrics.assign(**ent))
                _metrics.append(r_metrics.assign(**ent))
                _coefs.append(
                    coefs.mean(0)
                    .to_frame("original")
                    .join(r_coefs.mean(0).rename("shuffled"))
                    .assign(**ent)
                    .drop("shuffled")
                )

            preds_reds = pd.concat(_preds_reds)
            metrics = pd.concat(_metrics)
            coefs = pd.concat(_coefs)
            preds_reds.to_parquet(
                output_prefix.with_suffix(".predictions_residuals.pq")
            )
            metrics.to_parquet(output_prefix.with_suffix(".metrics.pq"))
            coefs.to_parquet(output_prefix.with_suffix(".coefficients.pq"))

    # # Pan-tissue clock
    splitters = [
        GroupKFold(20),
        StratifiedGroupKFold(20),
    ]
    for mode in ["no", "with"]:
        # Add tissue-specific covariate
        add = [per] if mode == "with" else []
        for model in models:
            model_name = model.__str__().split("(")[0]
            for splitter in splitters:
                cv_name = splitter.__str__().split("(")[0]
                output_prefix = (
                    output_dir
                    / f"pan-tissue_clock.{model_name}.{cv_name}.{mode}_tissue.SUFFIX"
                )
                if output_prefix.with_suffix(".coefficients.pq").exists():
                    continue
                tqdm.write(output_prefix.name)
                X = a.sample(frac=frac, axis=1).sort_index(axis=1)
                # TODO: implement getting X for low dimensional representations

                X = X.join(pd.get_dummies(obs[covariates + add]))
                y = pd.get_dummies(obs.loc[:, target_variables]).squeeze()
                groups = obs.reindex(y.index)[grouping_variable]

                args = [model, X, y, splitter, groups]
                preds_res, metrics, coefs = cross_validated_regression(*args)
                r_preds_res, r_metrics, r_coefs = cross_validated_regression(
                    *args, shuffle=True
                )

                # adjust regression-to-the mean
                preds_res, metrics_adj = counter_regression_to_mean(
                    preds_res, model, splitter, groups
                )
                r_preds_res, r_metrics_adj = counter_regression_to_mean(
                    r_preds_res, model, splitter, groups
                )
                metrics = metrics.assign(**metrics_adj.to_dict())
                r_metrics = r_metrics.assign(**r_metrics_adj.to_dict())

                p = pd.concat([preds_res, r_preds_res])
                p.to_parquet(output_prefix.with_suffix(".predictions_residuals.pq"))
                m = pd.concat([metrics, r_metrics])
                m.to_parquet(output_prefix.with_suffix(".metrics.pq"))
                c = (
                    coefs.mean(0)
                    .to_frame("original")
                    .join(r_coefs.mean(0).rename("shuffled"))
                    .drop("shuffled")
                )
                c.to_parquet(output_prefix.with_suffix(".coefficients.pq"))


def inspect_performance(
    feature_space: str = "X",
    feature_model_name: str = "fine_tuned",
    frac: float = 1.0,
    output_dir: Path | None = None,
):
    if output_dir is None:
        output_dir = (
            config.results_dir
            / feature_model_name
            / (
                f"age_{feature_space}"
                + (f"_frac{frac}" if feature_space == "X" else "")
            )
        )
        output_dir.mkdir(parents=True, exist_ok=True)

    model_names = [
        "LinearRegression",
        "Lasso",
        "LassoCV",
        "Ridge",
        "RidgeCV",
        "BayesianRidge",
        "ElasticNet",
        "ElasticNetCV",
        "RandomForestRegressor",
        # "XGBRegressor",
    ]
    cv_names = [
        "GroupKFold",
        "StratifiedGroupKFold",
        # "BalancedStratifiedGroupKFold(20:80:15)",
        # "BalancedStratifiedGroupKFold(0:90:10)",
        # "BalancedStratifiedGroupKFold(0:95:5)",
        # "TimeSeriesSplit",
    ]

    # # Tissue-specific clocks
    agg_preds_reds = list()
    agg_metrics = list()
    agg_coefs = list()
    for model_name in model_names:
        for cv_name in cv_names:
            s = output_dir / f"tissue-specific_clocks.{model_name}.{cv_name}.SUFFIX"
            if s.with_suffix(".predictions_residuals.pq").exists():
                preds_reds = pd.read_parquet(
                    s.with_suffix(".predictions_residuals.pq")
                ).assign(model_type="tissue_specific", model=model_name, cv=cv_name)
                metrics = pd.read_parquet(s.with_suffix(".metrics.pq")).assign(
                    model_type="tissue_specific", model=model_name, cv=cv_name
                )
                coefs = pd.read_parquet(s.with_suffix(".coefficients.pq")).assign(
                    model_type="tissue_specific", model=model_name, cv=cv_name
                )
                agg_preds_reds.append(preds_reds)
                agg_metrics.append(metrics)
                agg_coefs.append(coefs)

            for mode in ["no", "with"]:
                s = (
                    output_dir
                    / f"pan-tissue_clock.{model_name}.{cv_name}.{mode}_tissue.SUFFIX"
                )
                f = s.with_suffix(".predictions_residuals.pq")
                if f.exists():
                    preds_reds = pd.read_parquet(f).assign(
                        model_type="pan_tissue",
                        Tissue=f"All_{mode}",
                        model=model_name,
                        cv=cv_name,
                    )
                    metrics = pd.read_parquet(s.with_suffix(".metrics.pq")).assign(
                        model_type="pan_tissue",
                        Tissue=f"All_{mode}",
                        model=model_name,
                        cv=cv_name,
                    )
                    coefs = pd.read_parquet(s.with_suffix(".coefficients.pq")).assign(
                        model_type="pan_tissue",
                        Tissue=f"All_{mode}",
                        model=model_name,
                        cv=cv_name,
                    )
                    agg_preds_reds.append(preds_reds)
                    agg_metrics.append(metrics)
                    agg_coefs.append(coefs)

    # Inspect metrics and save
    q = pd.concat(agg_metrics)
    q["model"] = pd.Categorical(q["model"], ordered=True, categories=model_names)
    metrics = (
        q.groupby(
            ["Tissue", "model", "shuffled", "cv", "model_type"], observed=False
        ).mean()
        # .dropna()
    )
    metrics.to_csv(output_dir / "all_clocks.metrics.csv")
    # metrics = pd.read_csv(output_dir / "all_clocks.metrics.csv", index_col=0)
    metrics_red = metrics.groupby(
        level=["model", "shuffled", "cv", "model_type"]
    ).mean()
    metrics_red.to_csv(output_dir / "all_clocks.metrics.mean.csv")

    # Plot
    vmins = dict(
        n_train=10,
        n_valid=0,
        mean_y_train=20,
        mean_y_valid=20,
        time=0,
        r_squared=-1,
        mean_absolute_error=0,
        coefficient=0,
        intercept=0,
        explained_variance=-1,
        pearson=-1,
        r_squared_adj=-1,
        mean_absolute_error_adj=0,
        coefficient_adj=0,
        intercept_adj=0,
        explained_variance_adj=-1,
        pearson_adj=-1,
    )
    vmaxs = dict(
        n_train=1000,
        n_valid=1000,
        mean_y_train=70,
        mean_y_valid=70,
        time=15,
        r_squared=1,
        r_squared_adj=1,
        mean_absolute_error=20,
        mean_absolute_error_adj=20,
        coefficient=1,
        coefficient_adj=1,
        intercept=50,
        intercept_adj=15,
        explained_variance=1,
        explained_variance_adj=1,
        pearson=1,
        pearson_adj=1,
    )
    adds = dict(
        r_squared=dict(cmap="coolwarm", center=0),
        r_squared_adj=dict(cmap="coolwarm", center=0),
        explained_variance=dict(cmap="coolwarm", center=0),
        explained_variance_adj=dict(cmap="coolwarm", center=0),
    )
    metric_names = list(vmaxs)
    _m = len(metric_names)
    fig, axes = plt.subplots(
        2,
        _m,
        figsize=(_m * 4, 8 * 2),
        sharex=True,
        sharey=True,
        gridspec_kw=dict(hspace=0.01),
    )
    for axs, shuffled in zip(axes, [False, True]):
        for ax, m in zip(axs, metric_names):
            p = metrics.query("shuffled == @shuffled").pivot_table(
                index="Tissue", columns=["model", "cv"], values=m
            )
            sns.heatmap(
                p,
                cbar_kws=dict(label=m),
                ax=ax,
                xticklabels=True,
                yticklabels=True,
                vmin=vmins[m],
                vmax=vmaxs[m],
                **(adds[m] if m in adds else {}),
            )
            ax.set(xlabel="", ylabel="", title=m)
    fig.savefig(output_dir / "all_clocks.metrics.heatmap.svg", **config.figkws)

    # Make better heatmap contrasting real/shuffled in particular
    p = metrics.pivot_table(
        index="Tissue", columns=["model", "shuffled", "cv"], values=metric_names
    )
    fig, axes = plt.subplots(1, _m, figsize=(_m * 4, 8 * 1), sharey=True)
    for ax, m in zip(axes, metric_names):
        sns.heatmap(
            p[m],
            cbar_kws=dict(label=m),
            ax=ax,
            xticklabels=True,
            yticklabels=True,
            vmin=vmins[m],
            vmax=vmaxs[m],
            **(adds[m] if m in adds else {}),
        )
        ax.set(xlabel="", ylabel="", title=m)
    fig.savefig(
        output_dir / "all_clocks.metrics.heatmap.contrast_real_shuffled.svg",
        **config.figkws,
    )

    # Contrast models as scatter plots
    fig, ax = plt.subplots(figsize=(4 * 2, 3.8), sharex=True, sharey=True)
    ax.scatter(
        data=p.T.loc[:, :, False, :].T,
        x="r_squared",
        y="mean_absolute_error",
        label="Real",
        alpha=0.5,
    )
    ax.scatter(
        data=p.T.loc[:, :, True, :].T,
        x="r_squared",
        y="mean_absolute_error",
        label="Shuffled",
        alpha=0.5,
    )
    ax.set(xlabel="R squared", ylabel="MAE")
    ax.legend()
    fig.savefig(
        output_dir / "all_clocks.metrics.mean.joint_scatter.svg", **config.figkws
    )

    p = p.mean().unstack(None)
    fig, ax = plt.subplots(figsize=(4, 3.8))
    ax.scatter(
        data=p.loc[:, False, :], x="r_squared", y="mean_absolute_error", label="Real"
    )
    ax.scatter(
        data=p.loc[:, True, :], x="r_squared", y="mean_absolute_error", label="Shuffled"
    )
    for i in p.index:
        ax.text(
            p.loc[i, "r_squared"],
            p.loc[i, "mean_absolute_error"],
            s=f"{i[0]}: {i[1]}, {i[2]}",
        )
    ax.set(xlabel="R squared", ylabel="MAE")
    fig.savefig(output_dir / "all_clocks.metrics.mean.scatter.svg", **config.figkws)


def inspect_model_output(
    feature_space: str = "X",
    feature_model_name: str = "fine_tuned",
    frac: float = 1.0,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    model_name: str = "Ridge",  # "LinearRegression"
    cv_name: str = "GroupKFold",  # "StratifiedGroupKFold"
    target: str = "Age",
    per: str = "Tissue",
):
    from scipy.stats import normaltest

    if input_dir is None:
        input_dir = (
            config.results_dir
            / feature_model_name
            / (
                f"age_{feature_space}"
                + (f"_frac{frac}" if feature_space == "X" else "")
            )
        )
    if output_dir is None:
        output_dir = (
            config.results_dir
            / feature_model_name
            / (
                f"age_{feature_space}"
                + (f"_frac{frac}" if feature_space == "X" else "")
            )
            / f"result_inspection-{model_name}-{cv_name}"
        )
        output_dir.mkdir(exist_ok=True, parents=True)
    output_prefix = output_dir / f"inspection.{model_name}.{cv_name}.SUFFIX"

    # Join info from tissue-specific and pan-tissue clocks
    f = (
        input_dir
        / f"tissue-specific_clocks.{model_name}.{cv_name}.predictions_residuals.pq"
    )
    if not f.exists():
        print(f"Results for '{model_name}, {cv_name}' don't exist!")
        return
    df = pd.read_parquet(f)
    # Separate real and shuffled
    df = (
        df.query("shuffled == False")
        .drop(["shuffled"], axis=1)
        .join(
            df.query("shuffled == True")
            .drop(["Age", "Tissue", "shuffled"], axis=1)
            .add_suffix("-shuflled")
        )
    )
    df.columns += "_tissue_specific"

    f = input_dir / f"pan-tissue_clock.{model_name}.{cv_name}.predictions_residuals.pq"
    if f.exists():
        dfg = pd.read_parquet(f)
        # Separate real and shuffled
        dfg = (
            dfg.query("shuffled == False")
            .drop(["shuffled"], axis=1)
            .join(
                dfg.query("shuffled == True")
                .drop(["Age", "Tissue", "shuffled"], axis=1)
                .add_suffix("-shuflled")
            )
        )
        dfg.columns += "_pan_tissue"
        df = df.join(dfg).drop([f"{target}_pan_tissue"], axis=1)
    # df["target"] = df["prediction_pan_tissue"] + df["residuals_pan_tissue"]
    df = df.rename(
        columns={f"{per}_tissue_specific": per, f"{target}_tissue_specific": target}
    ).sort_index(axis=1)

    # Variables to inspect: predictions, residuals both real and shuffled
    vars = pd.Series([x for x in df.columns if "_" in x]).sort_values()[::-1]

    # Ranges of predictions and residuals (useful for plotting later)
    trange = tuple(df["Age"].describe().loc[["min", "max"]].squeeze())
    trangep = (trange[0] - trange[0] * 0.1, trange[1] + trange[1] * 0.1)
    q = df[[x for x in vars if "residuals" in x]].describe().loc[["min", "max"]]
    rrange = (q.loc["min"].mean(), q.loc["max"].mean())
    rrangep = (rrange[0] - rrange[0] * 0.1, rrange[1] + rrange[1] * 0.1)
    instances = sorted(df[per].unique())
    instance_grid = (4, 10)
    instance_figsize = (instance_grid[1] * 3, instance_grid[0] * 3)

    # Inspect predictions and residuals
    fig, axes = plt.subplots(
        1, len(vars), figsize=(len(vars) * 3, 1 * 3), sharey=True, sharex="row"
    )
    for ax, var in zip(axes.flat, vars):
        sns.histplot(df[var], ax=ax)
        ax.axvline(df[var].mean(), linestyle="-", color="red")
        if "residual" in var:
            ax.axvline(0, linestyle="--", color="grey")
    fig.savefig(
        output_prefix.with_suffix(".residuals.histplot.svg"),
        dpi=300,
        bbox_inches="tight",
    )

    # Inspect predictions
    for var in vars[vars.str.contains("prediction")]:
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.scatter(data=df, x=target, y=var, s=5, alpha=0.05, color="grey")
        sns.regplot(
            data=df,
            x=target,
            y=var,
            scatter=False,
            truncate=True,
            ax=ax,
            label="Predicted",
        )
        m = df[target].max() - df[target].min()
        ax.set(xlabel=target, ylabel=var, ylim=trangep)
        ax.plot(trange, trange, linestyle="--", color="black", label="Theoretical")
        fig.savefig(
            output_prefix.with_suffix(f".{var}.target_vs_prediction.scatter.svg"),
            dpi=300,
            bbox_inches="tight",
        )

        fig, axes = plt.subplots(
            *instance_grid, figsize=instance_figsize, sharex=True  # , sharey=True
        )
        for inst, ax in zip(instances, axes.flat):
            b = df.loc[df[per] == inst]
            ax.scatter(
                data=b,
                x=target,
                y=var,
                c=var.replace("prediction", "residuals"),
                label=inst,
                cmap="coolwarm",
                vmin=-5,
                vmax=5,
                alpha=0.5,
                rasterized=True,
            )
            sns.regplot(
                data=b,
                x=target,
                y=var,
                scatter=False,
                color="k",
                line_kws={"alpha": 0.25, "linestyle": "--"},
                truncate=True,
                ax=ax,
                label="Predicted",
            )
            ax.plot(
                trangep, trangep, linestyle="--", color="black", label="Theoretical"
            )
            ax.set(title=inst)
        for ax in axes[:-1, :].flat:
            ax.set(xlabel="")
        for ax in axes[:, 1:].flat:
            ax.set(ylabel="")
        # for ax in axes.flatten():
        #     ax.set(ylim=(-10, 110))
        fig.savefig(
            output_prefix.with_suffix(
                f".{var}.target_vs_prediction.scatter.per_tissue.fixed_y.svg"
            ),
            **config.figkws,
        )

    # Inspect residuals
    for var in vars[vars.str.contains("residual")]:
        fig, axes = plt.subplots(*instance_grid, figsize=instance_figsize, sharex=True)
        for inst, ax in zip(instances, axes.flat):
            q = df.query(f"Tissue == '{inst}'")[var]
            sns.histplot(q, ax=ax)
            p = normaltest(q).pvalue if q.shape[0] > 8 else np.nan
            ax.set(title=f"{inst}\np= {p:.3e}")
            ax.axhline(0, linestyle="--", color="grey")
        fig.savefig(
            output_prefix.with_suffix(f".residuals.normality.{var}.histplot.svg"),
            **config.figkws,
        )

        fig, axes = plt.subplots(*instance_grid, figsize=instance_figsize, sharex=True)
        for inst, ax in zip(instances, axes.flat):
            b = df.loc[df[per] == inst]
            ax.scatter(
                data=b,
                x=target,
                y=var,
                label=inst,
                alpha=0.05,
                s=5,
                rasterized=True,
            )
            sns.regplot(
                data=b,
                x=target,
                y=var,
                scatter=False,
                truncate=True,
                ax=ax,
            )
            ax.axhline(0, linestyle="--", color="grey")
            ax.set(title=inst)
        for ax in axes[1:, :].flat:
            ax.set(ylabel="")
        fig.savefig(
            output_prefix.with_suffix(
                f".residuals.normality.{var}.age_vs_residual.scatter.svg"
            ),
            **config.figkws,
        )

    # Compare tissue-specific and pan-tissue residuals
    if vars.isin(["residuals_pan_tissue", "residuals_tissue_specific"]).sum() >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(2 * 3.3, 3))
        for ax, v_ in zip(axes, ["", "_adj"]):
            ax.scatter(
                data=df,
                x=f"residuals{v_}_pan_tissue",
                y=f"residuals{v_}_tissue_specific",
                s=1,
                alpha=0.05,
                rasterized=True,
            )
            ax.plot(rrangep, rrangep, linestyle="--", color="grey")
            r = pg.corr(
                x=df[f"residuals{v_}_pan_tissue"],
                y=df[f"residuals{v_}_tissue_specific"],
            ).loc["pearson", "r"]
            ax.set(
                xlabel=f"residuals{v_}_pan_tissue",
                ylabel=f"residuals{v_}_tissue_specific",
                title=f"R = {r:.3f}",
            )
        fig.tight_layout()
        fig.savefig(
            output_prefix.with_suffix(
                ".residuals.comparison_tissue-specific_vs_pan_tissue.svg"
            ),
            dpi=300,
            bbox_inches="tight",
        )

    # Interpret the residuals

    # # prepare telomere lengths
    tl = get_telomere_lengths()
    print(f"{df.index.isin(tl.index).sum()} samples with telomere length data.")
    if "telomere_length" not in df:
        df = df.join(tl["TQImean"].rename("telomere_length"))

    # # prepare somatic mutations
    sm = get_somatic_mutation_counts().unstack().rename("somatic_mutations")
    df["SUBJID"] = df.index.str.extract(r"(GTEX-\w+)-\d+", expand=False)
    df = (
        df.reset_index()
        .merge(sm.reset_index(), on=["Tissue", "SUBJID"], how="left")
        .set_index("Tissue Sample ID")
    )
    m = df.dropna(subset="somatic_mutations").groupby("Tissue").size().mean()
    print(f"Mean of {m} samples with somatic mutation data per tissue.")

    # # get pathologies
    path_df = get_pathology_data()
    df["n_pathologies"] = path_df.sum(1)

    # # prepare comorbidities
    feats = get_engineered_info()
    morb = feats.loc[:, feats.columns.str.startswith("morbidity")].astype(bool)
    n = morb.sum(0).rename("n_individuals").sort_values()
    morb = morb.loc[:, n > 10]
    ind_n = morb.sum(1).rename("n_comorbidities")
    df["n_comorbidities"] = (
        df[["SUBJID"]]
        .reset_index()
        .merge(ind_n.reset_index(), on="SUBJID")
        .set_index("Tissue Sample ID")["n_comorbidities"]
    )
    morb_df = (
        df[["SUBJID"]]
        .reset_index()
        .merge(morb.reset_index(), on="SUBJID")
        .set_index("Tissue Sample ID")
        .drop("SUBJID", axis=1)
    )

    # # # quick inspection of the data
    pp = df.groupby("SUBJID")[[target, "n_pathologies", "n_comorbidities"]].agg(
        {target: "mean", "n_pathologies": "sum", "n_comorbidities": "mean"}
    )
    pp.corr()

    fig, axes = plt.subplots(3, 2, figsize=(4 * 2, 4 * 3))
    sns.histplot(data=pp, x="n_pathologies", ax=axes[0][0], bins=30)
    sns.scatterplot(data=pp, x="n_pathologies", y="Age", ax=axes[0][1], s=40, alpha=0.5)
    sns.histplot(data=pp, x="n_comorbidities", ax=axes[1][0], bins=30)
    sns.scatterplot(
        data=pp, x="n_comorbidities", y="Age", ax=axes[1][1], s=40, alpha=0.5
    )
    sns.scatterplot(
        data=pp, x="n_pathologies", y="n_comorbidities", ax=axes[2][1], s=40, alpha=0.5
    )
    axes[2][0].axis("off")
    fig.tight_layout()
    fig.savefig(
        output_prefix.with_suffix(
            ".pathologies.comorbidities.individual_level.histplot.scatter.svg"
        ),
        **config.figkws,
    )

    df[["n_pathologies", "n_comorbidities"]]
    for var in vars[vars.str.contains("residual")]:
        output_prefix2 = output_prefix.with_suffix(f".{var}.SUFFIX")

        # Rates of accelerated aging per tissue and compare
        rates = df.groupby(per)[var].mean().sort_values()
        fig, ax = plt.subplots(figsize=(3.3, 8))
        v = rates.std() * 3
        ax.scatter(rates, rates.index, c=rates, cmap="coolwarm", vmin=-v, vmax=v)
        ax.set(xlabel=var, ylabel=per)
        ax.axvline(0, linestyle="--", color="grey")
        fig.savefig(
            output_prefix2.with_suffix(".tissue_aging_rate.mean.scatter.svg"),
            **config.figkws,
        )

        # Rates per age bracket
        df[f"{target} Bracket"] = pd.cut(df[target], range(20, 80, 10))

        variances = (
            df.groupby([per, f"{target} Bracket"])[var].std().rename("residual_std")
        )

        fig, ax = plt.subplots(figsize=(4, 4))
        sns.barplot(
            data=variances.reset_index(), x=f"{target} Bracket", y="residual_std", ax=ax
        )
        fig.savefig(
            output_prefix2.with_suffix(".age_residuals.grouped_by_bracket.barplot.svg")
        )

        grid = clustermap(variances.unstack().dropna(), col_cluster=False)
        grid.savefig(
            output_prefix2.with_suffix(
                ".age_residuals.grouped_by_tissue_bracket.clustermap.svg"
            )
        )
        grid = clustermap(variances.unstack().dropna().T, config="z", row_cluster=False)
        grid.savefig(
            output_prefix2.with_suffix(
                ".age_residuals.grouped_by_tissue_bracket.clustermap.z.svg"
            )
        )

        # Visualize change with age per tissue
        fig, ax = plt.subplots(figsize=(3.3, 3))
        ax.scatter(data=df, x=target, y=var, rasterized=True, alpha=0.1, s=1)
        sns.regplot(data=df, x=target, y=var, scatter=False, ax=ax)
        ax.set(xlabel=target, ylabel=var)
        fig.savefig(
            output_prefix2.with_suffix(".scatter.age_vs_residual.svg"), **config.figkws
        )

        #

        #

        #

        # Connect to number of pathologies annotated per slide, or number of comorbidities
        for outcome, outcome_label in zip(
            ["n_pathologies", "n_comorbidities"],
            [
                "Number of pathologies in tissue",
                "Number of comorbidities per individual",
            ],
        ):
            # df[[outcome, var]].corr()
            m = df.groupby(outcome)[var].mean()
            m = (
                m.to_frame()
                .join(df.groupby(outcome)[var].size().rename("n"))
                .reset_index()
            )
            m = m.loc[m["n"] > 10]
            m[outcome] = m[outcome].astype("category")

            for window in [1, 2, 3]:
                mm = m.copy().drop(outcome, axis=1)
                mm[outcome] = m[outcome].to_numpy()
                mm.loc[mm[outcome] >= 10, outcome] = 10
                mm.loc[mm[outcome] == 10, "n"] = mm.loc[mm[outcome] == 10, "n"].sum()
                mm.loc[mm[outcome] == 10, var] = mm.loc[mm[outcome] == 10, var].mean()
                mm = mm.drop_duplicates()
                mm = mm.groupby(mm.index // window).agg(
                    {outcome: "mean", var: "mean", "n": "sum"}
                )
                mm[outcome] = (
                    ((mm[outcome] - (window / 2))).astype(int).astype(str)
                    + "-"
                    + ((mm[outcome] + (window / 2))).astype(int).astype(str)
                )
                fig, ax = plt.subplots(figsize=(4, 4))
                sns.barplot(
                    data=mm,
                    x=outcome,
                    y=var,
                    ax=ax,
                    hue=outcome,
                    palette="magma",
                    legend=False,
                    dodge=False,
                )
                for i in mm.index:
                    ax.annotate(
                        f"{mm.loc[i, 'n']}",
                        (i, mm.loc[i, var]),
                        ha="center",
                        va="bottom",
                    )
                ax.set(ylabel=f"{target} residual", xlabel=outcome_label)
                fig.savefig(
                    output_prefix2.with_suffix(
                        f".age_residuals.{outcome}.barplot.smoothed_{window=}.svg"
                    ),
                    dpi=300,
                    bbox_inches="tight",
                )

                if window != 1:
                    i = pd.IntervalIndex(
                        [
                            pd.Interval(*np.asarray(list(map(int, b.split("-")))))
                            for b in mm[outcome]
                        ]
                    )
                    # if i.is_overlapping:
                    #     continue
                    mmm = df.where(df[outcome].isin(m[outcome]))
                    mmm[outcome] = pd.cut(mmm[outcome], bins=i)
                else:
                    mmm = df.where(df[outcome].isin(m[outcome]))
                    mmm[outcome] = pd.Categorical(mmm[outcome])
                fig, ax = plt.subplots(figsize=(4, 4))
                sns.barplot(
                    data=mmm,
                    x=outcome,
                    y=var,
                    ax=ax,
                    hue=outcome,
                    palette="magma",
                    legend=False,
                    dodge=False,
                )
                for i in m.index:
                    ax.annotate(
                        f"{m.loc[i, 'n']}", (i, m.loc[i, var]), ha="center", va="bottom"
                    )
                ax.set(ylabel=f"{target} residual", xlabel=outcome_label)
                fig.savefig(
                    output_prefix2.with_suffix(
                        f".age_residuals.{outcome}.barplot.smoothed_{window=}.error_bars.svg"
                    ),
                    **config.figkws,
                )

                fig, ax = plt.subplots(figsize=(3, 3))
                mmm[outcome] = pd.Categorical(
                    mmm[outcome],
                    categories=mmm[outcome].cat.categories[::-1],
                    ordered=True,
                )
                sns.barplot(
                    data=mmm.sort_values(outcome, ascending=False),
                    y=outcome,
                    x=var,
                    ax=ax,
                    hue=outcome,
                    palette="magma",
                    legend=False,
                    dodge=False,
                    orient="horiz",
                )
                mm.index = mm.index[::-1]
                for i in mm.index:
                    ax.annotate(
                        f"{mm.loc[i, 'n']}",
                        (mm.loc[i, var], i),
                        ha="center",
                        va="bottom",
                    )
                ax.set(xlabel=f"{target} residual", ylabel=outcome_label)
                fig.savefig(
                    output_prefix2.with_suffix(
                        f".age_residuals.{outcome}.barplot.smoothed_{window=}.error_bars.horiz.svg"
                    ),
                    **config.figkws,
                )

        # For each pathological category calculate enrichment of residuals
        for outcome_df, outcome_label in zip(
            [path_df, morb_df], ["pathologies", "comorbidities"]
        ):
            _res_out: dict[str, dict[str, pd.Series]] = dict()
            for inst in instances:
                _res_out[inst] = dict()
                for path in outcome_df.columns:
                    d = (
                        df.query(f"{per} == '{inst}'")
                        .groupby(outcome_df[path])[var]
                        .mean()
                    )
                    if d.shape[0] == 1:
                        continue
                    _res_out[inst][path] = d[True] - d[False]
            res_out = pd.DataFrame(_res_out)

            n_o = outcome_df.sum(0).rename("n_individuals").astype(int)
            n_t = df.groupby("Tissue").size().rename("n_individuals")

            res_out = res_out.loc[:, res_out.columns[n_t > 300]]

            kwargs = dict(
                cmap="coolwarm",
                center=0,
                xticklabels=True,
                yticklabels=True,
                row_cluster=False,
                col_cluster=False,
                row_colors=n_o,
                col_colors=n_t,
                rasterized=True,
                dendrogram_ratio=0.1,
            )
            g = clustermap(res_out, **kwargs)
            g.ax_heatmap.set(xlabel="Tisues", ylabel=outcome_label)
            g.fig.savefig(
                output_prefix2.with_suffix(f".{outcome_label}_enrichment.heatmap.svg"),
                **config.figkws,
            )

            res_out = res_out.loc[
                res_out.sum(1).sort_values().index, res_out.sum(0).sort_values().index
            ]

            fig, ax = plt.subplots(figsize=(8, 7))
            g = clustermap(res_out, **kwargs)
            g.ax_heatmap.set(xlabel="Tisues", ylabel=outcome_label)
            g.fig.savefig(
                output_prefix2.with_suffix(
                    f".{outcome_label}_enrichment.heatmap.sorted.svg"
                ),
                **config.figkws,
            )

            # # Get Z-score across tissues, per outcome
            res_out_z = ((res_out.T - res_out.mean(1)) / res_out.std(1)).T
            g = clustermap(res_out_z, **kwargs)
            g.ax_heatmap.set(xlabel="Tisues", ylabel=outcome_label)
            g.fig.savefig(
                output_prefix2.with_suffix(
                    f".{outcome_label}_enrichment.heatmap.z.svg"
                ),
                **config.figkws,
            )

            kwargs.pop("row_cluster")
            kwargs.pop("col_cluster")
            g = clustermap(res_out_z.fillna(0), **kwargs, mask=res_out_z.isnull())
            g.ax_heatmap.set(xlabel="Tisues", ylabel=outcome_label)
            g.fig.savefig(
                output_prefix2.with_suffix(
                    f".{outcome_label}_enrichment.clustermap.z.svg"
                ),
                **config.figkws,
            )

        # Relationship between residuals and telomere length or somatic mutations
        for outcome, outcome_label in zip(
            ["telomere_length", "somatic_mutations"],
            ["Telomere length (TQI)", "Number of somatic mutations"],
        ):
            # # across all samples
            fig, ax = plt.subplots()
            ax.scatter(data=df, x=var, y=outcome, s=1, alpha=0.2)
            sns.regplot(data=df, x=var, y=outcome, ax=ax, scatter=False)
            fig.savefig(
                output_prefix2.with_suffix(f".age_residuals.{outcome}.scatter.svg"),
                **config.figkws,
            )

            # # continuous
            fig, axes = plt.subplots(
                *instance_grid, figsize=instance_figsize, sharex=False
            )
            for inst, ax in zip(instances, axes.flat):
                x = df.query(f"{per} == '{inst}'").dropna(subset=[outcome])
                if x.empty:
                    continue
                ax.axvline(0, linestyle="--", color="grey", linewidth=0.5)
                ax.scatter(data=x, x=var, y=outcome, s=1, alpha=0.2)
                sns.regplot(data=x, x=var, y=outcome, scatter=False, ax=ax)
                ax.set(title=inst)
                ax.text(0, x[outcome].mean(), s=f"n = {x.shape[0]}")
            for ax in axes[:, 1:].flatten():
                ax.set(ylabel="")
            for ax in axes[:-1, :].flatten():
                ax.set(xlabel="")
            # if axes is empty set axis('off')
            for ax in axes.flatten():
                if not ax.get_title():
                    ax.axis("off")
            fig.tight_layout()
            fig.savefig(
                output_prefix2.with_suffix(f".age_residuals.{outcome}.per_tissue.svg"),
                **config.figkws,
            )

            # # # selected tissues
            sel_instances = [
                inst
                for inst in instances
                if ((df[per] == inst) & (~df[outcome].isnull())).sum() > 200
            ]
            fig, axes = plt.subplots(4, 3, figsize=(3 * 3, 4 * 3), sharex=False)
            for inst, ax in zip(sel_instances, axes.flat):
                x = df.query(f"{per} == '{inst}'").dropna(subset=[outcome])
                ax.axvline(0, linestyle="--", color="grey", linewidth=0.5)
                ax.scatter(data=x, x=var, y=outcome, s=1, alpha=0.2)
                sns.regplot(data=x, x=var, y=outcome, scatter=False, ax=ax)
                ax.set(title=inst)
                ax.text(0, x[outcome].mean(), s=f"n = {x.shape[0]}")
            for ax in axes[:, 1:].flatten():
                ax.set(ylabel="")
            for ax in axes[:-1, :].flatten():
                ax.set(xlabel="")
            # if axes is empty set axis('off')
            for ax in axes.flatten():
                if not ax.get_title():
                    ax.axis("off")
            fig.tight_layout()
            fig.savefig(
                output_prefix2.with_suffix(
                    f".age_residuals.{outcome}.per_tissue.selected.svg"
                ),
                **config.figkws,
            )

            # # in bins
            fig, axes = plt.subplots(
                *instance_grid, figsize=instance_figsize, sharex=False
            )
            for inst, ax in zip(instances, axes.flat):
                x = df.query(f"{per} == '{inst}'").dropna(subset=[outcome])
                if x.shape[0] < 6:
                    ax.set(title=f"{inst} (n = {x.shape[0]})")
                    ax.axis("off")
                    continue
                x["split"] = pd.cut(
                    x[var],
                    bins=[-np.inf, -4, -1, 1, 4, np.inf],
                    labels=["low", "-4:-1", "-1:1", "1:4", "high"],
                )
                x = x.loc[x["split"].isin(["low", "high"])]
                x["split"] = x["split"].cat.remove_unused_categories()
                swarmboxenplot(data=x, x="split", y=outcome, ax=ax)
                n = x["split"].value_counts().sort_index().values
                ax.set(
                    title=f"{inst} (n = {x.shape[0]})",
                    xlabel=f"Aging residual\nn = {n}",
                    ylabel=outcome_label,
                )
            fig.tight_layout()
            fig.savefig(
                output_prefix2.with_suffix(
                    f".age_residuals.{outcome}.per_tissue.binary.swarmboxenplot.2_cuts.svg"
                ),
                **config.figkws,
            )
            fig, axes = plt.subplots(
                *instance_grid, figsize=instance_figsize, sharex=False
            )
            for inst, ax in zip(instances, axes.flat):
                x = df.query(f"{per} == '{inst}'").dropna(subset=[outcome])
                if x.shape[0] < 6:
                    ax.set(title=f"{inst} (n = {x.shape[0]})")
                    ax.axis("off")
                    continue
                x["cat"] = pd.cut(
                    x[var],
                    bins=[-np.inf, -4, -1, 1, 4, np.inf],
                    labels=["<-4", "-4:-1", "-1:1", "1:4", ">4"],
                )
                sns.violinplot(
                    data=x, x="cat", y=outcome, ax=ax, hue="cat", palette="magma"
                )
                n = x["cat"].value_counts().sort_index().values
                ax.set(
                    title=f"{inst} (n = {x.shape[0]})",
                    xlabel=f"Aging residual\nn = {n}",
                    ylabel=outcome_label,
                )
            fig.tight_layout()
            fig.savefig(
                output_prefix2.with_suffix(
                    f".age_residuals.{outcome}.per_tissue.binary.violinplot.5_cuts.svg"
                ),
                **config.figkws,
            )

            fig, axes = plt.subplots(
                *instance_grid, figsize=instance_figsize, sharex=False
            )
            for inst, ax in zip(instances, axes.flat):
                x = df.query(f"{per} == '{inst}'").dropna(subset=[outcome])
                if x.shape[0] < 6:
                    ax.set(title=f"{inst} (n = 0)")
                    ax.axis("off")
                    continue
                x["cat"] = pd.cut(
                    x[var], bins=8, labels=[f"{i}:{i+1}" for i in range(-4, 4)]
                )
                sns.violinplot(
                    data=x, x="cat", y=outcome, ax=ax, hue="cat", palette="magma"
                )
                n = x["cat"].value_counts().sort_index().values
                ax.set(
                    title=f"{inst} (n = {x.shape[0]})",
                    xlabel=f"Aging residual\nn = {n}",
                    ylabel=outcome_label,
                )
            fig.tight_layout()
            fig.savefig(
                output_prefix2.with_suffix(
                    f".age_residuals.{outcome}.per_tissue.binary.violinplot.8_cuts.svg"
                ),
                **config.figkws,
            )

            # # inspect dependency of telomere relationship on sample size
            corr = df.groupby(per)[[var, outcome]].corr()
            corr = corr.loc[:, var, :][outcome]
            n = df.dropna(subset=[outcome]).groupby(per).size()
            corr = corr.to_frame("correlation").join(n.rename("count"))

            fig, axes = plt.subplots(2, 1, figsize=(7, 5 * 2))
            for ax in axes:
                ax.scatter(data=corr, x="count", y="correlation")
                for i in corr.dropna().index:
                    ax.text(
                        corr.loc[i, "count"],
                        corr.loc[i, "correlation"],
                        s=i,
                        fontsize=8,
                    )
                ax.axhline(0, linestyle="--", color="grey")
                ax.set(
                    xlabel="Number of samples",
                    ylabel=f"Correlation between rate of aging and {outcome_label}",
                )
            axes[-1].set(xscale="log")
            fig.savefig(
                output_prefix2.with_suffix(
                    f".age_residuals.{outcome}.correlation.{per}_scatter.svg"
                ),
                **config.figkws,
            )
