from tqdm import tqdm
import numpy as np
import pandas as pd
from anndata import AnnData
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn_extensions import clustermap
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src import config

from src.utils import DummySplitter, get_restricted_info
from src.predict_histological_age import counter_regression_to_mean


feature_model_name: str = "fine_tuned"
model_name: str = "Ridge"
cv_name: str = "GroupKFold"
feature_space: str = "X"
frac: float = 1.0
target_variables: list[str] = ["Age"]
covariates: list[str] = ["Sex", "Ischemic Time (Minutes)", "Cohort"]
per: str = "Tissue"
exclude_entities: list[str] = []

input_dir = (
    config.results_dir
    / feature_model_name
    / (f"age_{feature_space}" + (f"_frac{frac}" if feature_space == "X" else ""))
)
input_prefix = input_dir / f"tissue-specific_clocks.{model_name}.{cv_name}.SUFFIX"
output_dir = (
    config.results_dir
    / feature_model_name
    / (f"age_{feature_space}" + (f"_frac{frac}" if feature_space == "X" else ""))
) / f"cross_prediction-{model_name}-{cv_name}"
output_dir.mkdir(parents=True, exist_ok=True)


def main():
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

    cross_prediction(a)

    compare_tissues()


def cross_prediction(a: AnnData):
    from sklearn import set_config

    assert len(target_variables) == 1
    target = target_variables[0]

    preprocessing = True

    # Get estimated coefficients
    coefs = pd.read_parquet(input_prefix.with_suffix(".coefficients.pq"))

    _input_prefix = (
        input_dir / f"pan-tissue_clock.{model_name}.{cv_name}.no_tissue.SUFFIX"
    )
    coefs = pd.concat(
        [
            coefs,
            pd.read_parquet(_input_prefix.with_suffix(".coefficients.pq")).assign(
                Tissue="All_no"
            ),
        ]
    )
    coefs = coefs.drop(["shuffled"], axis=1, errors="ignore")

    # Get same model type
    models = [
        # LinearRegression(fit_intercept=True, n_jobs=-1),
        # Lasso(2, fit_intercept=True),
        # LassoCV(alphas=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0], fit_intercept=True, cv=5),
        Ridge(2, fit_intercept=True),
        # RidgeCV(alphas=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0], fit_intercept=True, cv=5),
        # BayesianRidge(),
        # ElasticNet(0.1, l1_ratio=0.5, fit_intercept=True),
        # RandomForestRegressor(n_jobs=-1),
        # XGBRegressor(n_jobs=-1),
        # ARDRegression(fit_intercept=True),
        # SGDRegressor(),
    ]
    models_map = {m.__str__().split("(")[0]: m for m in models}
    model = models_map[model_name]
    splitter = DummySplitter()

    output_prefix = output_dir / f"tissue-specific_clocks.{model_name}.{cv_name}.SUFFIX"

    entities = [
        x for x in sorted(coefs["Tissue"].unique()) if x not in exclude_entities
    ]

    set_config(transform_output="pandas")
    _tqdm0 = tqdm(entities, position=0, leave=True)
    for entity_to in _tqdm0:
        if entity_to.startswith("All_"):
            continue
        # entity_to = entities[0]
        sel = a.obs[per] == entity_to
        _tqdm0.set_description(f"{entity_to}, n = {sel.sum()}")

        # N.B. make sure to use the `raw` values to avoid information leakage from normalization
        if feature_space == "X":
            X = (
                a.raw[:, a.var.index]
                .to_adata()[sel]
                .to_df()
                .sample(frac=frac, axis=1)
                .sort_index(axis=1)
            )
        else:
            # N.B. dimensionality reduction should be done inside each fold
            raise NotImplementedError

        X = X.join(pd.get_dummies(a.obs[covariates]))
        y = a.obs.loc[sel, target]

        if preprocessing:
            xscaler = StandardScaler()
            X = xscaler.fit_transform(X)
            yscaler = StandardScaler()
            yscaler.fit(y.to_frame())

        _res = list()
        _tqdm1 = tqdm(entities, position=1, leave=False)
        for entity_from in _tqdm1:
            # entity_from = entities[0]
            c = coefs.query(f"{per} == @entity_from")["original"]
            model = Ridge(2, fit_intercept=True)
            if model_name != "RandomForestRegressor":
                model.coef_ = c.drop("fit_intercept", errors="ignore")
                model.intercept_ = c.get("fit_intercept")
            else:
                raise NotImplementedError
                model.feature_importances_[:] = c.drop("fit_intercept", errors="ignore")
            model.feature_names_in_ = c.index.drop("fit_intercept")
            yp = model.predict(X)
            if preprocessing:
                yp = yscaler.inverse_transform(yp.reshape(-1, 1)).squeeze(axis=1)
            res = pd.DataFrame(dict(prediction=yp), index=X.index)
            res["residuals"] = res["prediction"] - y
            groups = a.obs.loc[sel, "Subject ID"]
            res, met = counter_regression_to_mean(res, model, splitter, groups)
            _res.append(res.assign(entity_from=entity_from, entity_to=entity_to))
        res = pd.concat(_res)
        res.to_parquet(output_prefix.with_suffix(f".to_{entity_to}.predictions.pq"))
    set_config(transform_output="default")


def compare_tissues(per="Tissue"):
    output_dir = (
        config.results_dir
        / feature_model_name
        / (f"age_{feature_space}" + (f"_frac{frac}" if feature_space == "X" else ""))
    ) / f"cross_prediction-{model_name}-{cv_name}"
    output_dir
    output_prefix = output_dir / f"tissue-specific_clocks.{model_name}.{cv_name}.SUFFIX"

    # Analysis
    files = output_dir.glob(output_prefix.name + "*.pq")
    res = (
        pd.concat(pd.read_parquet(f) for f in files)
        .sort_values(["entity_from", "entity_to"])
        .query(
            "~entity_from.isin(@exclude_entities) and ~entity_to.isin(@exclude_entities)"
        )
    ).drop(["value_type"], axis=1, errors="ignore")
    res = res.query("~entity_from.str.startswith('All_')")
    res["residuals:absolute"] = res["residuals"].abs()
    res["residuals_adj:absolute"] = res["residuals_adj"].abs()

    ages = (
        res.groupby(["entity_from", "entity_to"])["Age"].mean().unstack("entity_from")
    )
    preds = (
        res.groupby(["entity_from", "entity_to"])["prediction_adj"]
        .mean()
        .unstack("entity_from")
    )
    preds_norm = np.log2(preds / ages.mean(1))
    means = (
        res.groupby(["entity_from", "entity_to"])["residuals_adj:absolute"]
        .mean()
        .unstack("entity_from")
    )
    for col in means.columns[means.columns.str.startswith("All_")]:
        means[col] = (
            (means[col] - 0) / means[col].std() * means.drop([col], axis=1).std().mean()
        )
    means_directed = (
        res.groupby(["entity_from", "entity_to"])["residuals_adj"]
        .mean()
        .unstack("entity_from")
    )
    # means_directed = means_directed * means.mean()
    # means_directed = (means_directed.T * means.mean(1)).T
    means_directed = means_directed / means_directed.max(1).max()

    clock_metrics = (
        pd.read_parquet(input_prefix.with_suffix(".metrics.pq"))
        .groupby(per)
        .mean()[
            ["pearson", "r_squared", "mean_absolute_error"]
            + ["pearson_adj", "r_squared_adj", "mean_absolute_error_adj"]
        ]
    )

    kwargs = dict(square=False, col_colors=clock_metrics, figsize=(8, 8), cmap="PuOr_r")
    df: pd.DataFrame
    for df, label in [
        (ages, "Age"),
        (preds, "prediction_adj"),
        (preds_norm, "prediction_adj:log_ratio"),
        (means, "residuals_adj:absolute"),
        (means_directed, "residuals_adj:scaled"),
    ]:
        m = (df.mean(1) / df.mean(1).max()).rename("Relative age acceleration")
        kwargs.update(row_colors=m)
        # add jitter to 'Age' because all columns have same variance
        if df.var(0).var() < 1e-25:
            df += np.random.normal(0, 1e-5, df.shape)
        g = clustermap(
            df.copy(),
            config="abs",
            center=0,
            **kwargs,
        )
        g.fig.savefig(
            output_prefix.with_suffix(f".means.comparison.{label}.clustermap.abs.svg"),
            **config.figkws,
        )
        g = clustermap(df.copy(), config="z", **kwargs)
        g.fig.savefig(
            output_prefix.with_suffix(f".means.comparison.{label}.clustermap.z.svg"),
            **config.figkws,
        )
        g = clustermap(
            df.copy(),
            row_cluster=False,
            col_cluster=False,
            config="abs",
            center=0,
            **kwargs,
        )
        g.fig.savefig(
            output_prefix.with_suffix(
                f".means.comparison.{label}.clustermap.abs.sorted.svg"
            ),
            **config.figkws,
        )
        g = clustermap(
            df.copy(), row_cluster=False, col_cluster=False, config="z", **kwargs
        )
        g.fig.savefig(
            output_prefix.with_suffix(
                f".means.comparison.{label}.clustermap.z.sorted.svg"
            ),
            **config.figkws,
        )

    for df, label in [
        (ages, "Age"),
        (preds, "prediction_adj"),
        (preds_norm, "prediction_adj:log_ratio"),
        (means, "residuals_adj:absolute"),
        (means_directed, "residuals_adj:scaled"),
    ]:
        q = AnnData(df.copy())
        x = df.copy()
        # add jitter to 'Age' because all columns have same variance
        if x.var(0).var() < 1e-25:
            x += np.random.normal(0, 1e-5, x.shape)
            q.X = x.copy()
        q.obs = q.obs.join(clock_metrics)
        np.fill_diagonal(x.values, np.nan)
        q.obs["Relative age acceleration"] = x.mean(1) / x.mean(1).max()
        # sc.pp.scale(q)
        sc.pp.pca(q)
        sc.pp.neighbors(q, 5)
        sc.tl.umap(q)
        sc.tl.diffmap(q)
        q.obsm["X_diffmap"] = q.obsm["X_diffmap"][:, 1:]
        sc.tl.draw_graph(q)
        sc.tl.tsne(q)
        sc.tl.draw_graph(q)
        # sc.write(output_prefix.with_suffix(f".means.comparison.{label}.h5ad", q)
        q.write_csvs(output_prefix.with_suffix(f".means.comparison.{label}.adata"))

        embeddings = pd.Series(q.obsm.keys()).str.replace("X_", "")
        fig, axes = plt.subplots(
            len(embeddings),
            q.obs.shape[1],
            figsize=(9 * q.obs.shape[1], 6 * len(embeddings)),
            gridspec_kw=dict(wspace=0.1),
        )
        for axs, emb in zip(axes, embeddings):
            for ax, color in zip(axs, q.obs.columns):
                sc.pl.embedding(
                    q,
                    basis=emb,
                    color=color,
                    cmap="inferno",
                    s=75,
                    ax=ax,
                    return_fig=False,
                )
                for i, idx in enumerate(q.obs.index):
                    ax.text(*q.obsm[f"X_{emb}"][i, :2], s=idx)
        fig.savefig(
            output_prefix.with_suffix(f".means.comparison.{label}.embeddings.svg"),
            **config.figkws,
        )

    # # Contrast raw and normalized
    # for emb in embeddings:
    #     _projs = dict()
    #     for df, label in [
    #         (means, "residuals_adj:absolute"),
    #         (means_directed, "residuals_adj:scaled"),
    #     ]:
    #         _projs[label] = pd.read_csv(
    #             output_prefix.with_suffix(f".means.comparison.{label}.adata/obsm.csv")
    #         )[f"X_{emb}1"]
    #         _projs[label].index = pd.read_csv(
    #             output_prefix.with_suffix(f".means.comparison.{label}.adata/obs.csv")
    #         )["entity_to"]
    #     joint = pd.DataFrame(_projs)
    #     # joint.iloc[:, 0] *= -1

    #     fig, ax = plt.subplots(figsize=(6, 6))
    #     ax.scatter(*joint.T.values, s=10)
    #     texts = []
    #     for i, txt in enumerate(joint.index):
    #         texts += [ax.text(joint.iloc[i, 0], joint.iloc[i, 1], s=txt, ha="center")]
    #     # from adjustText import adjust_text
    #     # adjust_text(texts, arrowprops=dict(arrowstyle="->", lw=0.1), ax=ax, autoalign=True)
    #     v = joint.abs().max().max()
    #     ax.plot([-v, v], [-v, v], "k--", lw=0.5)
    #     ax.axhline(0, color="k", lw=0.25)
    #     ax.axvline(0, color="k", lw=0.25)
    #     fig.savefig(output_prefix + f"means.{emb}_comparison.scatter.svg")

    # Observe deviations in residuals along age range
    def minmax_scale(x):
        return (x - x.min()) / (x.max() - x.min())

    for scale, end in [(True, "scaled"), (False, "no_scaled")]:
        for df, label in [
            # (ages, "Age"),
            (preds, "prediction_adj"),
            # (preds_norm, "prediction_adj:log_ratio"),
            (means, "residuals_adj:absolute"),
            # (means_directed, "residuals_adj:scaled"),
        ]:
            _smoothed = dict()
            fig, axes = plt.subplots(
                4, 10, figsize=(10 * 2.5, 4 * 2.5), sharex=True, sharey=True
            )

            # Get nice y-axis lims first
            x = (
                res.groupby(level=0)[["Age", label]]
                .mean()
                .set_index("Age")
                .squeeze()
                .sort_index()
            )
            xp = x.ewm(span=10).mean()
            if scale:
                xp -= xp.mean()
            v = xp.abs().quantile(0.995)
            for ax, tissue in zip(axes.flatten(), means.index):
                x = (
                    res.query("entity_to == @tissue")  # & "entity_from != @tissue"
                    .groupby(level=0)[["Age", label]]
                    .mean()
                    .set_index("Age")
                    .squeeze()
                    .sort_index()
                )
                xp = x.ewm(span=10).mean()
                if scale:
                    xp -= xp.mean()
                ax.scatter(xp.index, xp, s=1, alpha=0.1, c="k")

                # Smooth
                r = (
                    x.ewm(span=10)
                    .mean()
                    .groupby(level=0)
                    .mean()
                    .rolling(10, center=True)
                    .mean()
                )
                if scale:
                    r = r - x.mean()
                _smoothed[tissue] = r
                # ax.plot(r, "-", linewidth=8, alpha=0.5)
                # sns.lineplot(
                #     x=xp.index,
                #     y=xp,
                #     ax=ax,
                #     color="blue",
                #     linewidth=1,
                #     alpha=0.85,
                #     estimator="mean",
                #     errorbar=("se", 3),
                #     err_style="band",
                # )
                sns.regplot(
                    x=xp.index,
                    y=xp,
                    ax=ax,
                    color="blue",
                    scatter=False,
                    line_kws=dict(linewidth=1, alpha=0.85),
                    truncate=True,
                    lowess=True,
                )
                sns.regplot(
                    x=xp.index,
                    y=xp,
                    ax=ax,
                    color="orange",
                    scatter=False,
                    line_kws=dict(linewidth=1, alpha=0.85),
                    truncate=True,
                    order=3,
                )
                ax.plot(r, "-", color="black", linewidth=1, alpha=1)

                ax.set(title=tissue, xlabel="", ylabel="")
                if scale:
                    ax.axhline(0, color="k", lw=0.5)
                    # v = xp.abs().quantile(.95)
                    # v += v * 0.1
                    ax.set(ylim=(-v, v))
                else:
                    ax.set_ylim(top=v)
            fig.supxlabel("Age", y=0.05)
            fig.supylabel("Rate of change relative to other tissues", x=0.1)
            fig.savefig(
                output_prefix.with_suffix(f".{label}.deviation_along_age.{end}.svg"),
                **config.figkws,
            )

            m = (df.mean(1) / df.mean(1).max()).rename("Relative age acceleration")
            smoothed = pd.DataFrame(_smoothed).T
            smoothed = smoothed.loc[:, ~smoothed.isnull().all()]
            add = {}
            if scale:
                v = np.nanpercentile(smoothed.abs().values, 99)
                v += v * 0.1
                add = dict(vmin=-v, vmax=v)

            g = clustermap(
                smoothed.fillna(0),
                mask=smoothed.isnull(),
                cmap="PuOr_r",
                center=0,
                col_cluster=False,
                row_colors=m,
                figsize=(6, 7),
                metric="cosine",
                dendrogram_ratio=0.1,
                **add,
            )
            # set xtick frequency to every 5 years
            g.ax_heatmap.set_xticks(np.arange(0, len(smoothed.columns), 5))
            g.ax_heatmap.set_xticklabels(smoothed.columns[::5])
            g.savefig(
                output_prefix.with_suffix(
                    f".{label}.deviation_along_age.clustermap.{end}.svg"
                ),
                **config.figkws,
            )
