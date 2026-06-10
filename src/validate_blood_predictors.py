from tqdm.auto import tqdm
import pandas as pd
import scanpy as sc
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
import pingouin as pg
from seaborn_extensions import swarmboxenplot, clustermap

from src import config
from src.prepare_archs4_data import load_archs4_data

study = "ARCHS4-all"


def main():
    inspect_archs4_data()
    predict_age_gaps_archs4()


def inspect_archs4_data():
    contrasts = ["Healthy", "Crohn's Disease", "Rheumatoid Arthritis"]

    a = load_archs4_data()
    a = a[
        a.obs.query(
            "`harmonized:group` in @contrasts & `harmonized:age`.notnull() & `harmonized:age` > 20"
        ).index
    ]
    a.obs["harmonized:group"] = pd.Categorical(
        a.obs["harmonized:group"], contrasts, ordered=True
    )
    palette = np.asarray(sns.color_palette("tab10"))[[2, 1, 0]]
    a.uns["harmonized:group_colors"] = [matplotlib.colors.rgb2hex(c) for c in palette]
    a.obs["harmonized:sex"] = (
        a.obs["harmonized:sex"]
        .str.replace(r"^f$", "female", regex=True)
        .str.replace(r"^m$", "male", regex=True)
    )
    a.obs["harmonized:age"] = a.obs["harmonized:age"].astype(float)
    a = a[~a.obs["series_id"].isin(["GSE161031", "GSE152197,GSE152543", "GSE110041"])]
    a = a[~a.obs["title"].str.contains("CD")]
    a.var.index = a.var["symbol"]
    a = a[:, a.X.sum(0) > 0]
    a.uns["voi"] = ["harmonized:age", "harmonized:sex", "harmonized:group"]

    sc.pp.normalize_total(a)
    sc.pp.log1p(a)
    a = a[:, a.X.mean(0) >= 1]
    sc.pp.scale(a)
    sc.pp.combat(a, key="harmonized:group")
    sc.pp.scale(a)
    sc.pp.pca(a)
    sc.pp.neighbors(a)
    sc.tl.umap(a)
    isomap(a)
    sc.tl.diffmap(a)
    a.obsm["X_diffmap"] = a.obsm["X_diffmap"][:, 2:]
    for emb in ["pca", "umap", "isomap", "diffmap"]:
        fig = sc.pl.embedding(
            a, basis=emb, color=a.uns["voi"], ncols=1, return_fig=True
        )
        fig.savefig(
            config.results_dir
            / "predict_gaps_from_blood_expression"
            / f"{study}.unsupervised_analysis.{emb}.svg",
            **config.figkws,
        )

    x = a.to_df()
    x = (x - x.mean()) / x.std()
    coefs = pd.Series(
        sklearn.linear_model.Ridge(2 * x.shape[1]).fit(x, a.obs["age"]).coef_,
        index=a.var.index,
    ).sort_values()
    sel = coefs.tail(50).index.tolist() + coefs.head(50).index.tolist()
    o = (x @ coefs).sort_values().index
    p = x.reindex(index=o, columns=sel)

    g = clustermap(
        p,
        config="z",
        square=False,
        row_colors=pd.get_dummies(
            a.obs[["harmonized:age", "harmonized:sex", "harmonized:group"]]
        ),
        row_cluster=False,
        dendrogram_ratio=0.1,
        figsize=(4, 2.5),
        xticklabels=False,
        yticklabels=False,
    )
    g.ax_heatmap.get_children()[0].set_rasterized(True)
    g.fig.savefig(
        config.results_dir
        / "predict_gaps_from_blood_expression"
        / f"{study}.unsupervised_analysis.age_heatmap.svg",
        **config.figkws,
    )


def get_predictor_coefficients(var: str = "Age", shuffled: bool = False):
    f = (
        config.results_dir
        / "predict_gaps_from_blood_expression"
        / f"tissue-specific_clocks.Ridge.KFold.predictions_residuals.predicted_from_blood_expression.age_regressed.{var}.coefs.csv"
    )
    coefs = (
        pd.read_csv(f, index_col=0)
        .query("Shuffled == @shuffled")
        .drop("Shuffled", axis=1)
        .groupby("Tissue")
        .mean()
        .drop(["max", "sum", "std"], axis=0, errors="ignore")
    )
    return coefs


def check_coeffients():
    var = "residuals_adj"
    coefs = get_predictor_coefficients(var)
    coefs = coefs.rename(index={"mean": "Systemic"})

    fig, axes = plt.subplots(
        4,
        10,
        figsize=(10 * 1.5, 4 * 1.5),
        sharex=False,
        sharey=False,
        gridspec_kw=dict(hspace=0.05, wspace=0.05),
    )
    fig2, ax2 = plt.subplots(1, 1, figsize=(1 * 4, 1 * 4))
    for ax, organ in zip(axes.ravel(), coefs.index):
        sns.kdeplot(coefs.loc[organ], ax=ax)
        sns.kdeplot(coefs.loc[organ], ax=ax2)
        ax.axvline(0, color="k", ls="--")
        ax.set(title=organ, xlabel="")
    ax2.set(xlabel="Coefficient")
    fig.savefig(
        config.results_dir
        / "predict_gaps_from_blood_expression"
        / f"{study}.unsupervised_analysis.{var}.coefs.separate.svg",
        **config.figkws,
    )
    fig2.savefig(
        config.results_dir
        / "predict_gaps_from_blood_expression"
        / f"{study}.unsupervised_analysis.{var}.coefs.joint.svg",
        **config.figkws,
    )


def check_predictions():
    var = "residuals_adj"

    preds = pd.read_csv(
        config.results_dir
        / "predict_gaps_from_blood_expression"
        / f"{study}.predicted_from_blood_expression.{var}.predictions.csv",
        index_col=0,
    )
    preds.index = preds.index.astype("str")

    preds_organ = (
        preds.T.groupby(preds.columns.str.replace(" - .+", "", regex=True)).mean().T
    ).rename(columns=dict(mean="Systemic"))

    fig, axes = plt.subplots(3, 7, figsize=(7 * 1.5, 3 * 1.5))
    fig2, ax2 = plt.subplots(1, 1, figsize=(1 * 4, 1 * 4))
    for ax, organ in zip(axes.ravel(), preds_organ.columns):
        sns.kdeplot(preds_organ[organ], ax=ax)
        sns.kdeplot(preds_organ[organ], ax=ax2)
        ax.axvline(0, color="k", ls="--")
        ax.set(title=organ)
    ax2.set(xlabel="Prediction")
    fig.savefig(
        config.results_dir
        / "predict_gaps_from_blood_expression"
        / f"{study}.unsupervised_analysis.{var}.predictions.separate.svg",
        **config.figkws,
    )
    fig2.savefig(
        config.results_dir
        / "predict_gaps_from_blood_expression"
        / f"{study}.unsupervised_analysis.{var}.predictions.joint.svg",
        **config.figkws,
    )


def predict_age_gaps_archs4():
    var = "residuals_adj"

    a = load_archs4_data()
    coefs = get_predictor_coefficients(var)

    # metadata fixes
    a.var.index = a.var["symbol"]
    a.obs["harmonized:group"] = a.obs["harmonized:group"].replace(
        "Lupus", "Systemic Lupus Erythematosus"
    )
    a.obs["harmonized:sex"] = (
        a.obs["harmonized:sex"].replace("f", "female").replace("m", "male")
    )
    a.obs[["readsaligned", "singlecellprobability"]] = a.obs[
        ["readsaligned", "singlecellprobability"]
    ].astype(float)
    cov = a.obs["readsaligned"].astype(int)
    a.obs["coverage"] = (cov - cov.mean()) / cov.std()

    # filter out
    blood_disorders = [
        "Chronic Lymphocytic Leukemia",
        "Chronic Myelogenous Leukemia",
        "Chronic Myeloid Leukemia",
        "Acute Myeloid Leukemia",
        "Acute Myeloid Leukemia",
        "Sickle Cell Disease",
        "Leukemia",
        "Thrombocytopenia",
    ]
    infectious = [
        "Tuberculosis",
        "COVID-19",
        "Sepsis",
        "Leprosy",
        "Bacterial Infection",
    ]
    cancer = [
        "Hepatocellular Carcinoma",
        "Cervical Intraepithelial Neoplasia",
    ]
    developmental = [
        "Bronchopulmonary Dysplasia",
    ]
    a = a[
        ~a.obs["harmonized:group"].isin(
            blood_disorders + infectious + cancer + developmental + ["Control"]
        ),
        :,
    ]
    a = a[~a.obs["characteristics_ch1"].str.contains("pediatric"), :]
    a = a[~a.obs["source_name_ch1"].str.contains("pediatric"), :]
    a = a[a.obs["harmonized:intervention"].isnull()]
    a = a[
        ~a.obs["series_id"].isin(
            [
                "GSE172009",
                "GSE152197,GSE152543",
                "GSE97263",
                "GSE122624",
                "GSE146447",
                "GSE174325",
                "GSE73570",
                "GSE190518",
                "GSE178764",
            ]
        )
    ]
    a = a[a.obs["instrument_model"].str.startswith("Illumina"), :]
    a = a[a.obs["readsaligned"] < 1e8, :]
    a = a[a.obs["singlecellprobability"] < 0.01, :]
    c = a.obs["series_id"].value_counts()
    a = a[a.obs["series_id"].isin(c[c > 1].index), :]
    a.obs[["series_id"]] = a.obs[["series_id"]].astype("category")
    a.obs[["harmonized:group"]] = a.obs[["harmonized:group"]].astype("category")
    c = a.obs["harmonized:group"].value_counts()
    groups = c[c > 20].index.tolist()
    groups = [g for g in groups if g not in ["Uncategorized Case", "Other"]]
    a = a[a.obs["harmonized:group"].isin(groups)]
    s = a.obs["series_id"].value_counts()
    series = s[s >= 3].index.tolist()
    a = a[a.obs["series_id"].isin(series)]
    a.obs.to_csv(
        config.results_dir
        / "predict_gaps_from_blood_expression"
        / f"{study}.predicted_from_blood_expression.metadata.csv",
    )

    fig, ax = plt.subplots(figsize=(1.5, 1.5))
    a.obs["harmonized:group"].value_counts().sort_values().plot.barh(ax=ax)
    ax.set(xlabel="Number of samples", ylabel="Group")
    fig.savefig(
        config.results_dir
        / "predict_gaps_from_blood_expression"
        / f"{study}.predicted_from_blood_expression.groups.bar.svg",
        **config.figkws,
    )

    a = a[:, a.var["symbol"].isin(coefs.columns)]

    ndf = a.to_df().T.groupby(level=0).max().T
    a = sc.AnnData(ndf, obs=a.obs, var=ndf.columns.to_frame())
    sc.pp.normalize_total(a)
    sc.pp.log1p(a)
    sc.pp.scale(a)
    sc.pp.regress_out(a, "coverage")
    sc.pp.scale(a)

    mean_corr = a.to_df().groupby(a.obs["harmonized:group"]).mean().T.corr()
    g = clustermap(
        mean_corr, figsize=(1.5, 1.5), cmap="RdBu_r", center=0, dendrogram_ratio=0.05
    )
    g.fig.savefig(
        config.results_dir
        / "predict_gaps_from_blood_expression"
        / f"{study}.predicted_from_blood_expression.mean_expression.corr.clustermap.svg",
        **config.figkws,
    )

    # regress age out
    b = a[
        a.obs.query("`harmonized:age`.notna() & `harmonized:group` == 'Healthy'").index,
        :,
    ].copy()
    x = b.to_df()
    y = b.obs["harmonized:age"]
    model = sklearn.linear_model.Ridge(2 * x.shape[1])
    model.fit(x, y)
    all_x = a.to_df()
    new_a_x = all_x - model.predict(all_x).reshape(-1, 1) * model.coef_
    a.X = new_a_x
    a.var.index = a.var["symbol"]

    # Predict age gaps from blood
    sex_specific_tissues_to_exclude = [
        "Breast - Mammary Tissue",
        "Vagina",
        "Uterus",
        "Fallopian Tube",
        "Ovary",
        "Testis",
        "Prostate",
    ]
    x = a.to_df().T.groupby(level=0).mean().reindex(coefs.columns).dropna().T
    _preds = dict()
    for tissue in tqdm(coefs.index):
        if tissue in sex_specific_tissues_to_exclude:
            continue
        model = sklearn.linear_model.Ridge()
        model.coef_ = coefs.loc[tissue, x.columns]
        model.feature_names_in_ = x.columns
        model.intercept_ = 0
        xz = (x - x.mean()) / x.std()
        _preds[tissue] = model.predict(xz)
    preds = pd.DataFrame(_preds, index=a.obs.index)
    preds.to_csv(
        config.results_dir
        / "predict_gaps_from_blood_expression"
        / f"{study}.predicted_from_blood_expression.{var}.predictions.csv",
    )

    preds = pd.read_csv(
        config.results_dir
        / "predict_gaps_from_blood_expression"
        / f"{study}.predicted_from_blood_expression.{var}.predictions.csv",
        index_col=0,
    )
    preds.index = preds.index.astype("str")

    # Normalize by group means
    r = preds.groupby(a.obs["harmonized:group"]).mean().T
    rn = (r.T - r[["Healthy"]].mean(1)).T
    rnn = rn - rn.loc["mean"]
    rnnn = (rnn.T - rnn.mean(1)).T

    # A few QC checks
    fig, ax = plt.subplots(figsize=(1.5, 1.5))
    am = a.obs.groupby("harmonized:group")["harmonized:age"].mean().loc[r.columns]
    pearsonr, pearsonp = pg.corr(
        *pd.concat([r.loc["mean"], am], axis=1)
        .dropna()
        .drop("Healthy")
        .T.values.astype(float)
    ).loc["pearson"][["r", "p-val"]]
    ax.scatter(r.loc["mean"].drop("Healthy"), am.drop("Healthy"))
    for disease in r.columns.drop("Healthy"):
        ax.annotate(disease, (r.loc["mean", disease], am.loc[disease]))
    ax.axvline(0, linestyle="--", color="grey")
    ax.set(
        xlabel="Mean estimated systemic age gap",
        ylabel="Mean cohort age",
        title=var + "\n" + f"r = {pearsonr:.2f}, p = {pearsonp:.2e}",
    )
    fig.savefig(
        config.results_dir
        / "predict_gaps_from_blood_expression"
        / f"{study}.predicted_from_blood_expression.{var}.age_vs_effect.scatter.svg",
        **config.figkws,
    )

    fig, ax = plt.subplots(figsize=(1.5, 1.5))
    ax.scatter(r.loc["mean"].drop("Healthy"), c.loc[r.columns].drop("Healthy"))
    pearsonr, pearsonp = pg.corr(r.loc["mean"].abs(), c.loc[r.columns]).loc["pearson"][
        ["r", "p-val"]
    ]
    for disease in r.columns:
        ax.annotate(disease, (r.loc["mean", disease], c.loc[disease]))
    ax.axvline(0, linestyle="--", color="grey")
    ax.set(
        yscale="log",
        xlabel="Mean estimated systemic age gap",
        ylabel="Sample size",
        title=var + "\n" + f"r = {pearsonr:.2f}, p = {pearsonp:.2e}",
    )
    fig.savefig(
        config.results_dir
        / "predict_gaps_from_blood_expression"
        / f"{study}.predicted_from_blood_expression.{var}.sample_size_vs_effect.scatter.svg",
        **config.figkws,
    )

    # Clustermaps
    kwargs = dict(cmap="RdBu_r", center=0, col_colors=c, dendrogram_ratio=0.05)
    g = clustermap(r, **kwargs)
    g.fig.savefig(
        config.results_dir
        / "predict_gaps_from_blood_expression"
        / f"{study}.predicted_from_blood_expression.{var}.group_comparison.clustermap.1.svg",
    )
    g = clustermap(rn, **kwargs)
    g.fig.savefig(
        config.results_dir
        / "predict_gaps_from_blood_expression"
        / f"{study}.predicted_from_blood_expression.{var}.group_comparison.clustermap.2.svg",
    )
    g = clustermap(rnn, **kwargs)

    g.fig.savefig(
        config.results_dir
        / "predict_gaps_from_blood_expression"
        / f"{study}.predicted_from_blood_expression.{var}.group_comparison.clustermap.3.svg",
    )
    g = clustermap(rnnn, **kwargs)
    g.fig.savefig(
        config.results_dir
        / "predict_gaps_from_blood_expression"
        / f"{study}.predicted_from_blood_expression.{var}.group_comparison.clustermap.4.svg",
    )

    # Normalize individual observations
    baseline = preds.loc[a.obs["harmonized:group"].isin(["Healthy"])].mean()
    dfn = preds.subtract(baseline, axis=1)
    dfn = dfn.sub(dfn["mean"], axis=0)
    dfn = dfn.sub(dfn.mean(axis=1), axis=0)
    dfnm = dfn.groupby(a.obs["harmonized:group"]).mean().T
    g = clustermap(dfnm, **kwargs)
    g.fig.savefig(
        config.results_dir
        / "predict_gaps_from_blood_expression"
        / f"{study}.predicted_from_blood_expression.{var}.group_comparison.clustermap.5.svg",
    )

    preds_organ = (
        preds.T.groupby(preds.columns.str.replace(" - .+", "", regex=True)).mean().T
    )

    clock_corr = preds_organ.corr()
    g = clustermap(
        clock_corr, figsize=(4.5, 4.5), cmap="RdBu_r", center=0, dendrogram_ratio=0.05
    )
    g.fig.savefig(
        config.results_dir
        / "predict_gaps_from_blood_expression"
        / f"{study}.predicted_from_blood_expression.clock_correlation.clustermap.svg",
        **config.figkws,
    )

    tech_vars = ["series_id", "harmonized:age", "harmonized:sex"]
    voi = preds_organ.columns.tolist()

    ap = sc.AnnData(preds_organ, a.obs.assign(mean_pred=preds_organ.mean(axis=1)))
    ap.obs["series_id"] = ap.obs["series_id"].cat.remove_unused_categories()
    sc.pp.scale(ap)
    sc.pp.pca(ap)
    sc.pp.neighbors(ap)
    sc.tl.umap(ap)
    sc.tl.draw_graph(ap)
    sc.tl.diffmap(ap)
    ap.obsm["X_diffmap"] = ap.obsm["X_diffmap"][:, 1:]
    from src.utils import mds, isomap

    if ap.shape[0] < 1500:
        mds(ap)
    isomap(ap)
    # sc.write(config.results_dir / "predict_gaps_from_blood_expression" / f"{study}.predicted_from_blood_expression.h5ad", ap)

    vmin = [None] + [0 for v in voi] + [None] * len(tech_vars)
    vmax = (
        [None] + [np.percentile(ap[:, v].X, 98) for v in voi] + [None] * len(tech_vars)
    )

    for emb in ap.obsm:
        fig = sc.pl.embedding(
            ap,
            basis=emb,
            color=["harmonized:group"] + voi + tech_vars,
            ncols=1,
            return_fig=True,
            vmin=vmin,
            vmax=vmax,
        )
        centroids = (
            pd.DataFrame(ap.obsm[emb][:, :2], index=ap.obs.index)
            .groupby(ap.obs["harmonized:group"])
            .mean()
        )
        for idx, xy in centroids.iterrows():
            fig.axes[0].text(xy[0], xy[1], idx)
        from src.utils import rasterize_scanpy

        rasterize_scanpy(fig)
        fig.savefig(
            config.results_dir
            / "predict_gaps_from_blood_expression"
            / f"{study}.predicted_from_blood_expression.{var}.{emb}.svg",
            **config.figkws,
        )

        fig, axes = plt.subplots(1, 2, figsize=(6 * 2, 4))
        for i, ax in enumerate(axes):
            colors = sns.color_palette("tab10")
            for color, d in zip(colors, centroids.index):
                sel = ap.obs.loc[ap.obs["harmonized:group"] == d].index
                tp = (
                    pd.DataFrame(ap[sel, :].obsm[emb][:, i], index=sel)
                    .squeeze()
                    .rename(i)
                )
                sns.kdeplot(tp, ax=ax, label=d, color=color)
                ax.axvline(tp.mean(), color=color, linewidth=2, linestyle="--")
            ax.legend()
        fig.savefig(
            config.results_dir
            / "predict_gaps_from_blood_expression"
            / f"{study}.predicted_from_blood_expression.{var}.{emb}.kde_per_group.svg",
            **config.figkws,
        )

    mean = ap.to_df().groupby(ap.obs["harmonized:group"]).mean()
    std = ap.to_df().groupby(ap.obs["harmonized:group"]).std()
    size = ap.to_df().groupby(ap.obs["harmonized:group"]).size()

    fig, ax = plt.subplots(figsize=(6, 4))
    m = mean.reset_index().melt(id_vars="harmonized:group")
    s = std.reset_index().melt(id_vars="harmonized:group")
    z = size.reset_index().melt(id_vars="harmonized:group")
    ax.scatter(m["value"], s["value"], s=z["value"].tolist() * mean.shape[1], alpha=0.5)
    for t in (
        m.sort_values("value").tail(10).index.tolist()
        + m.sort_values("value").head(10).index.tolist()
        + s.sort_values("value").tail(10).index.tolist()
        + s.sort_values("value").head(10).index.tolist()
    ):
        ax.annotate(
            m.loc[t, "harmonized:group"] + ", " + m.loc[t, "variable"],
            (m.loc[t, "value"], s.loc[t, "value"]),
        )
    ax.axhline(1, color="grey", linestyle="--")
    ax.axvline(0, color="grey", linestyle="--")
    ax.set(xlabel="mean", ylabel="std")
    fig.savefig(
        config.results_dir
        / "predict_gaps_from_blood_expression"
        / f"{study}.predicted_from_blood_expression.{var}.mean_vs_std.svg",
    )

    # T-tests
    _tests = list()
    for o in mean.columns:
        for d in mean.index:
            _tests.append(
                pg.ttest(
                    preds_organ.loc[a.obs["harmonized:group"] == d, o],
                    preds_organ.loc[a.obs["harmonized:group"] == "Healthy", o],
                    alternative="greater",
                ).assign(Organ=o, Disease=d)
            )

    tests = pd.concat(_tests)
    tests["p-adj"] = pg.multicomp(tests["p-val"].values, method="fdr_bh")[1]
    tests.to_csv(
        config.results_dir
        / "predict_gaps_from_blood_expression"
        / f"{study}.predicted_from_blood_expression.{var}.all_diseases.tests.csv",
        index=False,
    )
    tests = pd.read_csv(
        config.results_dir
        / "predict_gaps_from_blood_expression"
        / f"{study}.predicted_from_blood_expression.{var}.all_diseases.tests.csv"
    )
    p = tests.pivot(index="Disease", columns="Organ", values="p-adj").reindex_like(mean)
    v = abs(mean).max().max()
    v += v * 0.1
    tp = mean - mean.loc["Healthy"]
    tp.loc["Healthy", :] = np.random.rand(tp.shape[1]) * 0.1
    g = clustermap(
        tp,
        cmap="RdBu_r",
        vmin=-v,
        vmax=v,
        metric="euclidean",
        pvalues=p,
        dendrogram_ratio=0.05,
        figsize=(8, 4),
        first_pvalue_threshold=0.005,
        second_pvalue_threshold=1e-8,
    )
    g.fig.savefig(
        config.results_dir
        / "predict_gaps_from_blood_expression"
        / f"{study}.predicted_from_blood_expression.{var}.all_diseases.clustermap.with_healthy.svg",
        **config.figkws,
    )
    tp = tp.loc[:, (p < 0.005).any()]
    g = clustermap(
        tp,
        cmap="RdBu_r",
        vmin=-v,
        vmax=v,
        metric="euclidean",
        pvalues=p.reindex_like(tp),
        dendrogram_ratio=0.05,
        figsize=(8, 4),
        first_pvalue_threshold=0.005,
        second_pvalue_threshold=1e-8,
    )
    g.fig.savefig(
        config.results_dir
        / "predict_gaps_from_blood_expression"
        / f"{study}.predicted_from_blood_expression.{var}.all_diseases.clustermap.with_healthy.only_sig.svg",
        **config.figkws,
    )

    g = clustermap(
        (mean - mean.loc["Healthy"]).drop("Healthy", axis=0),
        cmap="RdBu_r",
        vmin=-v,
        vmax=v,
        metric="euclidean",
        pvalues=p.drop("Healthy", axis=0),
        dendrogram_ratio=0.05,
        figsize=(8, 4),
        first_pvalue_threshold=0.005,
        second_pvalue_threshold=1e-8,
    )
    g.fig.savefig(
        config.results_dir
        / "predict_gaps_from_blood_expression"
        / f"{study}.predicted_from_blood_expression.{var}.all_diseases.clustermap.no_healthy.svg",
        **config.figkws,
    )

    mean = mean.drop("Healthy", axis=0)
    p = p.drop("Healthy", axis=0)
    size = size.drop("Healthy", axis=0)

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.scatter(
        mean.melt()["value"],
        -np.log10(p.melt()["value"]),
        s=[y for x in size for y in [x] * mean.shape[1]],
        alpha=0.5,
        c=mean.melt()["value"],
        cmap="coolwarm",
        vmin=-v,
        vmax=v,
    )
    sel = p.reset_index().melt(id_vars="harmonized:group").query("value < 1e-8")
    for idx, t in sel.iterrows():
        ax.text(
            mean.melt().loc[idx, "value"],
            -np.log10(t["value"]),
            t["harmonized:group"] + " - " + t["variable"],
        )
    ax.axvline(0, linestyle="--", color="grey")
    ax.axhline(8, linestyle="--", color="grey")
    ax.set(xlabel="mean", ylabel="-log10(p)")
    fig.savefig(
        config.results_dir
        / "predict_gaps_from_blood_expression"
        / f"{study}.predicted_from_blood_expression.{var}.all_diseases.volcano_scatter.svg",
        **config.figkws,
    )

    fig, axes = plt.subplots(5, 4, figsize=(1.5 * 4, 1.5 * 5), sharex=True, sharey=True)
    for ax, o in zip(axes.flatten(), mean.columns):
        ax.scatter(
            mean[o],
            -np.log10(p[o]),
            s=size,
            alpha=0.5,
            c=mean[o],
            cmap="coolwarm",
            vmin=-v,
            vmax=v,
        )
        sel = p[o].loc[p[o] < 1e-8]
        for idx in sel.index:
            ax.text(mean.loc[idx, o], -np.log10(p.loc[idx, o]), idx, ha="right")
        ax.axvline(0, linestyle="--", color="grey")
        ax.axhline(8, linestyle="--", color="grey")
        ax.set(xlabel="mean", ylabel="-log10(p)", title=o)
    fig.savefig(
        config.results_dir
        / "predict_gaps_from_blood_expression"
        / f"{study}.predicted_from_blood_expression.{var}.all_diseases.volcano_scatter.per_organ.svg",
        **config.figkws,
    )

    groups = {
        d: {"groups": ["Healthy"] + [d], "visualize": mean.columns.tolist()}
        for d in mean.index
        if d != "Healthy"
    }
    for group, data in groups.items():
        q = (
            a.obs.loc[
                a.obs["harmonized:group"].isin(data["groups"]), ["harmonized:group"]
            ]
            .join(preds_organ)
            .melt(id_vars="harmonized:group")
            .query("variable.isin(@data['visualize'])")
        )
        q["harmonized:group"] = q["harmonized:group"].cat.remove_unused_categories()
        q["harmonized:group"] = q["harmonized:group"].cat.reorder_categories(
            data["groups"], ordered=True
        )
        print(q["harmonized:group"].value_counts())
        q["Organ"] = q["variable"].str.replace(r" - .+", "", regex=True)
        q = q.drop(["variable"], axis=1)

        fig, ax = plt.subplots()
        sns.boxenplot(
            data=q,
            hue="harmonized:group",
            x="value",
            y="Organ",
            orient="horiz",
            ax=ax,
        )
        ax.axvline(0, linestyle="--", color="grey")
        ax.set(xlabel="Mean age gap", ylabel="Organ")
        fig.savefig(
            config.results_dir
            / "predict_gaps_from_blood_expression"
            / f"{study}.predicted_from_blood_expression.{var}.group_comparison.{group}.boxenplot.svg",
            **config.figkws,
        )

        x = (
            a.obs[["harmonized:group"]]
            .query(f"`harmonized:group`.isin({['Healthy'] + data['groups']})")
            .join(preds_organ)
        )
        x["harmonized:group"] = x["harmonized:group"].cat.remove_unused_categories()
        x["harmonized:group"] = x["harmonized:group"].cat.reorder_categories(
            data["groups"], ordered=True
        )
        fig, stats = swarmboxenplot(
            data=x,
            x="harmonized:group",
            y=preds_organ.columns,
            test="t-test",
            test_upper_threshold=0.001,
            test_lower_threshold=0.001,
            swarm=False,
        )
        for ax in fig.axes:
            ax.axhline(0, linestyle="--", color="grey")
        fig.savefig(
            config.results_dir
            / "predict_gaps_from_blood_expression"
            / f"{study}.predicted_from_blood_expression.{var}.group_comparison.{group}.swarmboxenplot.svg",
            **config.figkws,
        )
        stats.to_csv(
            config.results_dir
            / "predict_gaps_from_blood_expression"
            / f"{study}.predicted_from_blood_expression.{var}.group_comparison.{group}.swarmboxenplot.csv",
            index=False,
        )

    # Classification metrics
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        average_precision_score,
    )

    threshold = 1
    _scores = list()
    for d in tqdm(mean.index.drop("Healthy")):
        sel = a.obs["harmonized:group"].isin([d, "Healthy"])
        xx = preds_organ.loc[sel]
        y = a.obs.loc[sel, "harmonized:group"] == d
        for o in mean.columns:
            r = pd.Series()
            r["Accuracy"] = accuracy_score(xx[o] > threshold, y)
            r["Precision"] = precision_score(xx[o] > threshold, y)
            r["Recall"] = recall_score(xx[o] > threshold, y)
            r["F1"] = f1_score(xx[o] > threshold, y)
            r["ROC AUC"] = roc_auc_score(xx[o] > threshold, y)
            r["AP"] = average_precision_score(xx[o] > threshold, y)
            r["Organ"] = o
            r["Disease"] = d
            _scores.append(r)
    scores = pd.concat(_scores, axis=1).T
    scores.to_csv(
        config.results_dir
        / "predict_gaps_from_blood_expression"
        / f"{study}.predicted_from_blood_expression.{var}.all_diseases.classification_metrics.{threshold=}.csv",
    )
    # scores.sort_values("AP").tail(60)

    g = clustermap(
        scores.melt(id_vars=["Disease", "Organ"])
        .pivot_table(
            index=["variable", "Disease"],
            columns="Organ",
            values="value",
            aggfunc="max",
        )
        .astype(float),
        row_cluster=False,
        yticklabels=True,
        center=0.5,
        cmap="coolwarm",
        dendrogram_ratio=0.05,
        figsize=(7, 10),
    )
    g.fig.savefig(
        config.results_dir
        / "predict_gaps_from_blood_expression"
        / f"{study}.predicted_from_blood_expression.{var}.all_diseases.classification_metrics.{threshold=}.clustermap.svg",
    )

    for metric in scores.columns.drop(["Organ", "Disease"]):
        ppp = (
            scores[[metric, "Organ", "Disease"]]
            .pivot_table(
                index=["Disease"],
                columns="Organ",
                values=metric,
            )
            .astype(float)
            .dropna()
        )
        lims = (
            {"vmin": 0, "vmax": 0.5}
            if ppp.max().max() < 0.5
            else {"vmin": 0, "vmax": 1, "cmap": "coolwarm"}
        )
        g = clustermap(
            ppp,
            dendrogram_ratio=0.05,
            cbar_kws={"label": metric},
            figsize=(6, 3),
            annot=True,
            annot_kws={"size": 4},
            **lims,
        )
        g.fig.savefig(
            config.results_dir
            / "predict_gaps_from_blood_expression"
            / f"{study}.predicted_from_blood_expression.{var}.all_diseases.{metric}.{threshold=}clustermap.svg",
        )
